import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime

# ==============================================================================
# 0. PATH CONFIGURATION AND PROJECT MODULE IMPORTS
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
  sys.path.insert(0, CURRENT_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.spatial.transform import Rotation as R

# Import only the modules that actually exist in candidate.py
try:
  from candidate import CandidateGenerator, MPCCostFunction, NeuralMPCController
except ImportError as e:
  raise ImportError(
      f"❌ No se pudo importar desde 'candidate.py'. Revisa que el archivo esté en {CURRENT_DIR}"
  ) from e

from continuum_robot.config import robot_config as cfg
from continuum_robot.data.logger import CSVLogger
from continuum_robot.hardware.dynamixel import create_controller

try:
  from continuum_robot.hardware.galga import PhidgetForceController
except Exception:
  PhidgetForceController = None

try:
  from NatNetClient import NatNetClient
except Exception:
  NatNetClient = None


# ==============================================================================
# 1. LSTM OBSERVER ARCHITECTURE
# ==============================================================================
class SoftRobotLSTM(nn.Module):

  def __init__(
      self,
      input_size=17,
      hidden_size=128,
      num_layers=2,
      output_size=3,
      dropout=0.2,
  ):
    super(SoftRobotLSTM, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    h0 = torch.zeros(
        self.num_layers, x.size(0), self.hidden_size, device=x.device
    )
    c0 = torch.zeros(
        self.num_layers, x.size(0), self.hidden_size, device=x.device
    )

    out, _ = self.lstm(x, (h0, c0))
    last_step = out[:, -1, :]
    out_regularized = self.dropout(last_step)
    return self.fc(out_regularized)

class MPCDirectPredictor(nn.Module):

  def __init__(
      self,
      input_hist_dim,
      u_cand_dim=4,
      hidden_size=128,
      num_layers=2,
      t_out=10,
      output_dim=3,
      dropout=0.1,
  ):
    super(MPCDirectPredictor, self).__init__()
    self.encoder_lstm = nn.LSTM(
        input_size=input_hist_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )

    fc_input_dim = hidden_size + u_cand_dim
    out_flat_dim = t_out * output_dim

    self.mlp = nn.Sequential(
        nn.Linear(fc_input_dim, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, out_flat_dim),
    )
    self.t_out = t_out
    self.output_dim = output_dim

  def forward(self, x_hist, u_cand):
    out, _ = self.encoder_lstm(x_hist)
    last_hidden = out[:, -1, :]
    combined = torch.cat([last_hidden, u_cand], dim=1)
    y_pred_flat = self.mlp(combined)
    return y_pred_flat.view(-1, self.t_out, self.output_dim)
  
# ==============================================================================
# 2. SMART MODEL / CHECKPOINT LOADER
# ==============================================================================
def _cargar_modelo_pytorch(path, device, tipo_modelo='predictor'):
  """Smart loader for PyTorch models (.pth / .pt).

  Returns (model, window_size). window_size is the T_IN window with which
  that particular model was TRAINED (it may differ between predictor and
  observer -- see checkpoint['window_size']/['t_in']); None if the
  checkpoint does not declare it (TorchScript, bare nn.Module, or an old
  state_dict without that key), in which case the caller must use a fallback.
  """
  # 1. Try TorchScript JIT
  try:
    return torch.jit.load(path, map_location=device).to(device), None
  except Exception:
    pass

  # 2. Load with torch.load
  checkpoint = torch.load(path, map_location=device)

  # Case A: It is already a complete nn.Module object
  if isinstance(checkpoint, torch.nn.Module):
    return checkpoint.to(device), None

  # Case B: It is a Checkpoint dictionary
  if isinstance(checkpoint, dict):
    print(
        f"📦 Checkpoint detectado en '{os.path.basename(path)}'."
        ' Reconstruyendo red...'
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    window_size = checkpoint.get('window_size', checkpoint.get('t_in'))

    if tipo_modelo == 'predictor':
      # Extract the exact parameters it was saved with
      model = MPCDirectPredictor(
          input_hist_dim=checkpoint['input_hist_dim'],
          u_cand_dim=checkpoint['u_cand_dim'],
          hidden_size=checkpoint['hidden_size'],
          num_layers=checkpoint['num_layers'],
          t_out=checkpoint['t_out'],
          output_dim=checkpoint['output_dim'],
      )
    elif tipo_modelo == 'observador':
      model = SoftRobotLSTM(
          input_size=checkpoint.get('input_size', 17),
          hidden_size=checkpoint.get('hidden_size', 128),
          num_layers=checkpoint.get('num_layers', 2),
          output_size=checkpoint.get('output_size', 3),
          dropout=checkpoint.get('dropout', 0.2),
      )

    model.load_state_dict(state_dict)
    return model.to(device), window_size

  raise TypeError(
      f"❌ Formato de archivo no reconocido en {path}. Tipo: {type(checkpoint)}"
  )


# ==============================================================================
# 3. TARGET POINTS AND INITIAL CONFIGURATION (IN MM)
# ==============================================================================
WAYPOINTS_MM = np.array(
    [
        [101, 262, -44],
        [6, 291.32, -57.14],
        [14.9, 304.4, -35.5],
        #[-153, 226, 81],
        [74, 305, 46],
        #[-63.6, 278.9, -34],
        [21, 250, -75],
    ],
    dtype=np.float32,
)

# 🧪 Waypoints for run_experimento_repeticiones: the 5 from WAYPOINTS_MM plus
# the 2 edge cases that were commented out above -- they are exactly the ones
# that in production showed the limits/kinematic redundancy problem
# (see USAR_SEMILLAS_FIJAS/SEED_COMMANDS_TICKS further below), so it is of
# interest to include them explicitly in the validation with repetitions.
WAYPOINTS_EXPERIMENTO_MM = np.array(
    [
        #1[101, 262, -44],
        #2[6, 291.32, -57.14],
        #3[14.9, 304.4, -35.5],
        #4[74, 305, 46],
        #[-74, 305, -46],
        #5[21, 250, -75],#5
        #6[-40.57,310.78,2.48],#6
        #7[-22.35, 224.55, -180],#7
        #8[-153, 226, 81],#8
        #9[-63.6, 278.9, -34],#9
        [-189.4, 221.2, 82.2],
        [160.9, 244.6, -83.1],
        [68.6, 233.4, 151.2],
        [-74.2, 237, -171.4],
        [-57, 277.3, 1.9],
        [40.2, 278.7, -17.3],
        [-0.9, 270.9, 71.6],
        [-16, 282.3, -23.8],
        [105.4, 288.9, -38.7],
        [-80, 247, 23.1],
        [-36, 275.2, -137.7],
        [93.6, 280.7, 76.1],
        [77.7, 303.3, 51.2],
        [-14.7, 304.4, -80.8],
        [53.9, 294.6, 6],
        [-112, 264.1, 100.6],
        [-105.1, 257.1, -120],
        [73.4, 269.4, -115.8],
        [-109.2, 248.5, -52.2],
        [-50.8, 269.5, 96.9]
    ],
    dtype=np.float32,
)

HOME_MOTORS = {f'm{mid}': cfg.HOME_POSITION[mid] for mid in cfg.MOTOR_IDS}

# 🌱 Seed commands per TARGET -- DISABLED (USAR_SEMILLAS_FIJAS=False).
# They were used to diagnose whether the points [-153,226,81]/[-63.6,278.9,-34]
# were reachable. Now that the CEM multi-start exists (see
# NeuralMPCController.optimize), seeding by hand would be "cheating" when
# evaluating whether the multi-start alone can find good solutions without
# help -- the dictionary is kept as a reference/comparison, but it is not
# used to control the robot. Set to True to re-enable it.
#
# FIXED (18/08): the original m4 values (1104/1111) had been
# taken from a frame IN TRANSITION (the motor still moving between
# consecutive readings of waypoints_relativos_largo.csv, not staying
# the same for more than one row), not from the truly settled value -- hence
# the error reported earlier (5.8mm/~13mm). Verified against the real plateaus
# of the CSV (m4 constant for >=10 consecutive rows, motor stopped): the
# settled value closest to each target gives ~1.45mm in both cases.
USAR_SEMILLAS_FIJAS = True
SEED_COMMANDS_TICKS = {
    (-153.0, 226.0, 81.0): [1876, 1406, 2130, 1079],  # settled: 1.45mm error
    (-63.6, 278.9, -34.0): [1601, 1682, 1481, 1080],  # settled: 1.45mm error
    (-22.35, 224.55, -179.51): [1599, 2229, 842, 1076],  # settled: 0.07mm error
    # (-40.57, 310.78, 2.48) has NO genuine plateau nearby in waypoints_relativos_largo.csv
    # (the "exact" reading that seemed to match was a frame in transition, not a
    # truly settled point) -- no seed is added for that one, it already reached 2/3
    # with the normal CEM before loosening the limits barrier.
}
SEED_TARGET_TOLERANCIA_MM = 1.0
SEED_TOLERANCIA_TICKS = 15.0
SEED_TIMEOUT_S = 15.0


def _buscar_semilla_para_target(target_wp, tolerancia_mm=SEED_TARGET_TOLERANCIA_MM):
  """Returns the seed command configured for this target (by cartesian
  position, with tolerance), or None if none is defined for it."""
  target_np = np.asarray(target_wp, dtype=np.float64)
  for target_key, ticks in SEED_COMMANDS_TICKS.items():
    if np.linalg.norm(np.asarray(target_key, dtype=np.float64) - target_np) <= tolerancia_mm:
      return ticks
  return None


# 🧭 Intermediate waypoints: if the direct jump to the next waypoint is
# large, the CEM/Jacobian (LOCAL methods) may get trapped in the
# kinematic redundancy branch closest to where the robot is, instead
# of the correct branch (confirmed in production: for a specific target,
# both the CEM and the Jacobian insisted on pushing a motor to its limit,
# while the real combination that reached it in the training data
# was on the other side of home). Chopping the jump into small steps
# preserves continuity in actuator space between consecutive cycles
# (the same technique used in Jacobian IK for redundant systems)
# without needing a reference table. The intermediate path does not need
# to imitate the real shape (arc) of the robot -- only that each
# sub-goal be close to the previous one; the robot curves however is
# natural for it to reach each one.
DISTANCIA_MAX_SALTO_MM = 60.0
PASO_SUBOBJETIVO_MM = 50.0
TOLERANCIA_SUBOBJETIVO_MM = 25.0


def _generar_subobjetivos(pos_actual_mm, target_mm, paso_mm=PASO_SUBOBJETIVO_MM,
                           salto_min_mm=DISTANCIA_MAX_SALTO_MM):
  """Intermediate points in a straight line between pos_actual_mm and target_mm, not
  including the final point (that one is still treated as the real waypoint, with
  its normal tolerance/dwell). Empty list if the jump is not large."""
  pos_actual_mm = np.asarray(pos_actual_mm, dtype=np.float32)
  target_mm = np.asarray(target_mm, dtype=np.float32)
  dist = float(np.linalg.norm(target_mm - pos_actual_mm))
  if dist <= salto_min_mm:
    return []
  n_pasos = max(1, int(np.ceil(dist / paso_mm)))
  return [
      pos_actual_mm + (target_mm - pos_actual_mm) * (i / n_pasos)
      for i in range(1, n_pasos)
  ]


# ==============================================================================
# 4. REAL-TIME CAUSAL FILTERING
# ==============================================================================
class RealTimeCausalFilter:

  def __init__(self, cutoff_hz, fs_hz=60.0, order=2, num_channels=1):
    self.nyquist = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / self.nyquist
    self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
    self.num_channels = num_channels
    zi_single = lfilter_zi(self.b, self.a)
    self.zi = np.tile(zi_single, (num_channels, 1)).T  # Shape: (order, num_channels)

  def filter(self, x):
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.zeros(self.num_channels, dtype=np.float64)
    for i in range(self.num_channels):
      out, zf = lfilter(self.b, self.a, [x[i]], zi=self.zi[:, i])
      y[i] = out[0]  # Explicitly extract scalar
      self.zi[:, i] = zf
    return y


# ==============================================================================
# 5. DYNAMIC SCALER AND NORMALIZER
# ==============================================================================
class InputScaler:

  def __init__(self, metadata_json, device='cuda'):
    self.device = device
    x_trans = metadata_json['X_transformer']
    self.feature_names = list(x_trans.keys())

    mins = [x_trans[k]['min_t'] for k in self.feature_names]
    maxs = [x_trans[k]['max_t'] for k in self.feature_names]

    self.min_tensor = torch.tensor(mins, device=device, dtype=torch.float32)
    self.max_tensor = torch.tensor(maxs, device=device, dtype=torch.float32)

  def scale(self, feature_dict):
    raw_vals = [feature_dict[k] for k in self.feature_names]
    raw_tensor = torch.tensor(raw_vals, device=self.device, dtype=torch.float32)
    scaled = -1.0 + 2.0 * (raw_tensor - self.min_tensor) / (
        self.max_tensor - self.min_tensor + 1e-8
    )
    return scaled


# ==============================================================================
# 6. MAIN CLOSED-LOOP MPC CONTROL SYSTEM
# ==============================================================================
class SoftRobotMPCSystem:

  def __init__(
      self,
      predictor_path,
      observer_path,
      metadata_pred_path,
      metadata_obs_path=None,
      device='cuda',
      simulate=True,
      use_optitrack=True,
      port=None,
      baud=None,
      action_hold_cycles=4,
      delta_min_ticks=20,
      ema_alpha=1.0,
      setpoint_lock_pct=0.0,
      num_samples=300,
      w_limite=0.05,
      margen_confort_ticks=100.0,
      anclas_fijas_ticks=None,
      modo_control='hibrido',
      k_iters_jacobiano=2,
  ):
    self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'🖥️ Ejecutando modelos en: {self.device}')
    self.simulate = simulate
    self.use_optitrack = use_optitrack
    self.port = port or cfg.SERIAL_PORT
    self.baud = baud or cfg.BAUDRATE
    self.hardware_lock = threading.Lock()
    self.t_prev = None

    # 🎯 Action Hold (decimated control): sensing runs at 60 Hz but the MPC
    # only recomputes/sends a command every N cycles, giving the cable time to
    # tense and deform the silicone before evaluating the next step.
    self.action_hold_cycles = max(1, int(action_hold_cycles))
    # Minimum jump threshold (ticks): if |command - last sent| is smaller,
    # the movement is absorbed by the cable slack and is not resent.
    self.delta_min_ticks = delta_min_ticks
    self.ultimo_comando_ticks = None

    # 🔺 Anti-deadband accumulator (integrator effect) per motor: if the CEM
    # consistently asks for a correction smaller than delta_min_ticks,
    # without this it is lost EVERY cycle (memoryless comparison) and the system
    # gets stuck unable to refine the last stretch (seen in production:
    # it could not get below ~15mm down to ~8mm). A fraction of the requested
    # correction is accumulated each cycle until it exceeds the threshold, and then it is
    # released all at once. Anti-windup: cap on the accumulator (see _reset_hist_acum) and
    # it is reset when the waypoint changes.
    self.acumulador_hist_ticks = [0.0, 0.0, 0.0, 0.0]
    self.ganancia_integral_hist = 0.3
    self._ultimo_target_wp_hist = None

    # 🌊 Exponential smoothing (LERP) of the adopted command, in scaled space
    # [-1,1]: u_sent(t) = alpha*u_opt + (1-alpha)*u_sent(t-1). DISABLED
    # by default (alpha=1.0 -> u_sent=u_opt with no mixing) because the CEM already
    # averages/smooths internally (elite mean + std with memory); EMA
    # stacked ON TOP of that reduces the change to something so small that the hysteresis
    # (delta_min_ticks) always discards it -> the command freezes
    # completely (seen in production: Command and Motor Position never change).
    self.ema_alpha = float(ema_alpha)

    # 🔒 Setpoint Lock (Waypoint Tracking / progress threshold): a new optimizer
    # target is not adopted until the real position has covered
    # `setpoint_lock_pct` (e.g. 0.8 = 80%) of the distance between the origin and
    # the current target. DISABLED by default (0.0): if the robot fails
    # to cover that % (friction/slack/insufficient torque), the system stops
    # resending commands and stays blocked forever (nothing updates the
    # origin/target again except a successful adoption). With EMA and the CEM's
    # `std` memory it should no longer be needed: both already prevent the
    # optimizer from jumping abruptly from one target to another.
    self.setpoint_lock_pct = float(setpoint_lock_pct)
    self.target_lock_origen_ticks = None

    self.num_samples = int(num_samples)
    self._home_ticks_cache = None  # cached on the 1st call (home_m is fixed per run)

    # Hardware handles (created in _setup_hardware)
    self.controlador_dynamixel = None
    self.streaming_client = None
    self.controlador_phidget = None
    self._csv_logger = None

    # 🧵 Background Dynamixel sensing thread (same pattern as main.py):
    # reads the serial bus in its own loop paced at 60 Hz; the control loop
    # never blocks on the read, it only reads the variables already cached below.
    self._sensor_thread_running = False
    self._sensor_thread_obj = None

    # 🚀 Shared state of the OptiTrack rigid bodies (updated async)
    self.p_base = [0.0, 0.0, 0.0]
    self.q_base = [0.0, 0.0, 0.0, 1.0]
    self.p_efector = [0.0, 0.0, 0.0]
    self.q_efector = [0.0, 0.0, 0.0, 1.0]
    # Timestamp of the last frame received per rigid body (None = never).
    # Used to distinguish "OptiTrack at the true origin" from "OptiTrack never connected".
    self._t_ultimo_frame_base = None
    self._t_ultimo_frame_efector = None

    # Last valid motor reading (fallback on communication failures)
    self.ultimas_posiciones_validas = [cfg.HOME_POSITION[mid] for mid in cfg.MOTOR_IDS]
    self.ultimos_torques_validos = [0] * len(cfg.MOTOR_IDS)
    self.ultimas_fuerzas_validas = [0.0] * 4

    # 1. Load JSON Metadata
    print(f'📄 Cargando metadatos predictor: {metadata_pred_path}')
    with open(metadata_pred_path, 'r') as f:
      self.meta_pred = json.load(f)

    if metadata_obs_path and os.path.exists(metadata_obs_path):
      print(f'📄 Cargando metadatos observador: {metadata_obs_path}')
      with open(metadata_obs_path, 'r') as f:
        self.meta_obs = json.load(f)
    else:
      self.meta_obs = self.meta_pred

    self.t_in = self.meta_pred.get('t_in', 45)
    self.t_out = self.meta_pred.get('t_out', 10)
    print(
        f'⚙️ Configuración temporal detectada: t_in={self.t_in} pasos,'
        f' t_out={self.t_out} pasos'
    )

    # 2. Load PyTorch Models
    print(f'🧠 Cargando predictor PINN desde: {predictor_path}')
    self.predictor, _ = _cargar_modelo_pytorch(
        predictor_path, self.device, tipo_modelo='predictor'
    )
    if hasattr(self.predictor, 'eval'):
      self.predictor.eval()

    print(f'👁️ Cargando observador LSTM desde: {observer_path}')
    self.observer, window_size_obs = _cargar_modelo_pytorch(
        observer_path, self.device, tipo_modelo='observador'
    )
    if hasattr(self.observer, 'eval'):
      self.observer.eval()

    # The observer may have been trained with a T_IN window different from
    # the predictor's (checkpoint['window_size']); if the checkpoint does not
    # declare it (old formats), fall back to the value with which the currently
    # deployed model is known to have been trained (90).
    if window_size_obs is None:
      window_size_obs = self.meta_obs.get('window_size', 90)
    self.t_in_obs = int(window_size_obs)
    print(f'⚙️ Ventana del observador: t_in_obs={self.t_in_obs} pasos')

    # Parameters to de-scale the observer output (rel_x, rel_y, rel_z)
    # when OptiTrack is not available (--no-optitrack)
    y_trans_obs = self.meta_obs.get('Y_transformer')
    if y_trans_obs:
      self.min_y_obs = torch.tensor(
          [y_trans_obs['rel_x']['min_t'], y_trans_obs['rel_y']['min_t'], y_trans_obs['rel_z']['min_t']],
          device=self.device,
      )
      self.max_y_obs = torch.tensor(
          [y_trans_obs['rel_x']['max_t'], y_trans_obs['rel_y']['max_t'], y_trans_obs['rel_z']['max_t']],
          device=self.device,
      )
    else:
      self.min_y_obs = None
      self.max_y_obs = None

    # 3. Initialize Normalizer and MPC
    self.scaler = InputScaler(self.meta_pred, device=self.device)
    self.mpc = NeuralMPCController(
        model_predictor=self.predictor,
        metadata_json=self.meta_pred,
        num_samples=self.num_samples,
        device=self.device,
        w_limite=w_limite,
        margen_confort_ticks=margen_confort_ticks,
        anclas_fijas_ticks=anclas_fijas_ticks,
        modo_control=modo_control,
        k_iters_jacobiano=k_iters_jacobiano,
    )

    # 4. Causal Filters
    self.filter_pos = RealTimeCausalFilter(
        cutoff_hz=6.0, fs_hz=60.0, num_channels=3
    )
    self.filter_torques = RealTimeCausalFilter(
        cutoff_hz=3.5, fs_hz=60.0, num_channels=4
    )
    self.filter_tensions = RealTimeCausalFilter(
        cutoff_hz=3.5, fs_hz=60.0, num_channels=4
    )

    # 5. Dynamic Buffer -- sized for the longer of the two models
    # (predictor: self.t_in, observer: self.t_in_obs), each one consumes
    # only the final stretch it needs (see _estimar_p_efector_observador and
    # run_control_loop).
    self._buffer_len = max(self.t_in, self.t_in_obs)
    self.buffer = deque(maxlen=self._buffer_len)
    for _ in range(self._buffer_len):
      self.buffer.append(
          torch.zeros(len(self.scaler.feature_names), device=self.device)
      )

    # Logging
    self.log_queue = queue.Queue()
    self.logging_running = False

  def _procesar_lecturas_cinematicas(
    self,
    p_base,
    q_base,
    p_efector,
    q_efector,
    real_m,
    couple_m,
    tension_m,
    meta_m,
    home_m,
    t_actual=None,  # Optional timestamp
    filtrar_posicion=True,
  ):
    # 1. Dynamic computation of delta_t (60 Hz by default on the 1st step)
    if t_actual is None:
        t_actual = time.perf_counter()

    if getattr(self, 't_prev', None) is None:
        delta_t = 1.0 / 60.0  # ~0.01667 seconds
    else:
        delta_t = t_actual - self.t_prev

    self.t_prev = t_actual

    # 2. Relative kinematics and Quaternions
    r_b = R.from_quat(q_base)
    r_e = R.from_quat(q_efector)

    p_rel = r_b.inv().apply(np.array(p_efector) - np.array(p_base))
    r_rel = r_b.inv() * r_e
    q_rel = r_rel.as_quat()

    # 3. Causal Filtering
    # The position is ONLY filtered (6 Hz Butterworth) when the source is a
    # raw reading with jitter (real OptiTrack or the synthetic noise of the
    # simulate mode). When the source is the output of the LSTM observer
    # (--no-optitrack), that signal WAS ALREADY trained against OptiTrack targets
    # smoothed at 6 Hz — filtering it again would be double filtering and would add
    # unnecessary phase delay in the control loop.
    if filtrar_posicion:
      pos_filt = self.filter_pos.filter(p_rel)
    else:
      pos_filt = np.asarray(p_rel, dtype=np.float64).flatten()
    torques_filt = self.filter_torques.filter(couple_m)
    tensiones_filt = self.filter_tensions.filter(tension_m)

    # 4. Build dictionary (Adding 'delta_t')
    dict_data = {
        'rel_x': pos_filt[0],
        'rel_y': pos_filt[1],
        'rel_z': pos_filt[2],
        'rel_qx': q_rel[0],
        'rel_qy': q_rel[1],
        'rel_qz': q_rel[2],
        'rel_qw': q_rel[3],
        'delta_t': delta_t,  # <-- Key requested by the scaler!
    }

    for i in range(1, 5):
        dict_data[f'delta_real_m{i}'] = float(real_m[i - 1] - home_m[f'm{i}'])
        dict_data[f'delta_meta_m{i}'] = float(meta_m[i - 1] - home_m[f'm{i}'])
        dict_data[f'couple_m{i}'] = torques_filt[i - 1]
        dict_data[f'tension_m{i}'] = tensiones_filt[i - 1]

    return dict_data, pos_filt

  def _on_rigid_body_frame(self, new_id, position, rotation):
    """PRODUCER: receives asynchronous OptiTrack frames (base=1, effector=2)."""
    if new_id == 1:
      self.p_base = position
      self.q_base = rotation
      self._t_ultimo_frame_base = time.time()
    elif new_id == 2:
      self.p_efector = position
      self.q_efector = rotation
      self._t_ultimo_frame_efector = time.time()

  def _optitrack_pose_relativa_mm(self, max_age=0.5):
    """Relative position (effector with respect to base) measured by OptiTrack, in mm.

    Returns None if a frame from both rigid bodies was never received or if
    the last reading is older than `max_age` seconds (Motive down/not
    transmitting), so as not to confuse "no data" with a real [0,0,0].
    """
    if self._t_ultimo_frame_base is None or self._t_ultimo_frame_efector is None:
      return None
    t_now = time.time()
    if (t_now - self._t_ultimo_frame_base) > max_age or (t_now - self._t_ultimo_frame_efector) > max_age:
      return None
    r_b = R.from_quat(self.q_base)
    p_rel = r_b.inv().apply(np.array(self.p_efector) - np.array(self.p_base))
    return p_rel * 1000.0

  @torch.inference_mode()
  def _estimar_p_efector_observador(self):
    """Estimates the relative position of the effector (m) with the LSTM observer.

    Used when --no-optitrack is active: closes the loop with the
    observer's prediction instead of the real OptiTrack measurement.
    """
    x_hist_tensor = torch.stack(list(self.buffer)[-self.t_in_obs:]).unsqueeze(0)
    pred = self.observer(x_hist_tensor).squeeze(0)
    if self.min_y_obs is not None:
      pos_m = self.min_y_obs + (pred + 1.0) * (self.max_y_obs - self.min_y_obs) / 2.0
    else:
      pos_m = pred
    return pos_m.cpu().numpy()

  def _home_ticks_tensor(self, home_m):
    """Home [4] tensor in ticks, cached (home_m is fixed during the run)."""
    if self._home_ticks_cache is None:
      self._home_ticks_cache = torch.tensor(
          [home_m[f'm{i}'] for i in range(1, 5)], device=self.device, dtype=torch.float32
      )
    return self._home_ticks_cache

  def _ticks_absolutos_desde_escalado(self, u_scaled_tensor, home_m):
    """Converts the normalized control vector [-1, 1] to

    absolute motor positions (ticks), using the per-motor ranges of the candidate
    generator and the reference HOME position.
    """
    ranges = self.mpc.candidate_generator.ranges
    return self._home_ticks_tensor(home_m) + u_scaled_tensor * (ranges / 2.0)

  def _escalado_desde_ticks_absolutos(self, ticks, home_m):
    """Inverse of `_ticks_absolutos_desde_escalado`: from absolute ticks of

    motor to normalized control vector [-1, 1]. Used to recompute
    `u_actual_scaled` from what was ACTUALLY commanded to the motor
    (after applying the minimum jump threshold), not what the MPC suggested.
    """
    ranges = self.mpc.candidate_generator.ranges
    if not torch.is_tensor(ticks):
      ticks = torch.tensor(ticks, device=self.device, dtype=torch.float32)
    return torch.clamp((ticks - self._home_ticks_tensor(home_m)) / (ranges / 2.0), -1.0, 1.0)

  def _sensor_thread_loop(self, freq_hz=60.0):
    """Dedicated thread: reads Dynamixel position+load in its own loop

    paced at `freq_hz`, updating `self.ultimas_posiciones_validas` /
    `self.ultimos_torques_validos`. The main control loop NEVER
    blocks on this read -- it only reads those already cached variables. Same
    pattern as `data_sampler_thread` in main.py.
    """
    intervalo = 1.0 / freq_hz
    ids = cfg.MOTOR_IDS
    while self._sensor_thread_running:
      t_loop_start = time.time()
      try:
        t_lock_start = time.perf_counter()
        with self.hardware_lock:
          t_esperando_lock_ms = (time.perf_counter() - t_lock_start) * 1000.0
          t_read_start = time.perf_counter()
          posiciones_reales, torques_reales, lectura_valida = (
              self.controlador_dynamixel.sync_get_present_position_and_load(ids)
          )
          t_read_ms = (time.perf_counter() - t_read_start) * 1000.0
        # Diagnostic: isolate whether the gap is waiting for the lock (held by the
        # main thread sending a move()) or the serial read itself
        # (a real communication hiccup with the Dynamixel bus).
        if t_esperando_lock_ms > 200.0 or t_read_ms > 200.0:
          print(f'🐢 [hilo sensor] esperó lock={t_esperando_lock_ms:.0f}ms'
                f' + lectura Dynamixel={t_read_ms:.0f}ms (anormal)')
        if lectura_valida == 1:
          self.ultimas_posiciones_validas = list(posiciones_reales)
          self.ultimos_torques_validos = list(torques_reales)
      except Exception as e:
        print(f'⚠️ Error leyendo Dynamixel (hilo sensor): {e}')

      t_ejec = time.time() - t_loop_start
      tiempo_espera = intervalo - t_ejec
      if tiempo_espera > 0:
        time.sleep(tiempo_espera)

  def _setup_hardware(self):
    """Initializes Dynamixel motors, OptiTrack and load cells (Phidget)."""
    self.controlador_dynamixel = create_controller(
        simulate=self.simulate, port=self.port, baudrate=self.baud
    )

    if self.simulate:
      return

    ids = cfg.MOTOR_IDS
    with self.hardware_lock:
      found = ids
      if hasattr(self.controlador_dynamixel, 'scan'):
        found = self.controlador_dynamixel.scan(ids) or ids
      if hasattr(self.controlador_dynamixel, 'enable_torque'):
        self.controlador_dynamixel.enable_torque(found)

      home_positions = [cfg.HOME_POSITION[mid] for mid in ids]
      self.controlador_dynamixel.move(
          ids, home_positions, speed=cfg.DEFAULT_SPEED, wait_for_reached=True
      )
    print(f'✓ Motores enviados a HOME_POSITION={[cfg.HOME_POSITION[mid] for mid in ids]}')

    # 🧵 Start the background Dynamixel sensing thread (its own 60 Hz).
    self._sensor_thread_running = True
    self._sensor_thread_obj = threading.Thread(
        target=self._sensor_thread_loop, args=(60.0,), daemon=True
    )
    self._sensor_thread_obj.start()
    print('✓ Hilo de sensado Dynamixel a 60 Hz iniciado.')

    if PhidgetForceController is not None:
      try:
        self.controlador_phidget = PhidgetForceController()
        print('✓ PhidgetBridge (4 celdas) inicializado correctamente.')
      except Exception as e:
        print(f'⚠️ No se pudo inicializar Phidget: {e}')
        self.controlador_phidget = None
    else:
      self.controlador_phidget = None

    # OptiTrack streaming is ALWAYS attempted whenever there is real hardware,
    # regardless of whether it controls the loop or not: --use-optitrack only
    # decides which source CLOSES the control loop; if Motive is not
    # transmitting, frames simply do not arrive (no error) and the OptiTrack
    # comparison columns remain empty in the log/console.
    if NatNetClient is None:
      print('⚠️ NatNetClient no disponible; no habrá datos de OptiTrack (ni control ni comparación).')
    else:
      self.streaming_client = NatNetClient()
      self.streaming_client.set_client_address(cfg.OPTITRACK_HOST)
      self.streaming_client.set_server_address(cfg.OPTITRACK_HOST)
      self.streaming_client.rigid_body_listener = self._on_rigid_body_frame
      self.streaming_client.run()
      rol = 'controlará el lazo (--use-optitrack)' if self.use_optitrack else 'solo se registrará para comparación (--no-optitrack controla con el observador)'
      print(f'✓ NatNetClient (OptiTrack) iniciado: {rol}.')

    if not self.use_optitrack:
      print('ℹ️ --no-optitrack activo: la pose del efector para el CONTROL se estima con el observador LSTM.')

  def _shutdown_hardware(self):
    """Safely shuts down motors, OptiTrack and Phidget (idempotent)."""
    # Stop the sensing thread BEFORE touching the port, so that it does not keep
    # trying to read while (or after) the torque is closed/disabled.
    if self._sensor_thread_running:
      self._sensor_thread_running = False
      if self._sensor_thread_obj is not None:
        self._sensor_thread_obj.join(timeout=1.0)
      print('✓ Hilo de sensado Dynamixel detenido.')

    if self.controlador_dynamixel is not None:
      try:
        if not self.simulate and hasattr(self.controlador_dynamixel, 'disable_torque'):
          with self.hardware_lock:
            self.controlador_dynamixel.disable_torque(cfg.MOTOR_IDS)
      except Exception as e:
        print(f'⚠️ Error al desactivar torque: {e}')
      try:
        with self.hardware_lock:
          self.controlador_dynamixel.close()
        print('✓ Controlador Dynamixel cerrado.')
      except Exception as e:
        print(f'⚠️ Error al cerrar Dynamixel: {e}')

    if self.streaming_client is not None:
      try:
        self.streaming_client.shutdown()
        print('✓ NatNetClient desconectado.')
      except Exception as e:
        print(f'⚠️ Error al cerrar NatNetClient: {e}')

    if self.controlador_phidget is not None:
      try:
        self.controlador_phidget.close()
        print('✓ PhidgetBridge desconectado correctamente.')
      except Exception as e:
        print(f'⚠️ Error al cerrar Phidget: {e}')

  def _consumer_logger(self, log_path):
    self._csv_logger = CSVLogger(log_path)
    while self.logging_running or not self.log_queue.empty():
      try:
        row = self.log_queue.get(timeout=0.05)
        self._csv_logger.log(row)
      except queue.Empty:
        pass

  def _configurar_hardware_inicial(self):
    """Hardware setup + GPU warm-up, ONLY ONCE per run
    (not per attempt/waypoint) -- extracted from run_control_loop so that it can
    also be reused in run_experimento_repeticiones without reconnecting
    OptiTrack/restarting the sensing thread on each repetition."""
    self._setup_hardware()

    # 🔥 Warm-up: the first GPU inference pays a one-time cost
    # (CUDA context init, cuDNN autotuning for the LSTM, etc.) that
    # can take tens of ms -- if that happens INSIDE the timed loop
    # it shows up as a giant spike in the first/second cycle (seen in
    # production: 89ms and 70ms, dropping to ~5-8ms from the 3rd cycle
    # onwards). It is paid here, before starting to measure, with a
    # throwaway call that exercises the same real code (full CEM).
    try:
      dummy_hist = torch.stack(list(self.buffer)[-self.t_in:]).unsqueeze(0)
      dummy_y_ref = torch.zeros(self.t_out, 3, device=self.device)
      dummy_u = torch.zeros(4, device=self.device)
      dummy_pos = torch.zeros(3, device=self.device)
      self.mpc.optimize(
          x_hist_tensor=dummy_hist, y_ref_mm=dummy_y_ref,
          u_current_scaled=dummy_u, current_pos_mm=dummy_pos,
      )
      if not self.use_optitrack:
        self._estimar_p_efector_observador()
      print('✓ Modelos precalentados (GPU/cuDNN listos).')
    except Exception as e:
      print(f'⚠️ No se pudo precalentar los modelos: {e}')

    # ⏳ Wait for the first valid OptiTrack frame before preloading the
    # buffer (otherwise the preload would start with stale/zero readings).
    if not self.simulate and self.use_optitrack:
      print('⏳ Esperando primer frame válido de OptiTrack (base + efector)...')
      t_espera_opti = time.perf_counter()
      while self._optitrack_pose_relativa_mm() is None:
        time.sleep(0.05)
        if time.perf_counter() - t_espera_opti > 10.0:
          print('⚠️ Sin frame válido de OptiTrack tras 10s -- arrancando de todas formas.')
          break

  def _precargar_buffer_en_posicion(self, home_m, waypoint_referencia_mm, u_actual_scaled, freq_hz=60.0):
    """Refreshes self.buffer with REAL sensor cycles (not simulated/zero).

    The buffer starts filled with zeros (see __init__), and the observer's LSTM
    never saw that kind of input in training -- its output there is
    unbounded garbage (the final layer has no activation), which when de-scaling
    can shoot up to hundreds of mm (seen in production:
    pos_ctrl > 900mm in the first cycles). It is called once at the start of
    the run, and again after each return to Home in
    run_experimento_repeticiones (so that the LSTM history reflects
    the current real position, not that of the previous attempt).

    Returns pos_filt_m0 (last filtered position reading, in meters).
    """
    intervalo = 1.0 / freq_hz
    print(f'⏳ Precargando búfer con {self._buffer_len} ciclos reales de sensor '
          f'(~{self._buffer_len / freq_hz:.1f}s)...')
    pos_filt_m0 = None
    for _ in range(self._buffer_len):
      t_ciclo = time.perf_counter()
      p_b0, q_b0, p_e0, q_e0, real_m0, couple_m0, tension_m0, filtrar0 = (
          self._leer_ciclo_sensores(waypoint_referencia_mm, home_m)
      )
      meta_m0 = self._ticks_absolutos_desde_escalado(u_actual_scaled, home_m).cpu().numpy().tolist()
      dict_features0, pos_filt_m0 = self._procesar_lecturas_cinematicas(
          p_b0, q_b0, p_e0, q_e0, real_m0, couple_m0, tension_m0, meta_m0, home_m,
          filtrar_posicion=filtrar0,
      )
      self.buffer.append(self.scaler.scale(dict_features0))
      dt_ciclo = time.perf_counter() - t_ciclo
      if dt_ciclo < intervalo:
        time.sleep(intervalo - dt_ciclo)
    print('✓ Búfer precargado con datos reales.')
    return pos_filt_m0

  def _resetear_estado_mpc_para_nuevo_intento(self):
    """Clears all memory between control cycles (CEM + hysteresis +
    setpoint lock) when starting a new attempt, so that each repetition
    is an independent test (does not "remember" the convergence of the previous
    attempt to the same point) -- see run_experimento_repeticiones."""
    self.mpc.error_history.clear()
    self.mpc.stagnation_counter = 0
    self.mpc.cem_std_ticks = None
    self.mpc.last_delta_ticks = None
    self.mpc.last_target_pos_mm = None
    self.acumulador_hist_ticks = [0.0, 0.0, 0.0, 0.0]
    self.ultimo_comando_ticks = None
    self.target_lock_origen_ticks = None
    self._ultimo_target_wp_hist = None

  def _mover_a_home_bloqueante(self, home_m):
    """Sends the robot to HOME_POSITION by direct motor command (without
    going through the CEM -- Home is the calibration reference, always
    reachable in motor space) and blocks until the driver confirms
    arrival. Used between attempts of run_experimento_repeticiones so that
    each repetition starts from the SAME physical position."""
    ids = cfg.MOTOR_IDS
    home_positions = [home_m[f'm{mid}'] for mid in ids]
    if self.simulate:
      return
    with self.hardware_lock:
      self.controlador_dynamixel.move(
          ids, home_positions, speed=cfg.DEFAULT_SPEED, wait_for_reached=True
      )

  def _leer_ciclo_sensores(self, target_wp_ref, home_m):
    """Reads one sensor cycle (effector pose, motors, forces).

    Extracted from the main loop so that it can also be reused during the
    preload of the observer buffer (see run_control_loop), before the
    control starts "for real".
    """
    if self.simulate:
      p_b, q_b = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
      p_e = (target_wp_ref / 1000.0) + np.random.normal(0, 0.0005, 3)
      q_e = [0.0, 0.0, 0.0, 1.0]
      real_m = [home_m[f'm{i}'] for i in range(1, 5)]
      couple_m = [0.0] * 4
      tension_m = [0.0] * 4
      # Camera-like synthetic noise -> do filter
      filtrar_pos_ctrl = True
    else:
      # 1a. Effector pose: real OptiTrack or estimate with the observer
      if self.use_optitrack:
        p_b, q_b = self.p_base, self.q_base
        p_e, q_e = self.p_efector, self.q_efector
        # Raw camera reading with jitter -> do filter (6 Hz Butterworth)
        filtrar_pos_ctrl = True
      else:
        p_b, q_b = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        p_e = self._estimar_p_efector_observador()
        q_e = [0.0, 0.0, 0.0, 1.0]
        # LSTM output: trained against OptiTrack already smoothed at 6 Hz -> do NOT refilter
        filtrar_pos_ctrl = False

      # 1b. Real position and torque of the Dynamixel motors: INSTANT
      # reading of what the background sensing thread has already updated
      # (Perf: the control loop no longer blocks waiting for
      # the 8 serial transactions every cycle).
      real_m = list(self.ultimas_posiciones_validas)
      couple_m = list(self.ultimos_torques_validos)

      # 1c. Tendon tension (Phidget load cells)
      if self.controlador_phidget is not None:
        try:
          t_phidget_start = time.perf_counter()
          tension_m = self.controlador_phidget.leer_fuerzas_gramos()
          t_phidget_ms = (time.perf_counter() - t_phidget_start) * 1000.0
          if t_phidget_ms > 200.0:
            print(f'🐢 [lazo principal] leer_fuerzas_gramos()={t_phidget_ms:.0f}ms (anormal)')
          self.ultimas_fuerzas_validas = tension_m
        except Exception:
          tension_m = list(self.ultimas_fuerzas_validas)
      else:
        tension_m = list(self.ultimas_fuerzas_validas)

    return p_b, q_b, p_e, q_e, real_m, couple_m, tension_m, filtrar_pos_ctrl

  def run_control_loop(
      self, waypoints_mm, home_m, target_tolerance_mm=10.0, hold_time_s=0.4,
      dwell_reset_tolerance_mm=13.0,
  ):
    # The CEM "small error" threshold (which disables widening/multi-
    # start on stagnation) has to be tied to the REAL arrival tolerance
    # of this run, not to the one it had by default when
    # self.mpc was built -- it is updated here so as not to depend on the order of
    # construction.
    self.mpc.error_chico_umbral_mm = target_tolerance_mm * 2.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use absolute path to avoid errors with os.makedirs on Windows
    log_path = os.path.abspath(f"mpc_experiment_{timestamp}.csv")

    self.logging_running = True
    log_thread = threading.Thread(
        target=self._consumer_logger, args=(log_path,), daemon=True
    )
    log_thread.start()

    # Header row (the 1st real column of the CSV is a UTC timestamp that
    # CSVLogger.log() automatically prepends to each row, this one included).
    self.log_queue.put([
        't_epoch', 'wp_idx',
        'target_x_mm', 'target_y_mm', 'target_z_mm',
        'pos_ctrl_x_mm', 'pos_ctrl_y_mm', 'pos_ctrl_z_mm',
        'pos_obs_x_mm', 'pos_obs_y_mm', 'pos_obs_z_mm',
        'pos_opti_x_mm', 'pos_opti_y_mm', 'pos_opti_z_mm',
        'err_mm', 'err_x_mm', 'err_y_mm', 'err_z_mm',
        'u1', 'u2', 'u3', 'u4',
        'real_m1_ticks', 'real_m2_ticks', 'real_m3_ticks', 'real_m4_ticks',
        'cmd_m1_ticks', 'cmd_m2_ticks', 'cmd_m3_ticks', 'cmd_m4_ticks',
        'raw_opt_m1_ticks', 'raw_opt_m2_ticks', 'raw_opt_m3_ticks', 'raw_opt_m4_ticks',
        'bloqueado_hist_m1', 'bloqueado_hist_m2', 'bloqueado_hist_m3', 'bloqueado_hist_m4',
        'en_limite_m1', 'en_limite_m2', 'en_limite_m3', 'en_limite_m4',
        'es_action_hold',
        'pred_tout_x_mm', 'pred_tout_y_mm', 'pred_tout_z_mm',
        't_calc_ms', 'cost',
    ])

    u_actual_scaled = torch.zeros(4, device=self.device)
    current_wp_idx = 0
    freq_hz = 60.0
    intervalo = 1.0 / freq_hz

    ids = cfg.MOTOR_IDS
    # Safety limits of the final command: they use the SAME range with which it
    # was trained (self.mpc.candidate_generator.homes/ranges, from the JSON), NOT
    # cfg.MOTOR_HOME_RANGES/cfg.LIMITS (smaller, used only for the sampling grid
    # of other data-collection scripts). This way the MPC can
    # reach the whole range that the model actually learned to control.
    home_ticks_np = self.mpc.candidate_generator.homes.cpu().numpy()
    half_range_np = (self.mpc.candidate_generator.ranges / 2.0).cpu().numpy()
    limits_low = np.clip(home_ticks_np - half_range_np, 0, cfg.DXL_MAXIMUM_POSITION_VALUE)
    limits_high = np.clip(home_ticks_np + half_range_np, 0, cfg.DXL_MAXIMUM_POSITION_VALUE)

    # 🎯 Action Hold state: sensing runs every cycle (60 Hz), but the
    # MPC only recomputes/sends a command every `action_hold_cycles` cycles.
    control_counter = 0
    cost_val = 0.0
    t_calc_ms = 0.0

    # ⏱️ "Dwell" state: instant (perf_counter) at which tolerance was first
    # entered for the current waypoint. None = not reached yet.
    wp_reached_time = None

    # Last observer estimate computed (to reuse in cycles where
    # it is not recomputed, see Perf in step 1d below).
    pos_obs_mm = np.zeros(3)

    print(f'📄 Guardando registros en: {log_path}')
    print(f'🚀 Iniciando Lazo de Control MPC a {freq_hz} Hz (sensado) /'
          f' ~{freq_hz / self.action_hold_cycles:.1f} Hz (comando, cada'
          f' {self.action_hold_cycles} ciclos) | Hold en waypoint:'
          f' {hold_time_s:.2f}s...\n')

    try:
      self._configurar_hardware_inicial()
      pos_filt_m0 = self._precargar_buffer_en_posicion(
          home_m, waypoints_mm[0], u_actual_scaled, freq_hz=freq_hz
      )

      wp_idx_sembrado = -1
      ultima_pos_conocida_mm = pos_filt_m0 * 1000.0
      # Same as at each waypoint transition: if the first waypoint
      # is far from where the preload ends, chop the jump too.
      cola_subobjetivos = _generar_subobjetivos(ultima_pos_conocida_mm, waypoints_mm[0])
      if cola_subobjetivos:
        print(f'🧭 Salto grande al WP 1: insertando {len(cola_subobjetivos)} '
              'sub-objetivo(s) intermedio(s).')

      while current_wp_idx < len(waypoints_mm):
        target_real_wp = waypoints_mm[current_wp_idx]
        en_subobjetivo = len(cola_subobjetivos) > 0
        target_wp = cola_subobjetivos[0] if en_subobjetivo else target_real_wp

        # Reset the anti-deadband accumulator when changing target: do not
        # carry over a pending correction from ANOTHER waypoint.
        if (
            self._ultimo_target_wp_hist is None
            or not np.allclose(target_wp, self._ultimo_target_wp_hist, atol=1e-6)
        ):
          self.acumulador_hist_ticks = [0.0, 0.0, 0.0, 0.0]
          self._ultimo_target_wp_hist = target_wp.copy()
        # Fixed seeds and multi-start only apply to the REAL
        # waypoint, not to the intermediate step sub-goals.
        seed_para_este_wp = None if en_subobjetivo else _buscar_semilla_para_target(target_real_wp)

        # 🌱 Seed command (see SEED_COMMANDS_TICKS): send a known
        # fixed command and wait for the real position to converge there BEFORE
        # activating the predictor for this particular waypoint. It is done
        # only once per waypoint (wp_idx_sembrado avoids repeating it every cycle).
        if (
            USAR_SEMILLAS_FIJAS
            and not self.simulate
            and seed_para_este_wp is not None
            and current_wp_idx != wp_idx_sembrado
        ):
          seed_ticks = [int(round(v)) for v in seed_para_este_wp]
          print(
              f'🌱 [WP {current_wp_idx + 1}] Enviando comando semilla fijo '
              f'{dict(zip(ids, seed_ticks))} (sin MPC) -- esperando <= '
              f'{SEED_TOLERANCIA_TICKS:.0f} ticks antes de activar el predictor...'
          )
          with self.hardware_lock:
            self.controlador_dynamixel.move(
                ids, seed_ticks, speed=cfg.DEFAULT_SPEED, wait_for_reached=False
            )

          t_seed_start = time.perf_counter()
          while True:
            t_ciclo_seed = time.perf_counter()
            p_bS, q_bS, p_eS, q_eS, real_mS, couple_mS, tension_mS, filtrarS = (
                self._leer_ciclo_sensores(target_wp, home_m)
            )
            dict_featuresS, pos_filt_mS = self._procesar_lecturas_cinematicas(
                p_bS, q_bS, p_eS, q_eS, real_mS, couple_mS, tension_mS, seed_ticks,
                home_m, filtrar_posicion=filtrarS,
            )
            self.buffer.append(self.scaler.scale(dict_featuresS))

            # Also record the preload cycles in the CSV (before they
            # were left out -- that produced a temporal gap that plot_resultados.py
            # drew as a misleading straight ramp between the last point from
            # before the seed and the first one after).
            pos_opti_mm_seed_log = self._optitrack_pose_relativa_mm()
            if pos_opti_mm_seed_log is None:
              pos_opti_mm_seed_log = np.full(3, np.nan)
            pos_ctrl_mm_seed_log = pos_filt_mS * 1000.0
            diff_mm_seed_log = pos_ctrl_mm_seed_log - target_wp
            u_seed_scaled_log = self._escalado_desde_ticks_absolutos(seed_ticks, home_m)
            self.log_queue.put(
                [time.time(), current_wp_idx]
                + target_wp.tolist()
                + pos_ctrl_mm_seed_log.tolist()
                + pos_obs_mm.tolist()
                + pos_opti_mm_seed_log.tolist()
                + [float(np.linalg.norm(diff_mm_seed_log))] + diff_mm_seed_log.tolist()
                + u_seed_scaled_log.cpu().numpy().tolist()
                + [float(t) for t in real_mS]
                + [float(t) for t in seed_ticks]
                + [float('nan')] * 4  # raw_opt_m1-4_ticks: not applicable (no CEM here)
                + [0, 0, 0, 0]  # bloqueado_hist_m1-4: not applicable
                + [0, 0, 0, 0]  # en_limite_m1-4: not applicable
                + [0]  # es_action_hold
                + [float('nan')] * 3  # pred_tout_x/y/z_mm: not applicable
                + [0.0, float('nan')]  # t_calc_ms, cost: not applicable
            )

            diffs = [abs(r - s) for r, s in zip(real_mS, seed_ticks)]
            if max(diffs) <= SEED_TOLERANCIA_TICKS:
              print(f'✓ Semilla alcanzada (máx. diff={max(diffs):.1f} ticks).')
              break
            if time.perf_counter() - t_seed_start > SEED_TIMEOUT_S:
              print(f'⚠️ No se alcanzó la tolerancia de semilla en '
                    f'{SEED_TIMEOUT_S:.0f}s (máx. diff={max(diffs):.1f} ticks) '
                    '-- activando predictor de todas formas.')
              break

            dt_ciclo_seed = time.perf_counter() - t_ciclo_seed
            if dt_ciclo_seed < intervalo:
              time.sleep(intervalo - dt_ciclo_seed)

          # Seed error against the target, BEFORE the predictor
          # touches anything -- to know whether the known-good combination
          # still reaches the point (physical drift) or not.
          pos_filt_mm_S = pos_filt_mS * 1000.0
          err_seed_mm = float(np.linalg.norm(pos_filt_mm_S - target_wp))
          pos_opti_mm_S = self._optitrack_pose_relativa_mm()
          if pos_opti_mm_S is not None:
            err_seed_opti_mm = float(np.linalg.norm(pos_opti_mm_S - target_wp))
            print(f'📍 Error de la semilla vs. target: {err_seed_mm:.2f} mm (control) | '
                  f'{err_seed_opti_mm:.2f} mm (OptiTrack) | Ctrl={np.round(pos_filt_mm_S, 1)} mm '
                  f'| Opti={np.round(pos_opti_mm_S, 1)} mm | Target={target_wp} mm')
          else:
            print(f'📍 Error de la semilla vs. target: {err_seed_mm:.2f} mm (control) '
                  f'| Ctrl={np.round(pos_filt_mm_S, 1)} mm | Target={target_wp} mm '
                  '| (sin dato OptiTrack fresco)')
          print('   Activando predictor para este waypoint desde acá.')

          u_actual_scaled = self._escalado_desde_ticks_absolutos(seed_ticks, home_m)
          self.ultimo_comando_ticks = list(seed_ticks)
          wp_idx_sembrado = current_wp_idx
          continue  # start the normal control cycle clean for this waypoint

        t_start = time.perf_counter()
        control_counter += 1

        y_ref_mm = (
            torch.tensor(target_wp, device=self.device)
            .unsqueeze(0)
            .repeat(self.t_out, 1)
        )

        # 1. Read Raw Sensors (Simulation / Hardware)
        p_b, q_b, p_e, q_e, real_m, couple_m, tension_m, filtrar_pos_ctrl = (
            self._leer_ciclo_sensores(target_wp, home_m)
        )

        # 1d. LSTM observer estimate, to compare against
        # OptiTrack/target in the log. If it was already computed above as p_e
        # (--no-optitrack), it is reused (free, no extra cost). If it is
        # only for comparison (--use-optitrack), Perf: it is recomputed only
        # at the Action Hold rate instead of every 60 Hz sensing cycle — it is
        # a full LSTM inference that does not need to run 4x more often
        # than necessary. In simulate it is always recomputed
        # (irrelevant in cost, useful for logic tests).
        if not self.simulate and not self.use_optitrack:
          pos_obs_mm = np.asarray(p_e, dtype=np.float64) * 1000.0
        elif self.simulate or control_counter % self.action_hold_cycles == 0:
          pos_obs_mm = self._estimar_p_efector_observador() * 1000.0
        # else: the pos_obs_mm computed in a previous cycle is reused

        # 1e. Real OptiTrack pose for comparison (None if there is no fresh
        # data, e.g. Motive is not transmitting).
        pos_opti_mm = self._optitrack_pose_relativa_mm()
        if pos_opti_mm is None:
          pos_opti_mm = np.full(3, np.nan)

        # 2. Relative Kinematic Transformation and Causal Filtering
        meta_ticks_abs = self._ticks_absolutos_desde_escalado(u_actual_scaled, home_m)
        meta_m = meta_ticks_abs.cpu().numpy().tolist()
        dict_features, pos_filt_m = self._procesar_lecturas_cinematicas(
            p_b,
            q_b,
            p_e,
            q_e,
            real_m,
            couple_m,
            tension_m,
            meta_m,
            home_m,
            filtrar_posicion=filtrar_pos_ctrl,
        )

        # 3. Normalize Vector [17] and Save in Buffer
        vec_scaled = self.scaler.scale(dict_features)
        self.buffer.append(vec_scaled)
        x_hist_tensor = torch.stack(list(self.buffer)[-self.t_in:]).unsqueeze(0)

        # CURRENT REAL position (Cartesian, mm) perceived by OptiTrack/observer
        # (the same one shown as "Ctrl" in the log), so that the MPC computes
        # its error against the target using the real sensor, not an internal assumption.
        pos_filt_mm = pos_filt_m * 1000.0
        current_pos_mm = torch.tensor(pos_filt_mm, device=self.device, dtype=torch.float32)
        ultima_pos_conocida_mm = pos_filt_mm.copy()

        # 4-5. MPC Optimization + command sending (Action Hold: only every
        # `action_hold_cycles` cycles is it recomputed; in the intermediate cycles
        # the last command is kept, giving the cable time to tense and
        # deform the silicone).
        es_ciclo_action_hold = (control_counter % self.action_hold_cycles == 0)
        # Per-cycle diagnostic (for the CSV): raw suggestion of the optimizer
        # before clip/hysteresis, and whether the hysteresis/physical limit
        # blocked the correction of each motor -- NaN/0 in the cycles where
        # it is not recomputed (no action-hold) or the setpoint lock does not advance.
        raw_opt_ticks_row = [float('nan')] * 4
        bloqueado_hist_row = [0, 0, 0, 0]
        en_limite_row = [0, 0, 0, 0]
        pred_tout_mm_row = [float('nan')] * 3
        # (Completely freezing the optimizer during the dwell was tried,
        # but it turned out WORSE: the robot drifts on its own while nothing is
        # sent -- real, not an artifact -- and when reactivated the buffer ends up
        # full of that "uncommanded" drift, the model cost shoots up
        # and it corrects all at once violently instead of smoothly. Better to keep
        # correcting the small drift in real time; the other defenses
        # -directional persistence, no widening with small error, jump
        # cap- already prevent the "dance" that originally motivated the freeze.)
        if es_ciclo_action_hold:
          u_opt, cost_val, t_calc_ms, top_u_scaled, top_costs, y_pred_final_mm = self.mpc.optimize(
              x_hist_tensor=x_hist_tensor,
              y_ref_mm=y_ref_mm,
              u_current_scaled=u_actual_scaled,
              current_pos_mm=current_pos_mm,
          )
          # Predictor's belief about where it will end up (step t_out,
          # the one that w_terminal weighs most) under the chosen command -- to
          # compare in the CSV against what the sensor actually measures a few
          # cycles later (see pred_tout_*_mm columns).
          pred_tout_mm_row = y_pred_final_mm[-1].cpu().numpy().tolist()

          # Top-K candidates of the refinement pass, for diagnostics
          top_ticks = self._ticks_absolutos_desde_escalado(top_u_scaled, home_m).cpu().numpy()
          top_costs_np = top_costs.cpu().numpy()
          top_str = ' | '.join(
              f'#{i + 1} ticks={np.round(top_ticks[i]).astype(int).tolist()} costo={top_costs_np[i]:.3f}'
              for i in range(len(top_costs_np))
          )
          print(f'🏆 Top-{len(top_costs_np)} candidatos: {top_str}')

          if not self.simulate:
            # 🔒 Setpoint Lock: the optimizer's new target is not adopted
            # until the real position has covered `setpoint_lock_pct`
            # of the distance between the origin and the current target. The
            # optimizer has already computed u_opt (it keeps "thinking" in the
            # background); if the progress is not enough, the command is simply not touched
            # and the servo continues its ongoing trajectory without interruption.
            if self.target_lock_origen_ticks is None or self.ultimo_comando_ticks is None or self.setpoint_lock_pct <= 0:
              avance_ok = True
              progreso = 1.0
            else:
              origen = np.array(self.target_lock_origen_ticks, dtype=np.float32)
              target_vigente = np.array(self.ultimo_comando_ticks, dtype=np.float32)
              real_actual = np.array(real_m, dtype=np.float32)
              dist_total = np.linalg.norm(target_vigente - origen)
              dist_avanzada = np.linalg.norm(real_actual - origen)
              progreso = float(dist_avanzada / dist_total) if dist_total > 1e-6 else 1.0
              avance_ok = progreso >= self.setpoint_lock_pct

            if not avance_ok:
              print(f'🔒 Setpoint Lock: avance {progreso * 100:.0f}%/'
                    f'{self.setpoint_lock_pct * 100:.0f}% -> manteniendo target vigente.')
            else:
              # 🌊 Exponential smoothing (LERP) in scaled space: instead of
              # jumping straight to u_opt, it interpolates towards it. low alpha ->
              # smooth transition (avoids the abrupt jump of the optimizer).
              u_suavizado = self.ema_alpha * u_opt + (1.0 - self.ema_alpha) * u_actual_scaled

              posiciones_objetivo_ticks = self._ticks_absolutos_desde_escalado(u_suavizado, home_m)
              posiciones_objetivo_crudo = posiciones_objetivo_ticks.cpu().numpy()
              raw_opt_ticks_row = posiciones_objetivo_crudo.tolist()
              posiciones_objetivo = np.clip(
                  posiciones_objetivo_crudo, limits_low, limits_high
              )
              posiciones_objetivo = [int(round(p)) for p in posiciones_objetivo]

              # Diagnostic: warn if the target pushes any motor to the configured
              # physical limit (home ± range). It is the signal that the point
              # may be at the edge or outside of what the model saw in
              # training -- the optimizer gets "stuck" at the limit without
              # a stable solution to converge to.
              margen_limite_ticks = 15
              en_el_limite = [
                  ids[i] for i in range(len(ids))
                  if posiciones_objetivo_crudo[i] <= limits_low[i] + margen_limite_ticks
                  or posiciones_objetivo_crudo[i] >= limits_high[i] - margen_limite_ticks
              ]
              en_limite_row = [1 if mid in en_el_limite else 0 for mid in ids]
              if en_el_limite:
                print(f'⚠️ Motor(es) {en_el_limite} en/cerca del límite físico '
                      f'configurado -- el target puede estar fuera de lo que el '
                      f'modelo vio en entrenamiento.')

              if self.ultimo_comando_ticks is None:
                self.ultimo_comando_ticks = [int(home_m[f'm{i}']) for i in range(1, 5)]

              # Minimum jump threshold (hysteresis) WITH anti-
              # deadband accumulator: if the change requested this cycle is small, it is not
              # simply discarded -- a fraction (ganancia_integral_hist) is added
              # to that motor's accumulator. If the requested correction is
              # consistent cycle after cycle, the accumulator crosses the threshold and
              # is released all at once (integrator effect); if it is
              # one-off noise that changes sign, it cancels itself out. Anti-windup
              # cap so it does not grow without limit if it stays blocked
              # for another reason (e.g. setpoint lock).
              tope_acumulador = self.delta_min_ticks * 1.5
              posiciones_finales = []
              for idx_motor, (prev, nuevo) in enumerate(
                  zip(self.ultimo_comando_ticks, posiciones_objetivo)
              ):
                self.acumulador_hist_ticks[idx_motor] += (
                    (nuevo - prev) * self.ganancia_integral_hist
                )
                self.acumulador_hist_ticks[idx_motor] = float(np.clip(
                    self.acumulador_hist_ticks[idx_motor],
                    -tope_acumulador, tope_acumulador,
                ))
                correccion_efectiva = self.acumulador_hist_ticks[idx_motor]
                if abs(correccion_efectiva) < self.delta_min_ticks:
                  posiciones_finales.append(prev)
                  bloqueado_hist_row[idx_motor] = 1
                else:
                  nuevo_final = int(round(np.clip(
                      prev + correccion_efectiva,
                      limits_low[idx_motor], limits_high[idx_motor],
                  )))
                  posiciones_finales.append(nuevo_final)
                  self.acumulador_hist_ticks[idx_motor] = 0.0  # released, reset
              self.ultimo_comando_ticks = posiciones_finales
              # New origin to measure progress towards THIS target
              self.target_lock_origen_ticks = list(real_m)

              t_lock_start = time.perf_counter()
              with self.hardware_lock:
                t_esperando_lock_ms = (time.perf_counter() - t_lock_start) * 1000.0
                t_move_start = time.perf_counter()
                self.controlador_dynamixel.move(
                    ids,
                    posiciones_finales,
                    speed=cfg.DEFAULT_SPEED,
                    wait_for_reached=False,  # CRITICAL: do not block the MPC loop
                )
                t_move_ms = (time.perf_counter() - t_move_start) * 1000.0
              if t_esperando_lock_ms > 200.0 or t_move_ms > 200.0:
                print(f'🐢 [lazo principal] esperó lock={t_esperando_lock_ms:.0f}ms'
                      f' + move()={t_move_ms:.0f}ms (anormal)')

              # Recenter u_actual_scaled on what was ACTUALLY sent (post-threshold),
              # not on the raw MPC suggestion, so that the next search
              # starts from the real physical state of the robot.
              u_actual_scaled = self._escalado_desde_ticks_absolutos(posiciones_finales, home_m)

              # Command sent vs. real position read in this same cycle (step 1b),
              # motor by motor, to verify that the robot follows the commands.
              comando_str = ', '.join(f'm{mid}={p}' for mid, p in zip(ids, posiciones_finales))
              actual_str = ', '.join(f'm{mid}={p}' for mid, p in zip(ids, real_m))
              print(f'🔧 Comando -> [{comando_str}] | Posición motor -> [{actual_str}]')
          else:
            u_actual_scaled = u_opt

        # 6. Record Row in CSV (target, control, observer and OptiTrack)
        diff_mm = pos_filt_mm - target_wp
        err_mm_val = float(np.linalg.norm(diff_mm))
        cmd_ticks_row = (
            list(self.ultimo_comando_ticks) if self.ultimo_comando_ticks is not None
            else [int(home_m[f'm{i}']) for i in range(1, 5)]
        )
        log_row = (
            [time.time(), current_wp_idx]
            + target_wp.tolist()
            + pos_filt_mm.tolist()
            + pos_obs_mm.tolist()
            + pos_opti_mm.tolist()
            + [err_mm_val] + diff_mm.tolist()
            + u_actual_scaled.cpu().numpy().tolist()
            + [float(t) for t in real_m]
            + [float(t) for t in cmd_ticks_row]
            + [float(t) for t in raw_opt_ticks_row]
            + bloqueado_hist_row
            + en_limite_row
            + [int(es_ciclo_action_hold)]
            + [float(v) for v in pred_tout_mm_row]
            + [t_calc_ms, cost_val]
        )
        self.log_queue.put(log_row)

        # 7. Evaluate Waypoint Arrival (always with respect to the
        # control source, every cycle -> dist_mm and the dwell need fine resolution).
        dist_mm = err_mm_val

        # Perf: printing to the console is expensive (terminal I/O) and is not
        # needed at 60 Hz -> it is shown only at the Action Hold rate.
        if es_ciclo_action_hold:
          opti_str = (
              f'{np.round(pos_opti_mm, 1)} mm' if not np.isnan(pos_opti_mm).any() else 'N/D'
          )
          etiqueta_wp = (
              f'[Sub-objetivo -> WP {current_wp_idx + 1}/{len(waypoints_mm)}]'
              if en_subobjetivo else f'[WP {current_wp_idx + 1}/{len(waypoints_mm)}]'
          )
          pred_tout_str = np.round(np.asarray(pred_tout_mm_row), 1).tolist()
          print(
              f'📍 {etiqueta_wp} Target:'
              f' {target_wp} mm' #| Ctrl: {np.round(pos_filt_mm, 1)} mm
              f'| Obs:'
              f' {np.round(pos_obs_mm, 1)} mm | Opti: {opti_str} | Err:'
              f' {dist_mm:.2f} mm | Calc: {t_calc_ms:.2f} ms'
              f' | Pred(t+{self.mpc.t_out}): {pred_tout_str} mm'
          )

        if en_subobjetivo:
          # 🧭 Intermediate sub-goal: lax tolerance, no dwell -- just
          # "pass nearby and continue" towards the next one (or towards the real
          # waypoint if it was the last in the queue). It does not touch wp_reached_time nor
          # current_wp_idx, that is reserved for the real waypoint.
          if dist_mm <= TOLERANCIA_SUBOBJETIVO_MM:
            cola_subobjetivos.pop(0)
        else:
          # Hysteresis band: entering the dwell requires <= target_tolerance_mm,
          # but the count is only restarted if the error escapes above
          # dwell_reset_tolerance_mm (wider). Without this, any noise micro-bounce
          # (±a few mm, typical of the LSTM observer/cable compliance)
          # just above target_tolerance_mm restarts the whole count and
          # the waypoint never gets to be "reached" even though it is practically there.
          if dist_mm <= target_tolerance_mm:
            if wp_reached_time is None:
              wp_reached_time = time.perf_counter()
              print(f'🎯 Waypoint {current_wp_idx + 1} dentro de tolerancia,'
                    f' manteniendo posición {hold_time_s:.2f}s...')
            elif (time.perf_counter() - wp_reached_time) >= hold_time_s:
              print(f'✅ Waypoint {current_wp_idx + 1} alcanzado (mantenido'
                    f' {hold_time_s:.2f}s).\n')
              current_wp_idx += 1
              wp_reached_time = None
              # Build the queue of sub-goals for the NEXT real waypoint,
              # if the jump towards it is large (see DISTANCIA_MAX_SALTO_MM).
              if current_wp_idx < len(waypoints_mm) and ultima_pos_conocida_mm is not None:
                cola_subobjetivos = _generar_subobjetivos(
                    ultima_pos_conocida_mm, waypoints_mm[current_wp_idx]
                )
                if cola_subobjetivos:
                  print(f'🧭 Salto grande al WP {current_wp_idx + 1}: '
                        f'insertando {len(cola_subobjetivos)} sub-objetivo(s) intermedio(s).')
          elif dist_mm > dwell_reset_tolerance_mm:
            # It really escaped (not noise) -> only then restart the count
            wp_reached_time = None
          # else: target_tolerance_mm < dist_mm <= dwell_reset_tolerance_mm ->
          # the ongoing count is kept (if it had already started) without adding or
          # restarting, absorbing the measurement noise within the band.

        # 8. Strict 60 Hz Synchronization (Real Loop Pacing)
        t_ejec = time.perf_counter() - t_start
        tiempo_espera = intervalo - t_ejec
        if tiempo_espera > 0:
          time.sleep(tiempo_espera)
        else:
          print(
              f'⚠️ ({t_ejec * 1000:.2f} ms)'
          )

    except KeyboardInterrupt:
      print('\n🛑 Interrumpido por el usuario (Ctrl+C). Cerrando de forma segura...')

    finally:
      # Safe shutdown of the hardware and the log files
      self._shutdown_hardware()
      self.logging_running = False
      log_thread.join(timeout=2.0)
      if self._csv_logger is not None:
        self._csv_logger.close()
      print('🏁 Trayectoria finalizada y datos exportados a CSV.')

  def run_experimento_repeticiones(
      self, waypoints_mm, home_m, n_repeticiones=3,
      target_tolerance_mm=10.0, hold_time_s=0.4, dwell_reset_tolerance_mm=12.0,
      timeout_intento_s=18.0, fraccion_semilla=1.0,
  ):
    """Validation protocol with independent repetitions.

    `fraccion_semilla` (0,1]: what fraction of the Home->seed path (in
    TICKS space, not cartesian) is sent in open loop before handing
    control to the CEM. With 1.0 (default) it is the full bypass as
    always. With, for example, 0.3 or 0.5, the CEM is given a partial push
    towards the correct branch and left to complete the rest of the path
    autonomously -- it serves to distinguish whether the problem is only
    "discovering" the branch from Home (in that case, a partial push is enough
    for the CEM to converge on its own) or whether the model is miscalibrated across all of
    that neighborhood of the actuator space (in that case, not even with the partial
    push does it converge).

    Each waypoint is attempted `n_repeticiones` times, ALWAYS starting
    from HOME_POSITION (direct motor command, no CEM). Per attempt, each cycle is
    recorded (timestamp, wp_idx, intento_idx, target, control
    position, raw OptiTrack position -- "ground truth" --, instantaneous
    error against both) in memory, and when the attempt closes (it arrived
    within tolerance sustained for `hold_time_s`, or `timeout_intento_s`
    ran out without arriving) the whole block is dumped to the CSV with the
    LAST row marked 'exito'/'fallo' in the `resultado` column (the rest
    stay ''). No attempt is discarded, even if it fails.

    Unlike run_control_loop (continuous trajectory, intended for real
    operation visiting several waypoints in sequence), here each
    attempt is an INDEPENDENT test: all the CEM memory is reset
    (see _resetear_estado_mpc_para_nuevo_intento) when starting each one, so
    that a repetition does not "inherit" the convergence of the previous one to the same
    point -- otherwise, a success could be due to the std/direction memory
    of the previous attempt, not to the system really reaching it
    reliably from scratch.

    NOTE: unlike run_control_loop, this method does not implement EMA
    (--ema-alpha) nor Setpoint Lock (--setpoint-lock-pct) -- both are
    disabled by default (no-op) in the normal configuration, so there is
    no difference in behavior if it is run with the default
    parameters. If at some point those flags are used for the experiment,
    they have to be added here too.
    """
    self.mpc.error_chico_umbral_mm = target_tolerance_mm * 2.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.abspath(f"mpc_experimento_repeticiones_{timestamp}.csv")

    self.logging_running = True
    log_thread = threading.Thread(
        target=self._consumer_logger, args=(log_path,), daemon=True
    )
    log_thread.start()

    self.log_queue.put([
        't_epoch', 'wp_idx', 'intento_idx',
        'target_x_mm', 'target_y_mm', 'target_z_mm',
        'pos_ctrl_x_mm', 'pos_ctrl_y_mm', 'pos_ctrl_z_mm',
        'pos_obs_x_mm', 'pos_obs_y_mm', 'pos_obs_z_mm',
        'pos_opti_x_mm', 'pos_opti_y_mm', 'pos_opti_z_mm',
        'err_mm', 'err_x_mm', 'err_y_mm', 'err_z_mm',
        'err_opti_mm', 'err_opti_x_mm', 'err_opti_y_mm', 'err_opti_z_mm',
        'u1', 'u2', 'u3', 'u4',
        'real_m1_ticks', 'real_m2_ticks', 'real_m3_ticks', 'real_m4_ticks',
        'cmd_m1_ticks', 'cmd_m2_ticks', 'cmd_m3_ticks', 'cmd_m4_ticks',
        'bloqueado_hist_m1', 'bloqueado_hist_m2', 'bloqueado_hist_m3', 'bloqueado_hist_m4',
        'en_limite_m1', 'en_limite_m2', 'en_limite_m3', 'en_limite_m4',
        'es_action_hold', 't_calc_ms', 'cost', 'resultado',
    ])

    freq_hz = 60.0
    intervalo = 1.0 / freq_hz
    ids = cfg.MOTOR_IDS
    home_ticks_np = self.mpc.candidate_generator.homes.cpu().numpy()
    half_range_np = (self.mpc.candidate_generator.ranges / 2.0).cpu().numpy()
    limits_low = np.clip(home_ticks_np - half_range_np, 0, cfg.DXL_MAXIMUM_POSITION_VALUE)
    limits_high = np.clip(home_ticks_np + half_range_np, 0, cfg.DXL_MAXIMUM_POSITION_VALUE)

    u_actual_scaled = torch.zeros(4, device=self.device)
    resumen = []  # (wp_idx, intento_idx, result, final_opti_err_mm, duration_s)

    n_total_intentos = len(waypoints_mm) * n_repeticiones
    print(f'📄 Guardando registros en: {log_path}')
    print(f'🧪 Experimento de repeticiones: {len(waypoints_mm)} waypoints x '
          f'{n_repeticiones} intentos = {n_total_intentos} intentos totales | '
          f'timeout por intento: {timeout_intento_s:.1f}s\n')

    try:
      self._configurar_hardware_inicial()
      pos_filt_m0 = self._precargar_buffer_en_posicion(
          home_m, waypoints_mm[0], u_actual_scaled, freq_hz=freq_hz
      )
      ultima_pos_conocida_mm = pos_filt_m0 * 1000.0

      for wp_idx, target_wp in enumerate(waypoints_mm):
        for intento_idx in range(n_repeticiones):
          es_primer_intento_absoluto = (wp_idx == 0 and intento_idx == 0)

          if not es_primer_intento_absoluto:
            print(f'🏠 Retornando a HOME antes del intento {intento_idx + 1}/'
                  f'{n_repeticiones} del WP {wp_idx + 1}/{len(waypoints_mm)}...')
            self._mover_a_home_bloqueante(home_m)
            u_actual_scaled = torch.zeros(4, device=self.device)
            pos_filt_m0 = self._precargar_buffer_en_posicion(
                home_m, target_wp, u_actual_scaled, freq_hz=freq_hz
            )
            ultima_pos_conocida_mm = pos_filt_m0 * 1000.0

          # Independent test: no memory of the previous attempt.
          self._resetear_estado_mpc_para_nuevo_intento()

          filas_intento = []
          # Last VALID model prediction (t+1 and t+t_out) seen during
          # this attempt -- it persists between cycles (unlike
          # pred_tout_mm_row, which is reset to NaN every cycle for the CSV)
          # so that the last recompute cycle's prediction can be reported in the final
          # SUMMARY, useful for comparing in the report.
          ultimo_pred_t1_mm = [float('nan')] * 3
          ultimo_pred_tout_mm = [float('nan')] * 3

          # 🌱 Known-good seed command (see SEED_COMMANDS_TICKS): if there is
          # a confirmed seed for THIS target and USAR_SEMILLAS_FIJAS is
          # active, send the fixed command directly (no CEM) and wait for the
          # real position to converge there, BEFORE letting the CEM take
          # control -- same as in run_control_loop. The chopping into
          # sub-goals is skipped (the seed is already a direct jump confirmed by hand).
          seed_para_este_target = _buscar_semilla_para_target(target_wp)
          usar_semilla_este_intento = (
              USAR_SEMILLAS_FIJAS and not self.simulate and seed_para_este_target is not None
          )

          if usar_semilla_este_intento:
            seed_ticks_full = [int(round(v)) for v in seed_para_este_target]
            if fraccion_semilla < 1.0:
              home_ticks_actual = [home_m[f'm{i}'] for i in range(1, 5)]
              seed_ticks = [
                  int(round(h + fraccion_semilla * (s - h)))
                  for h, s in zip(home_ticks_actual, seed_ticks_full)
              ]
              print(f'🌱 [WP {wp_idx + 1}] Empujón parcial ({fraccion_semilla*100:.0f}% '
                    f'Home->semilla): {dict(zip(ids, seed_ticks))} (sin MPC) -- esperando '
                    f'<= {SEED_TOLERANCIA_TICKS:.0f} ticks, después el CEM completa el resto...')
            else:
              seed_ticks = seed_ticks_full
              print(f'🌱 [WP {wp_idx + 1}] Enviando comando semilla fijo '
                    f'{dict(zip(ids, seed_ticks))} (sin MPC) -- esperando <= '
                    f'{SEED_TOLERANCIA_TICKS:.0f} ticks...')
            with self.hardware_lock:
              self.controlador_dynamixel.move(
                  ids, seed_ticks, speed=cfg.DEFAULT_SPEED, wait_for_reached=False
              )

            t_seed_start = time.perf_counter()
            pos_ctrl_mm_seed = ultima_pos_conocida_mm.copy()
            while True:
              t_ciclo_seed = time.perf_counter()
              p_bS, q_bS, p_eS, q_eS, real_mS, couple_mS, tension_mS, filtrarS = (
                  self._leer_ciclo_sensores(target_wp, home_m)
              )
              dict_featuresS, pos_filt_mS = self._procesar_lecturas_cinematicas(
                  p_bS, q_bS, p_eS, q_eS, real_mS, couple_mS, tension_mS, seed_ticks,
                  home_m, filtrar_posicion=filtrarS,
              )
              self.buffer.append(self.scaler.scale(dict_featuresS))

              pos_opti_mm_seed = self._optitrack_pose_relativa_mm()
              if pos_opti_mm_seed is None:
                pos_opti_mm_seed = np.full(3, np.nan)
              pos_ctrl_mm_seed = pos_filt_mS * 1000.0
              diff_mm_seed = pos_ctrl_mm_seed - target_wp
              diff_opti_seed = pos_opti_mm_seed - target_wp
              err_opti_seed = (
                  float(np.linalg.norm(diff_opti_seed))
                  if not np.isnan(pos_opti_mm_seed).any() else float('nan')
              )
              u_seed_scaled = self._escalado_desde_ticks_absolutos(seed_ticks, home_m)
              filas_intento.append(
                  [time.time(), wp_idx, intento_idx]
                  + target_wp.tolist()
                  + pos_ctrl_mm_seed.tolist()
                  + [0.0, 0.0, 0.0]
                  + pos_opti_mm_seed.tolist()
                  + [float(np.linalg.norm(diff_mm_seed))] + diff_mm_seed.tolist()
                  + [err_opti_seed] + diff_opti_seed.tolist()
                  + u_seed_scaled.cpu().numpy().tolist()
                  + [float(t) for t in real_mS]
                  + [float(t) for t in seed_ticks]
                  + [0, 0, 0, 0]
                  + [0, 0, 0, 0]
                  + [0, 0.0, float('nan'), '']
              )

              diffs = [abs(r - s) for r, s in zip(real_mS, seed_ticks)]
              if max(diffs) <= SEED_TOLERANCIA_TICKS:
                print(f'✓ Semilla alcanzada (máx. diff={max(diffs):.1f} ticks).')
                break
              if time.perf_counter() - t_seed_start > SEED_TIMEOUT_S:
                print(f'⚠️ No se alcanzó la tolerancia de semilla en '
                      f'{SEED_TIMEOUT_S:.0f}s (máx. diff={max(diffs):.1f} ticks) '
                      '-- activando CEM de todas formas.')
                break

              dt_ciclo_seed = time.perf_counter() - t_ciclo_seed
              if dt_ciclo_seed < intervalo:
                time.sleep(intervalo - dt_ciclo_seed)

            u_actual_scaled = self._escalado_desde_ticks_absolutos(seed_ticks, home_m)
            self.ultimo_comando_ticks = list(seed_ticks)
            ultima_pos_conocida_mm = pos_ctrl_mm_seed.copy()
            cola_subobjetivos = []
          else:
            cola_subobjetivos = _generar_subobjetivos(ultima_pos_conocida_mm, target_wp)
            if cola_subobjetivos:
              print(f'🧭 Salto grande Home -> WP {wp_idx + 1}: insertando '
                    f'{len(cola_subobjetivos)} sub-objetivo(s) intermedio(s).')

          print(f'\n▶ WP {wp_idx + 1}/{len(waypoints_mm)} (target={target_wp} mm) '
                f'| Intento {intento_idx + 1}/{n_repeticiones}')

          wp_reached_time = None
          control_counter = 0
          t_intento_inicio = time.perf_counter()
          resultado_intento = None
          err_final_opti_mm = float('nan')
          # Initial value before the first real computation (see the same initialization
          # in run_control_loop) -- without this, the first cycle of each attempt
          # blows up with UnboundLocalError if --use-optitrack + action_hold_cycles>1
          # (it is only recomputed in the first Action Hold cycle, not in the 1st).
          pos_obs_mm = np.zeros(3)

          while resultado_intento is None:
            t_start = time.perf_counter()
            control_counter += 1

            en_subobjetivo = len(cola_subobjetivos) > 0
            target_activo = cola_subobjetivos[0] if en_subobjetivo else target_wp

            y_ref_mm = (
                torch.tensor(target_activo, device=self.device)
                .unsqueeze(0)
                .repeat(self.t_out, 1)
            )

            # 1. Sensors + observer estimate (for diagnostics) +
            # raw OptiTrack pose (ground truth, None if there is no fresh data).
            p_b, q_b, p_e, q_e, real_m, couple_m, tension_m, filtrar_pos_ctrl = (
                self._leer_ciclo_sensores(target_activo, home_m)
            )

            if not self.simulate and not self.use_optitrack:
              pos_obs_mm = np.asarray(p_e, dtype=np.float64) * 1000.0
            elif self.simulate or control_counter % self.action_hold_cycles == 0:
              pos_obs_mm = self._estimar_p_efector_observador() * 1000.0

            pos_opti_mm = self._optitrack_pose_relativa_mm()
            if pos_opti_mm is None:
              pos_opti_mm = np.full(3, np.nan)

            # 2. Relative kinematic transformation + causal filtering
            meta_ticks_abs = self._ticks_absolutos_desde_escalado(u_actual_scaled, home_m)
            meta_m = meta_ticks_abs.cpu().numpy().tolist()
            dict_features, pos_filt_m = self._procesar_lecturas_cinematicas(
                p_b, q_b, p_e, q_e, real_m, couple_m, tension_m, meta_m, home_m,
                filtrar_posicion=filtrar_pos_ctrl,
            )

            # 3. Normalize and save in buffer
            vec_scaled = self.scaler.scale(dict_features)
            self.buffer.append(vec_scaled)
            x_hist_tensor = torch.stack(list(self.buffer)[-self.t_in:]).unsqueeze(0)

            pos_filt_mm = pos_filt_m * 1000.0
            current_pos_mm = torch.tensor(pos_filt_mm, device=self.device, dtype=torch.float32)
            ultima_pos_conocida_mm = pos_filt_mm.copy()

            # 4-5. CEM (Action Hold) + command sending with the same
            # anti-deadband hysteresis and limits barrier as run_control_loop.
            es_ciclo_action_hold = (control_counter % self.action_hold_cycles == 0)
            bloqueado_hist_row = [0, 0, 0, 0]
            en_limite_row = [0, 0, 0, 0]
            cost_val = float('nan')
            t_calc_ms = 0.0
            pred_tout_mm_row = [float('nan')] * 3

            if es_ciclo_action_hold:
              u_opt, cost_val, t_calc_ms, _, _, y_pred_final_mm = self.mpc.optimize(
                  x_hist_tensor=x_hist_tensor, y_ref_mm=y_ref_mm,
                  u_current_scaled=u_actual_scaled, current_pos_mm=current_pos_mm,
              )
              # Model prediction for step t+t_out under the chosen
              # command -- to compare in the print/CSV against Ctrl/Opti.
              pred_tout_mm_row = y_pred_final_mm[-1].cpu().numpy().tolist()
              # t+1 and t+t_out of the last recompute cycle -- persists until
              # the end of the attempt, for the final SUMMARY (see below).
              ultimo_pred_t1_mm = y_pred_final_mm[0].cpu().numpy().tolist()
              ultimo_pred_tout_mm = pred_tout_mm_row

              if not self.simulate:
                posiciones_objetivo_crudo = self._ticks_absolutos_desde_escalado(
                    u_opt, home_m
                ).cpu().numpy()
                posiciones_objetivo = np.clip(posiciones_objetivo_crudo, limits_low, limits_high)
                posiciones_objetivo = [int(round(p)) for p in posiciones_objetivo]

                margen_limite_ticks = 15
                en_el_limite = [
                    ids[i] for i in range(len(ids))
                    if posiciones_objetivo_crudo[i] <= limits_low[i] + margen_limite_ticks
                    or posiciones_objetivo_crudo[i] >= limits_high[i] - margen_limite_ticks
                ]
                en_limite_row = [1 if mid in en_el_limite else 0 for mid in ids]
                if en_el_limite:
                  print(f'⚠️ Motor(es) {en_el_limite} en/cerca del límite físico.')

                if self.ultimo_comando_ticks is None:
                  self.ultimo_comando_ticks = [int(home_m[f'm{i}']) for i in range(1, 5)]

                tope_acumulador = self.delta_min_ticks * 1.5
                posiciones_finales = []
                for idx_motor, (prev, nuevo) in enumerate(
                    zip(self.ultimo_comando_ticks, posiciones_objetivo)
                ):
                  self.acumulador_hist_ticks[idx_motor] += (
                      (nuevo - prev) * self.ganancia_integral_hist
                  )
                  self.acumulador_hist_ticks[idx_motor] = float(np.clip(
                      self.acumulador_hist_ticks[idx_motor],
                      -tope_acumulador, tope_acumulador,
                  ))
                  correccion_efectiva = self.acumulador_hist_ticks[idx_motor]
                  if abs(correccion_efectiva) < self.delta_min_ticks:
                    posiciones_finales.append(prev)
                    bloqueado_hist_row[idx_motor] = 1
                  else:
                    nuevo_final = int(round(np.clip(
                        prev + correccion_efectiva,
                        limits_low[idx_motor], limits_high[idx_motor],
                    )))
                    posiciones_finales.append(nuevo_final)
                    self.acumulador_hist_ticks[idx_motor] = 0.0
                self.ultimo_comando_ticks = posiciones_finales

                with self.hardware_lock:
                  self.controlador_dynamixel.move(
                      ids, posiciones_finales, speed=cfg.DEFAULT_SPEED,
                      wait_for_reached=False,
                  )
                u_actual_scaled = self._escalado_desde_ticks_absolutos(posiciones_finales, home_m)
              else:
                u_actual_scaled = u_opt

            # 6. Cycle row -- accumulated in memory; it is dumped entirely
            # at the end of the attempt (so that 'resultado' can be marked in the
            # last row without having to rewrite the CSV afterwards).
            diff_mm = pos_filt_mm - target_wp
            err_mm_val = float(np.linalg.norm(diff_mm))
            diff_opti_mm = pos_opti_mm - target_wp
            err_opti_mm_val = (
                float(np.linalg.norm(diff_opti_mm))
                if not np.isnan(pos_opti_mm).any() else float('nan')
            )

            cmd_ticks_row = (
                list(self.ultimo_comando_ticks) if self.ultimo_comando_ticks is not None
                else [int(home_m[f'm{i}']) for i in range(1, 5)]
            )
            fila = (
                [time.time(), wp_idx, intento_idx]
                + target_wp.tolist()
                + pos_filt_mm.tolist()
                + pos_obs_mm.tolist()
                + pos_opti_mm.tolist()
                + [err_mm_val] + diff_mm.tolist()
                + [err_opti_mm_val] + diff_opti_mm.tolist()
                + u_actual_scaled.cpu().numpy().tolist()
                + [float(t) for t in real_m]
                + [float(t) for t in cmd_ticks_row]
                + bloqueado_hist_row
                + en_limite_row
                + [int(es_ciclo_action_hold), t_calc_ms, cost_val, '']
            )
            filas_intento.append(fila)

            # 7. Evaluate sub-goal / arrival (dwell) / timeout.
            # dist_mm (err_mm_val/err_opti_mm_val) is ALWAYS against target_wp
            # (the final waypoint) -- correct for the reported success/failure
            # criterion, but it is NOT useful for deciding when to move on to the next
            # intermediate sub-goal (if target_activo != target_wp, the robot
            # may be exactly on the sub-goal and still far from
            # target_wp -- the queue would never advance). For that a separate
            # distance is used, against target_activo, same as run_control_loop
            # (there target_wp IS REASSIGNED to the current sub-goal every cycle;
            # here it is kept fixed at the final destination during the whole attempt,
            # so this separate distance is needed).
            dist_mm = err_mm_val
            dist_a_target_activo = float(np.linalg.norm(pos_filt_mm - target_activo))

            if es_ciclo_action_hold:
              opti_str = (
                  f'{np.round(pos_opti_mm, 1)} mm' if not np.isnan(pos_opti_mm).any() else 'N/D'
              )
              # Err (ctrl and opti) are ALWAYS computed against target_wp (the real
              # final waypoint) -- that is why the print must also
              # show target_wp as "Target", never target_activo. If
              # target_activo is an intermediate sub-goal (target_activo !=
              # target_wp), it is shown separately and clearly labeled, so as not to
              # confuse it with the destination against which the error is measured
              # (before, the print showed target_activo under the label
              # "Target:", giving a visual distance that did not match the
              # computed Err -- confusing but it was not a computation error).
              err_opti_str = f'{err_opti_mm_val:.2f} mm' if not np.isnan(err_opti_mm_val) else 'N/D'
              subobj_str = (
                  f' (vía sub-objetivo: {np.round(target_activo, 1)} mm)' if en_subobjetivo else ''
              )
              t_transcurrido = time.perf_counter() - t_intento_inicio
              pred_tout_str = np.round(np.asarray(pred_tout_mm_row), 1).tolist()
              print(f'📍 [WP {wp_idx + 1} intento {intento_idx + 1}] Target: '
                    f'{target_wp} mm{subobj_str} | Opti: {opti_str} | '
                    f'Err(ctrl): {dist_mm:.2f} mm | Err(opti): {err_opti_str} | '
                    f'Pred(t+{self.mpc.t_out}): {pred_tout_str} mm | '
                    f't={t_transcurrido:.1f}/{timeout_intento_s:.0f}s')

            if en_subobjetivo:
              # Internal navigation (chopping large jumps): distance to the
              # current SUB-GOAL, not to the final destination -- it is not what
              # is reported as the experiment result, it only decides when to
              # move on to the next point in the queue.
              if dist_a_target_activo <= TOLERANCIA_SUBOBJETIVO_MM:
                cola_subobjetivos.pop(0)
            else:
              # Real SUCCESS/FAILURE criterion: against pos_opti (ground
              # truth), not against pos_ctrl. If OptiTrack has no fresh
              # frame (err_opti_mm_val = NaN), the numeric comparison gives
              # False on its own -- that cycle counts neither for advancing the dwell nor
              # for resetting it, it is simply "paused" until there is
              # fresh data again (it does not falsify the result with stale data).
              if err_opti_mm_val <= target_tolerance_mm:
                if wp_reached_time is None:
                  wp_reached_time = time.perf_counter()
                elif (time.perf_counter() - wp_reached_time) >= hold_time_s:
                  resultado_intento = 'exito'
                  err_final_opti_mm = err_opti_mm_val
              elif err_opti_mm_val > dwell_reset_tolerance_mm:
                wp_reached_time = None

            if resultado_intento is None and (time.perf_counter() - t_intento_inicio) > timeout_intento_s:
              resultado_intento = 'fallo'
              err_final_opti_mm = err_opti_mm_val
              print(f'⏱️ Timeout ({timeout_intento_s:.0f}s) sin alcanzar tolerancia -- fallo.')

            # 8. Pacing at 60 Hz
            t_ejec = time.perf_counter() - t_start
            tiempo_espera = intervalo - t_ejec
            if tiempo_espera > 0:
              time.sleep(tiempo_espera)

          # End of the attempt: mark the result in the LAST row and dump
          # the whole block to the logger (only now, with 'resultado' already set).
          filas_intento[-1][-1] = resultado_intento
          for fila in filas_intento:
            self.log_queue.put(fila)

          duracion_s = time.perf_counter() - t_intento_inicio
          simbolo = '✅' if resultado_intento == 'exito' else '❌'
          pred_t1_str = np.round(np.asarray(ultimo_pred_t1_mm), 1).tolist()
          pred_tout_str = np.round(np.asarray(ultimo_pred_tout_mm), 1).tolist()
          print(f'{simbolo} WP {wp_idx + 1} intento {intento_idx + 1}: '
                f'{resultado_intento.upper()} (err_opti final='
                f'{err_final_opti_mm:.2f} mm, {duracion_s:.1f}s) | '
                f'Pred(t+1)={pred_t1_str} mm | Pred(t+{self.mpc.t_out})={pred_tout_str} mm\n')
          resumen.append((
              wp_idx, intento_idx, resultado_intento, err_final_opti_mm, duracion_s,
              ultimo_pred_t1_mm, ultimo_pred_tout_mm,
          ))

    except KeyboardInterrupt:
      print('\n🛑 Interrumpido por el usuario (Ctrl+C). Cerrando de forma segura...')

    finally:
      self._shutdown_hardware()
      self.logging_running = False
      log_thread.join(timeout=2.0)
      if self._csv_logger is not None:
        self._csv_logger.close()

      print(f"\n{'=' * 70}\nRESUMEN DEL EXPERIMENTO\n{'=' * 70}")
      exitos_por_wp = {}
      for wp_idx, intento_idx, resultado, err_mm, dur, pred_t1, pred_tout in resumen:
        simbolo = '✅' if resultado == 'exito' else '❌'
        pred_t1_str = np.round(np.asarray(pred_t1), 1).tolist()
        pred_tout_str = np.round(np.asarray(pred_tout), 1).tolist()
        print(f'{simbolo} WP {wp_idx + 1} ({waypoints_mm[wp_idx]}) intento '
              f'{intento_idx + 1}: {resultado} | err_opti={err_mm:.2f} mm | {dur:.1f}s | '
              f'Pred(t+1)={pred_t1_str} mm | Pred(t+{self.mpc.t_out})={pred_tout_str} mm')
        exitos_por_wp.setdefault(wp_idx, []).append(resultado == 'exito')
      print(f"{'-' * 70}")
      for wp_idx, resultados in exitos_por_wp.items():
        n_ok = sum(resultados)
        print(f'WP {wp_idx + 1}: {n_ok}/{len(resultados)} éxitos')

      # Separate summary in CSV (one row per attempt, with target + t+1/t+t_out
      # predictions) -- to have it saved and be able to compare/plot in the
      # report without having to reprocess the whole 60Hz log.
      resumen_path = os.path.abspath(f'mpc_experimento_repeticiones_resumen_{timestamp}.csv')
      with open(resumen_path, 'w', newline='') as f_resumen:
        writer = csv.writer(f_resumen)
        writer.writerow([
            'wp_idx', 'target_x_mm', 'target_y_mm', 'target_z_mm', 'intento_idx',
            'resultado', 'err_opti_final_mm', 'duracion_s',
            'pred_t1_x_mm', 'pred_t1_y_mm', 'pred_t1_z_mm',
            f'pred_t{self.mpc.t_out}_x_mm', f'pred_t{self.mpc.t_out}_y_mm', f'pred_t{self.mpc.t_out}_z_mm',
        ])
        for wp_idx, intento_idx, resultado, err_mm, dur, pred_t1, pred_tout in resumen:
          writer.writerow([
              wp_idx + 1, *waypoints_mm[wp_idx].tolist(), intento_idx + 1,
              resultado, err_mm, dur, *pred_t1, *pred_tout,
          ])
      print(f'📄 Resumen (con Pred t+1/t+{self.mpc.t_out} por intento) guardado en: {resumen_path}')
      print(f'📄 Registros completos en: {log_path}')


# ==============================================================================
# 7. ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Lazo de control MPC en tiempo real para el robot continuo.'
  )
  parser.add_argument(
      '--no-simulate', action='store_true',
      help='Desactiva el modo simulación e inicializa el hardware real.'
  )
  parser.add_argument(
      '--use-optitrack', dest='use_optitrack', action='store_true',
      help='Usar la posición real de la cámara OptiTrack (por defecto).'
  )
  parser.add_argument(
      '--no-optitrack', dest='use_optitrack', action='store_false',
      help='Estimar la pose con el observador LSTM/cinemático en lugar de OptiTrack.'
  )
  parser.set_defaults(use_optitrack=True)
  parser.add_argument('--port', type=str, default=cfg.SERIAL_PORT, help='Puerto serie Dynamixel.')
  parser.add_argument('--baud', type=int, default=cfg.BAUDRATE, help='Baudrate Dynamixel.')
  parser.add_argument(
      '--action-hold-cycles', type=int, default=4,
      help='Ciclos de sensado (60 Hz) por cada recálculo/envío de comando MPC. '
           '4 -> ~15 Hz de comando (por defecto).'
  )
  parser.add_argument(
      '--delta-min-ticks', type=float, default=20.0,
      help='Salto mínimo en ticks para reenviar comando a un motor; por debajo '
           'se asume absorbido por la holgura del cable (por defecto 20).'
  )
  parser.add_argument(
      '--target-tolerance-mm', type=float, default=10.0,
      help='Tolerancia (mm) para considerar alcanzado un waypoint (por defecto 10).'
  )
  parser.add_argument(
      '--hold-time-s', type=float, default=0.4,
      help='Segundos que debe permanecer dentro de tolerancia antes de pasar '
           'al siguiente waypoint (por defecto 0.4).'
  )
  parser.add_argument(
      '--dwell-reset-tolerance-mm', type=float, default=14.0,
      help='Banda de histéresis del dwell: el conteo de hold_time_s solo se '
           'reinicia si el error supera este valor (mayor a '
           'target-tolerance-mm); rebotes de ruido dentro de la banda no '
           'reinician el conteo (por defecto 14).'
  )
  parser.add_argument(
      '--ema-alpha', type=float, default=1.0,
      help='Factor de suavizado exponencial (LERP) del comando adoptado, en '
           '[0,1]. 1.0 = sin suavizado (por defecto; CEM ya suaviza '
           'internamente). Bajar esto combinado con --delta-min-ticks puede '
           'congelar el comando por completo (doble amortiguamiento) -- '
           'si lo bajás, considera reducir --delta-min-ticks también.'
  )
  parser.add_argument(
      '--setpoint-lock-pct', type=float, default=0.0,
      help='Fracción [0,1] de avance real hacia el target vigente que se '
           'exige antes de adoptar un nuevo target del optimizador. '
           'DESACTIVADO por defecto (0.0): si el robot no logra cubrir ese '
           '%%, el sistema deja de reenviar comandos y queda bloqueado sin '
           'recuperación posible. Usar con cuidado.'
  )
  parser.add_argument(
      '--num-samples', type=int, default=500,
      help='Candidatos totales evaluados por el MPC en cada recálculo '
           '(coarse + fine pass). Bajarlo reduce el tiempo de cálculo por '
           'ciclo (por defecto 500).'
  )
  parser.add_argument(
      '--experimento-repeticiones', action='store_true',
      help='En vez de correr la trayectoria continua normal (run_control_loop), '
           'corre el protocolo de validación (run_experimento_repeticiones): '
           'cada waypoint de WAYPOINTS_EXPERIMENTO_MM se intenta --n-repeticiones '
           'veces, arrancando siempre desde HOME.'
  )
  parser.add_argument(
      '--n-repeticiones', type=int, default=3,
      help='Repeticiones por waypoint en modo --experimento-repeticiones (por defecto 3).'
  )
  parser.add_argument(
      '--timeout-intento-s', type=float, default=18.0,
      help='Tiempo límite (s) por intento antes de marcarlo como fallo, en modo '
           '--experimento-repeticiones (por defecto 18).'
  )
  parser.add_argument(
      '--fraccion-semilla', type=float, default=1.0,
      help='Fracción (0,1] del camino Home->semilla (en espacio de ticks) que '
           'se manda en lazo abierto antes de entregarle el control al CEM, '
           'en modo --experimento-repeticiones con USAR_SEMILLAS_FIJAS=True '
           '(por defecto 1.0 = bypass completo). Con 0.3-0.5 se le da un '
           'empujón parcial al CEM y se deja que complete el resto solo.'
  )
  parser.add_argument(
      '--w-limite', type=float, default=0.05,
      help='Peso de la barrera de límites/confort en el costo del CEM (por '
           'defecto 0.05). Bajarlo (o ponerlo en 0) relaja cuánto penaliza '
           'acercarse al límite físico de cada motor -- diagnóstico para '
           'targets que puedan requerir operar cerca del límite.'
  )
  parser.add_argument(
      '--margen-confort-ticks', type=float, default=100.0,
      help='Ancho (ticks) de la zona de confort antes de cada límite físico '
           'que penaliza w_limite (por defecto 100). Bajarlo permite que el '
           'CEM se acerque más al límite sin ser penalizado.'
  )
  parser.add_argument(
      '--usar-anclas-fijas-conocidas', action='store_true',
      help='Agrega los ticks de SEED_COMMANDS_TICKS (semillas confirmadas a '
           'mano para puntos difíciles) como anclas EXTRA del multi-arranque '
           '-- compiten con las demás anclas por costo, no fuerzan nada '
           '(a diferencia de USAR_SEMILLAS_FIJAS, que bypassea el CEM).'
  )
  parser.add_argument(
      '--modo-control', type=str, choices=['cem', 'jacobiano', 'hibrido'],
      default='hibrido',
      help="Método de optimización del NeuralMPCController: 'cem' (solo "
           "Cross-Entropy Method, sin ancla de Jacobiano), 'jacobiano' (solo "
           "cinemática inversa vía Jacobiano local con re-linealización, "
           "sin CEM/multi-arranque -- ver --k-iters-jacobiano), o 'hibrido' "
           "(default, comportamiento de siempre: Jacobiano como una ancla "
           "más del multi-arranque del CEM)."
  )
  parser.add_argument(
      '--k-iters-jacobiano', type=int, default=2,
      help='Iteraciones de re-linealización local del ancla de Jacobiano '
           '(Gauss-Newton local): en cada paso se reevalúa el predictor Y '
           'el Jacobiano en el punto ya corregido por el paso anterior, en '
           'vez de una sola linealización congelada en el punto de '
           'partida. 1 = comportamiento histórico de un solo paso; 2-3 '
           'recomendado (por defecto 2). Aplica tanto al modo '
           "'jacobiano' puro como al ancla de Jacobiano del multi-arranque "
           "en modo 'hibrido'."
  )
  args = parser.parse_args()

  simulate = not args.no_simulate

  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

  PREDICTOR_PATH = os.path.join(SCRIPT_DIR, 'best_mpc_pinn_predictor_8_10_dir.pth')
  #best_mpc_pinn_predictor_8_10_dir
  #best_mpc_pinn_predictor_3
  OBSERVER_PATH = os.path.join(SCRIPT_DIR, 'soft_robot_lstm_v8_17_w90_vel.pth')
  METADATA_PRED_PATH = os.path.join(
      SCRIPT_DIR, 'dataset_pred_v07_completo_10_sinRot_directo_filt_params.json'
      #dataset_pred_v07_completo_10_sinRot_directo_filt_params.json
      #dataset_pred_v08_completo_conRot_directo_filt_params.json
  )
  METADATA_OBS_PATH = os.path.join(
      SCRIPT_DIR, 'dataset_v08_completo_sinRot_filt_params.json'
  )

  archivos_requeridos = [
      PREDICTOR_PATH,
      OBSERVER_PATH,
      METADATA_PRED_PATH,
      METADATA_OBS_PATH,
  ]
  for ruta in archivos_requeridos:
    if not os.path.exists(ruta):
      raise FileNotFoundError(
          f"\n❌ ERROR: No se encontró el archivo:\n   {ruta}\n"
          "   Asegúrate de copiarlo dentro de la carpeta 'control'."
      )

  anclas_fijas_ticks = (
      list(SEED_COMMANDS_TICKS.values()) if args.usar_anclas_fijas_conocidas else None
  )

  sistema = SoftRobotMPCSystem(
      predictor_path=PREDICTOR_PATH,
      observer_path=OBSERVER_PATH,
      metadata_pred_path=METADATA_PRED_PATH,
      metadata_obs_path=METADATA_OBS_PATH,
      simulate=simulate,
      use_optitrack=args.use_optitrack,
      port=args.port,
      baud=args.baud,
      action_hold_cycles=args.action_hold_cycles,
      delta_min_ticks=args.delta_min_ticks,
      ema_alpha=args.ema_alpha,
      setpoint_lock_pct=args.setpoint_lock_pct,
      num_samples=args.num_samples,
      w_limite=args.w_limite,
      margen_confort_ticks=args.margen_confort_ticks,
      anclas_fijas_ticks=anclas_fijas_ticks,
      modo_control=args.modo_control,
      k_iters_jacobiano=args.k_iters_jacobiano,
  )

  if args.experimento_repeticiones:
    sistema.run_experimento_repeticiones(
        WAYPOINTS_EXPERIMENTO_MM, HOME_MOTORS,
        n_repeticiones=args.n_repeticiones,
        target_tolerance_mm=args.target_tolerance_mm,
        hold_time_s=args.hold_time_s,
        dwell_reset_tolerance_mm=args.dwell_reset_tolerance_mm,
        timeout_intento_s=args.timeout_intento_s,
        fraccion_semilla=args.fraccion_semilla,
    )
  else:
    sistema.run_control_loop(
        WAYPOINTS_MM, HOME_MOTORS,
        target_tolerance_mm=args.target_tolerance_mm,
        hold_time_s=args.hold_time_s,
        dwell_reset_tolerance_mm=args.dwell_reset_tolerance_mm,
    )