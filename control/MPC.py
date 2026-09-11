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
# 0. CONFIGURACIÓN DE RUTAS E IMPORTS DE MÓDULOS DEL PROYECTO
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

# Importar solo los módulos que realmente existen en candidate.py
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
# 1. ARQUITECTURA DEL OBSERVADOR LSTM
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
# 2. CARGADOR INTELIGENTE DE MODELOS / CHECKPOINTS
# ==============================================================================
def _cargar_modelo_pytorch(path, device, tipo_modelo='predictor'):
  """Carga inteligente de modelos PyTorch (.pth / .pt).

  Devuelve (model, window_size). window_size es la ventana T_IN con la que
  se ENTRENÓ ese modelo en particular (puede diferir entre predictor y
  observador -- ver checkpoint['window_size']/['t_in']); None si el
  checkpoint no la declara (TorchScript, nn.Module suelto, o un state_dict
  viejo sin esa clave), en cuyo caso el llamador debe usar un fallback.
  """
  # 1. Intentar TorchScript JIT
  try:
    return torch.jit.load(path, map_location=device).to(device), None
  except Exception:
    pass

  # 2. Cargar con torch.load
  checkpoint = torch.load(path, map_location=device)

  # Caso A: Ya es un objeto nn.Module completo
  if isinstance(checkpoint, torch.nn.Module):
    return checkpoint.to(device), None

  # Caso B: Es un diccionario Checkpoint
  if isinstance(checkpoint, dict):
    print(
        f"📦 Checkpoint detectado en '{os.path.basename(path)}'."
        ' Reconstruyendo red...'
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    window_size = checkpoint.get('window_size', checkpoint.get('t_in'))

    if tipo_modelo == 'predictor':
      # Extraer los parámetros exactos con los que fue guardado
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
# 3. PUNTOS OBJETIVO Y CONFIGURACIÓN INICIAL (EN MM)
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

# 🧪 Waypoints para run_experimento_repeticiones: los 5 de WAYPOINTS_MM más
# los 2 casos de borde que estaban comentados arriba -- son justo los que
# en producción mostraron el problema de límites/redundancia cinemática
# (ver USAR_SEMILLAS_FIJAS/SEED_COMMANDS_TICKS más abajo), así que interesa
# incluirlos explícitamente en la validación con repeticiones.
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

# 🌱 Comandos semilla por TARGET -- DESACTIVADOS (USAR_SEMILLAS_FIJAS=False).
# Se usaron para diagnosticar si el punto [-153,226,81]/[-63.6,278.9,-34]
# eran alcanzables. Ahora que existe el multi-arranque del CEM (ver
# NeuralMPCController.optimize), sembrar a mano sería "hacer trampa" para
# evaluar si el multi-arranque solo puede encontrar buenas soluciones sin
# ayuda -- se deja el diccionario como referencia/comparación, pero no se
# usa para controlar el robot. Poner en True para volver a activarlo.
#
# CORREGIDO (18/08): los valores de m4 originales (1104/1111) se habían
# tomado de un frame EN TRANSICIÓN (el motor todavía moviéndose entre
# lecturas consecutivas de waypoints_relativos_largo.csv, sin mantenerse
# igual más de una fila), no del valor realmente asentado -- de ahí el
# error reportado antes (5.8mm/~13mm). Verificado contra las mesetas reales
# del CSV (m4 constante >=10 filas seguidas, motor detenido): el valor
# asentado más cercano a cada target da ~1.45mm en ambos casos.
USAR_SEMILLAS_FIJAS = True
SEED_COMMANDS_TICKS = {
    (-153.0, 226.0, 81.0): [1876, 1406, 2130, 1079],  # asentado: 1.45mm de error
    (-63.6, 278.9, -34.0): [1601, 1682, 1481, 1080],  # asentado: 1.45mm de error
    (-22.35, 224.55, -179.51): [1599, 2229, 842, 1076],  # asentado: 0.07mm de error
    # (-40.57, 310.78, 2.48) NO tiene meseta genuina cerca en waypoints_relativos_largo.csv
    # (la lectura "exacta" que parecía matchear era un frame en transición, no un
    # punto realmente asentado) -- no se agrega semilla para ese, ya llegaba 2/3
    # con el CEM normal antes de aflojar la barrera de límites.
}
SEED_TARGET_TOLERANCIA_MM = 1.0
SEED_TOLERANCIA_TICKS = 15.0
SEED_TIMEOUT_S = 15.0


def _buscar_semilla_para_target(target_wp, tolerancia_mm=SEED_TARGET_TOLERANCIA_MM):
  """Devuelve el comando semilla configurado para este target (por posición
  cartesiana, con tolerancia), o None si no hay ninguno definido para él."""
  target_np = np.asarray(target_wp, dtype=np.float64)
  for target_key, ticks in SEED_COMMANDS_TICKS.items():
    if np.linalg.norm(np.asarray(target_key, dtype=np.float64) - target_np) <= tolerancia_mm:
      return ticks
  return None


# 🧭 Waypoints intermedios: si el salto directo al próximo waypoint es
# grande, el CEM/Jacobiano (métodos LOCALES) pueden quedar atrapados en la
# rama de redundancia cinemática más cercana a donde está el robot, en vez
# de la rama correcta (confirmado en producción: para un target específico,
# tanto el CEM como el Jacobiano insistían en empujar un motor a su límite,
# mientras la combinación real que lo alcanzaba en los datos de
# entrenamiento estaba del otro lado de home). Trocear el salto en pasos
# chicos preserva la continuidad en el espacio de actuadores entre ciclos
# consecutivos (la misma técnica usada en IK Jacobiana para sistemas
# redundantes) sin necesitar una tabla de referencia. No hace falta que el
# camino intermedio imite la forma real (arco) del robot -- solo que cada
# sub-objetivo esté cerca del anterior; el robot curva como le resulte
# natural para alcanzar cada uno.
DISTANCIA_MAX_SALTO_MM = 60.0
PASO_SUBOBJETIVO_MM = 50.0
TOLERANCIA_SUBOBJETIVO_MM = 25.0


def _generar_subobjetivos(pos_actual_mm, target_mm, paso_mm=PASO_SUBOBJETIVO_MM,
                           salto_min_mm=DISTANCIA_MAX_SALTO_MM):
  """Puntos intermedios en línea recta entre pos_actual_mm y target_mm, sin
  incluir el punto final (ese sigue tratándose como el waypoint real, con
  su tolerancia/dwell normales). Lista vacía si el salto no es grande."""
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
# 4. FILTRADO CAUSAL EN TIEMPO REAL
# ==============================================================================
class RealTimeCausalFilter:

  def __init__(self, cutoff_hz, fs_hz=60.0, order=2, num_channels=1):
    self.nyquist = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / self.nyquist
    self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
    self.num_channels = num_channels
    zi_single = lfilter_zi(self.b, self.a)
    self.zi = np.tile(zi_single, (num_channels, 1)).T  # Forma: (order, num_channels)

  def filter(self, x):
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.zeros(self.num_channels, dtype=np.float64)
    for i in range(self.num_channels):
      out, zf = lfilter(self.b, self.a, [x[i]], zi=self.zi[:, i])
      y[i] = out[0]  # Extraer escalar explícitamente
      self.zi[:, i] = zf
    return y


# ==============================================================================
# 5. ESCALADOR Y NORMALIZADOR DINÁMICO
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
# 6. SISTEMA PRINCIPAL DE CONTROL CLOSED-LOOP MPC
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

    # 🎯 Action Hold (control decimado): el sensado corre a 60 Hz pero el MPC
    # solo recalcula/envía comando cada N ciclos, dando tiempo al cable de
    # tensarse y deformar la silicona antes de evaluar el siguiente paso.
    self.action_hold_cycles = max(1, int(action_hold_cycles))
    # Umbral mínimo de salto (ticks): si |comando - último enviado| es menor,
    # el movimiento se absorbe por la holgura del cable y no se reenvía.
    self.delta_min_ticks = delta_min_ticks
    self.ultimo_comando_ticks = None

    # 🔺 Acumulador anti-deadband (efecto integrador) por motor: si el CEM
    # pide consistentemente una corrección más chica que delta_min_ticks,
    # sin esto se pierde CADA ciclo (comparación sin memoria) y el sistema
    # queda estancado sin poder afinar el último tramo (visto en producción:
    # no lograba bajar de ~15mm a ~8mm). Se acumula una fracción de la
    # corrección pedida cada ciclo hasta superar el umbral, y ahí se libera
    # de una vez. Anti-windup: tope al acumulador (ver _reset_hist_acum) y
    # se resetea al cambiar de waypoint.
    self.acumulador_hist_ticks = [0.0, 0.0, 0.0, 0.0]
    self.ganancia_integral_hist = 0.3
    self._ultimo_target_wp_hist = None

    # 🌊 Suavizado exponencial (LERP) del comando adoptado, en espacio escalado
    # [-1,1]: u_enviado(t) = alpha*u_opt + (1-alpha)*u_enviado(t-1). DESACTIVADO
    # por defecto (alpha=1.0 -> u_enviado=u_opt sin mezclar) porque CEM ya
    # promedia/suaviza internamente (media del élite + std con memoria); EMA
    # apilado ENCIMA de eso reduce el cambio a algo tan chico que la histéresis
    # (delta_min_ticks) lo descarta siempre -> el comando se congela por
    # completo (visto en producción: Comando y Posición motor nunca cambian).
    self.ema_alpha = float(ema_alpha)

    # 🔒 Setpoint Lock (Waypoint Tracking / umbral de avance): no se adopta un
    # nuevo target del optimizador hasta que la posición real haya recorrido
    # `setpoint_lock_pct` (p.ej. 0.8 = 80%) de la distancia entre el origen y
    # el target vigente. DESACTIVADO por defecto (0.0): si el robot no llega
    # a cubrir ese % (fricción/holgura/torque insuficiente), el sistema deja
    # de reenviar comandos y queda bloqueado para siempre (nada vuelve a
    # actualizar el origen/target salvo una adopción exitosa). Con EMA y la
    # memoria de `std` del CEM ya no debería hacer falta: ambos ya evitan que
    # el optimizador salte bruscamente de un target a otro.
    self.setpoint_lock_pct = float(setpoint_lock_pct)
    self.target_lock_origen_ticks = None

    self.num_samples = int(num_samples)
    self._home_ticks_cache = None  # cacheado en la 1ra llamada (home_m es fijo por corrida)

    # Handles de hardware (se crean en _setup_hardware)
    self.controlador_dynamixel = None
    self.streaming_client = None
    self.controlador_phidget = None
    self._csv_logger = None

    # 🧵 Hilo de sensado Dynamixel en background (igual patrón que main.py):
    # lee el bus serial en su propio loop pausado a 60 Hz; el lazo de control
    # nunca bloquea en el read, solo lee las variables ya cacheadas abajo.
    self._sensor_thread_running = False
    self._sensor_thread_obj = None

    # 🚀 Estado compartido de los cuerpos rígidos OptiTrack (actualizado async)
    self.p_base = [0.0, 0.0, 0.0]
    self.q_base = [0.0, 0.0, 0.0, 1.0]
    self.p_efector = [0.0, 0.0, 0.0]
    self.q_efector = [0.0, 0.0, 0.0, 1.0]
    # Marca de tiempo del último frame recibido por cuerpo rígido (None = nunca).
    # Sirve para distinguir "OptiTrack en el origen real" de "OptiTrack nunca conectó".
    self._t_ultimo_frame_base = None
    self._t_ultimo_frame_efector = None

    # Última lectura válida de motores (fallback ante fallos de comunicación)
    self.ultimas_posiciones_validas = [cfg.HOME_POSITION[mid] for mid in cfg.MOTOR_IDS]
    self.ultimos_torques_validos = [0] * len(cfg.MOTOR_IDS)
    self.ultimas_fuerzas_validas = [0.0] * 4

    # 1. Cargar Metadatos JSON
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

    # 2. Cargar Modelos PyTorch
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

    # El observador puede haberse entrenado con una ventana T_IN distinta a
    # la del predictor (checkpoint['window_size']); si el checkpoint no la
    # declara (formatos viejos), caer al valor con el que se sabe que se
    # entrenó el modelo desplegado actualmente (90).
    if window_size_obs is None:
      window_size_obs = self.meta_obs.get('window_size', 90)
    self.t_in_obs = int(window_size_obs)
    print(f'⚙️ Ventana del observador: t_in_obs={self.t_in_obs} pasos')

    # Parámetros para desescalar la salida del observador (rel_x, rel_y, rel_z)
    # cuando no hay OptiTrack disponible (--no-optitrack)
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

    # 3. Inicializar Normalizador y MPC
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

    # 4. Filtros Causales
    self.filter_pos = RealTimeCausalFilter(
        cutoff_hz=6.0, fs_hz=60.0, num_channels=3
    )
    self.filter_torques = RealTimeCausalFilter(
        cutoff_hz=3.5, fs_hz=60.0, num_channels=4
    )
    self.filter_tensions = RealTimeCausalFilter(
        cutoff_hz=3.5, fs_hz=60.0, num_channels=4
    )

    # 5. Búfer Dinámico -- dimensionado para el más largo de los dos modelos
    # (predictor: self.t_in, observador: self.t_in_obs), cada uno consume
    # solo el tramo final que necesita (ver _estimar_p_efector_observador y
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
    t_actual=None,  # Timestamp opcional
    filtrar_posicion=True,
  ):
    # 1. Cálculo dinámico de delta_t (60 Hz por defecto en el 1er paso)
    if t_actual is None:
        t_actual = time.perf_counter()

    if getattr(self, 't_prev', None) is None:
        delta_t = 1.0 / 60.0  # ~0.01667 segundos
    else:
        delta_t = t_actual - self.t_prev

    self.t_prev = t_actual

    # 2. Cinemática relativa y Cuaterniones
    r_b = R.from_quat(q_base)
    r_e = R.from_quat(q_efector)

    p_rel = r_b.inv().apply(np.array(p_efector) - np.array(p_base))
    r_rel = r_b.inv() * r_e
    q_rel = r_rel.as_quat()

    # 3. Filtrado Causal
    # La posición SOLO se filtra (Butterworth 6 Hz) cuando la fuente es una
    # lectura cruda con jitter (OptiTrack real o el ruido sintético del modo
    # simulate). Cuando la fuente es la salida del observador LSTM
    # (--no-optitrack), esa señal YA fue entrenada contra targets de OptiTrack
    # suavizados a 6 Hz — filtrarla de nuevo sería doble filtrado y metería
    # retardo de fase innecesario en el lazo de control.
    if filtrar_posicion:
      pos_filt = self.filter_pos.filter(p_rel)
    else:
      pos_filt = np.asarray(p_rel, dtype=np.float64).flatten()
    torques_filt = self.filter_torques.filter(couple_m)
    tensiones_filt = self.filter_tensions.filter(tension_m)

    # 4. Construir diccionario (Añadiendo 'delta_t')
    dict_data = {
        'rel_x': pos_filt[0],
        'rel_y': pos_filt[1],
        'rel_z': pos_filt[2],
        'rel_qx': q_rel[0],
        'rel_qy': q_rel[1],
        'rel_qz': q_rel[2],
        'rel_qw': q_rel[3],
        'delta_t': delta_t,  # <-- ¡Clave que solicitaba el escalador!
    }

    for i in range(1, 5):
        dict_data[f'delta_real_m{i}'] = float(real_m[i - 1] - home_m[f'm{i}'])
        dict_data[f'delta_meta_m{i}'] = float(meta_m[i - 1] - home_m[f'm{i}'])
        dict_data[f'couple_m{i}'] = torques_filt[i - 1]
        dict_data[f'tension_m{i}'] = tensiones_filt[i - 1]

    return dict_data, pos_filt

  def _on_rigid_body_frame(self, new_id, position, rotation):
    """PRODUCTOR: recibe frames asíncronos de OptiTrack (base=1, efector=2)."""
    if new_id == 1:
      self.p_base = position
      self.q_base = rotation
      self._t_ultimo_frame_base = time.time()
    elif new_id == 2:
      self.p_efector = position
      self.q_efector = rotation
      self._t_ultimo_frame_efector = time.time()

  def _optitrack_pose_relativa_mm(self, max_age=0.5):
    """Posición relativa (efector respecto a base) medida por OptiTrack, en mm.

    Devuelve None si nunca se recibió un frame de ambos cuerpos rígidos o si
    la última lectura es más vieja que `max_age` segundos (Motive caído/no
    transmitiendo), para no confundir "sin dato" con un [0,0,0] real.
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
    """Estima la posición relativa del efector (m) con el observador LSTM.

    Se usa cuando --no-optitrack está activo: cierra el lazo con la
    predicción del observador en lugar de la medición real de OptiTrack.
    """
    x_hist_tensor = torch.stack(list(self.buffer)[-self.t_in_obs:]).unsqueeze(0)
    pred = self.observer(x_hist_tensor).squeeze(0)
    if self.min_y_obs is not None:
      pos_m = self.min_y_obs + (pred + 1.0) * (self.max_y_obs - self.min_y_obs) / 2.0
    else:
      pos_m = pred
    return pos_m.cpu().numpy()

  def _home_ticks_tensor(self, home_m):
    """Tensor [4] de home en ticks, cacheado (home_m es fijo durante la corrida)."""
    if self._home_ticks_cache is None:
      self._home_ticks_cache = torch.tensor(
          [home_m[f'm{i}'] for i in range(1, 5)], device=self.device, dtype=torch.float32
      )
    return self._home_ticks_cache

  def _ticks_absolutos_desde_escalado(self, u_scaled_tensor, home_m):
    """Convierte el vector de control normalizado [-1, 1] a posiciones

    absolutas de motor (ticks), usando los rangos por motor del generador
    de candidatos y la posición HOME de referencia.
    """
    ranges = self.mpc.candidate_generator.ranges
    return self._home_ticks_tensor(home_m) + u_scaled_tensor * (ranges / 2.0)

  def _escalado_desde_ticks_absolutos(self, ticks, home_m):
    """Inversa de `_ticks_absolutos_desde_escalado`: de ticks absolutos de

    motor a vector de control normalizado [-1, 1]. Se usa para recalcular
    `u_actual_scaled` a partir de lo que REALMENTE se comandó al motor
    (después de aplicar el umbral mínimo de salto), no lo que sugirió el MPC.
    """
    ranges = self.mpc.candidate_generator.ranges
    if not torch.is_tensor(ticks):
      ticks = torch.tensor(ticks, device=self.device, dtype=torch.float32)
    return torch.clamp((ticks - self._home_ticks_tensor(home_m)) / (ranges / 2.0), -1.0, 1.0)

  def _sensor_thread_loop(self, freq_hz=60.0):
    """Hilo dedicado: lee posición+carga de Dynamixel en su propio loop

    pausado a `freq_hz`, actualizando `self.ultimas_posiciones_validas` /
    `self.ultimos_torques_validos`. El lazo de control principal NUNCA
    bloquea en este read -- solo lee esas variables ya cacheadas. Mismo
    patrón que `data_sampler_thread` en main.py.
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
        # Diagnóstico: aislar si el hueco es esperar el lock (lo tiene el
        # hilo principal mandando un move()) o la lectura serial en sí
        # (hipo de comunicación real con el bus Dynamixel).
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
    """Inicializa motores Dynamixel, OptiTrack y celdas de carga (Phidget)."""
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

    # 🧵 Iniciar el hilo de sensado Dynamixel en background (60 Hz propio).
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

    # El streaming de OptiTrack se intenta SIEMPRE que haya hardware real,
    # independientemente de si controla el lazo o no: --use-optitrack solo
    # decide qué fuente CIERRA el lazo de control; si Motive no está
    # transmitiendo, simplemente no llegan frames (sin error) y las columnas
    # de comparación OptiTrack quedan vacías en el log/consola.
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
    """Cierra de forma segura motores, OptiTrack y Phidget (idempotente)."""
    # Detener el hilo de sensado ANTES de tocar el puerto, para que no siga
    # intentando leer mientras (o después de) se cierra/deshabilita el torque.
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
    """Setup de hardware + precalentamiento GPU, UNA SOLA VEZ por corrida
    (no por intento/waypoint) -- extraído de run_control_loop para poder
    reusarlo también en run_experimento_repeticiones sin reconectar
    OptiTrack/reiniciar el hilo de sensado en cada repetición."""
    self._setup_hardware()

    # 🔥 Precalentamiento: la primera inferencia en GPU paga un costo único
    # (init de contexto CUDA, autotuning de cuDNN para la LSTM, etc.) que
    # puede tardar decenas de ms -- si eso pasa DENTRO del lazo cronometrado
    # se ve como un pico gigante en el primer/segundo ciclo (visto en
    # producción: 89ms y 70ms, cayendo a ~5-8ms desde el 3er ciclo en
    # adelante). Se paga acá, antes de arrancar a medir, con una llamada
    # descartable que ejercita el mismo código real (CEM completo).
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

    # ⏳ Esperar el primer frame válido de OptiTrack antes de precargar el
    # búfer (si no, la precarga arrancaría con lecturas viejas/cero).
    if not self.simulate and self.use_optitrack:
      print('⏳ Esperando primer frame válido de OptiTrack (base + efector)...')
      t_espera_opti = time.perf_counter()
      while self._optitrack_pose_relativa_mm() is None:
        time.sleep(0.05)
        if time.perf_counter() - t_espera_opti > 10.0:
          print('⚠️ Sin frame válido de OptiTrack tras 10s -- arrancando de todas formas.')
          break

  def _precargar_buffer_en_posicion(self, home_m, waypoint_referencia_mm, u_actual_scaled, freq_hz=60.0):
    """Refresca self.buffer con ciclos de sensor REALES (no simulados/cero).

    El búfer arranca relleno con ceros (ver __init__), y la LSTM del
    observador nunca vio ese tipo de entrada en entrenamiento -- su salida
    ahí es basura sin acotar (la capa final no tiene activación), que al
    desescalar puede dispararse a cientos de mm (visto en producción:
    pos_ctrl > 900mm en los primeros ciclos). Se llama una vez al arrancar
    la corrida, y de nuevo después de cada retorno a Home en
    run_experimento_repeticiones (para que el historial de la LSTM refleje
    la posición real actual, no la del intento anterior).

    Devuelve pos_filt_m0 (última lectura de posición filtrada, en metros).
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
    """Limpia toda la memoria entre ciclos de control (CEM + histéresis +
    setpoint lock) al arrancar un intento nuevo, para que cada repetición
    sea una prueba independiente (no "recuerde" la convergencia del intento
    anterior al mismo punto) -- ver run_experimento_repeticiones."""
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
    """Manda el robot a HOME_POSITION por comando directo de motor (sin
    pasar por el CEM -- Home es la referencia de calibración, siempre
    alcanzable en espacio de motor) y bloquea hasta que el driver confirma
    llegada. Usado entre intentos de run_experimento_repeticiones para que
    cada repetición arranque desde la MISMA posición física."""
    ids = cfg.MOTOR_IDS
    home_positions = [home_m[f'm{mid}'] for mid in ids]
    if self.simulate:
      return
    with self.hardware_lock:
      self.controlador_dynamixel.move(
          ids, home_positions, speed=cfg.DEFAULT_SPEED, wait_for_reached=True
      )

  def _leer_ciclo_sensores(self, target_wp_ref, home_m):
    """Lee un ciclo de sensores (pose del efector, motores, fuerzas).

    Extraído del lazo principal para poder reusarse también durante la
    precarga del búfer del observador (ver run_control_loop), antes de que
    arranque el control "en serio".
    """
    if self.simulate:
      p_b, q_b = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
      p_e = (target_wp_ref / 1000.0) + np.random.normal(0, 0.0005, 3)
      q_e = [0.0, 0.0, 0.0, 1.0]
      real_m = [home_m[f'm{i}'] for i in range(1, 5)]
      couple_m = [0.0] * 4
      tension_m = [0.0] * 4
      # Ruido sintético tipo cámara -> sí filtrar
      filtrar_pos_ctrl = True
    else:
      # 1a. Pose del efector: OptiTrack real o estimación con el observador
      if self.use_optitrack:
        p_b, q_b = self.p_base, self.q_base
        p_e, q_e = self.p_efector, self.q_efector
        # Lectura cruda de cámara con jitter -> sí filtrar (Butterworth 6 Hz)
        filtrar_pos_ctrl = True
      else:
        p_b, q_b = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        p_e = self._estimar_p_efector_observador()
        q_e = [0.0, 0.0, 0.0, 1.0]
        # Salida del LSTM: entrenada contra OptiTrack ya suavizado a 6 Hz -> NO refiltrar
        filtrar_pos_ctrl = False

      # 1b. Posición y torque reales de los motores Dynamixel: lectura
      # INSTANTÁNEA de lo que ya actualizó el hilo de sensado en
      # background (Perf: el lazo de control ya no bloquea esperando
      # las 8 transacciones seriales cada ciclo).
      real_m = list(self.ultimas_posiciones_validas)
      couple_m = list(self.ultimos_torques_validos)

      # 1c. Tensión de los tendones (celdas de carga Phidget)
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
    # El umbral de "error chico" del CEM (que desactiva ensanche/multi-
    # arranque por estancamiento) tiene que estar atado a la tolerancia de
    # llegada REAL de esta corrida, no a la que tenía por defecto al
    # construir self.mpc -- se actualiza acá para no depender del orden de
    # construcción.
    self.mpc.error_chico_umbral_mm = target_tolerance_mm * 2.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Usar ruta absoluta para evitar errores con os.makedirs en Windows
    log_path = os.path.abspath(f"mpc_experiment_{timestamp}.csv")

    self.logging_running = True
    log_thread = threading.Thread(
        target=self._consumer_logger, args=(log_path,), daemon=True
    )
    log_thread.start()

    # Fila de encabezado (la 1ra columna real del CSV es un timestamp UTC que
    # antepone CSVLogger.log() automáticamente a cada fila, incluida ésta).
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
    # Límites de seguridad del comando final: usan el MISMO rango con el que
    # se entrenó (self.mpc.candidate_generator.homes/ranges, del JSON), NO
    # cfg.MOTOR_HOME_RANGES/cfg.LIMITS (más chico, usado solo para la grilla
    # de muestreo de otros scripts de recolección de datos). Así el MPC puede
    # llegar a todo el rango que el modelo realmente aprendió a controlar.
    home_ticks_np = self.mpc.candidate_generator.homes.cpu().numpy()
    half_range_np = (self.mpc.candidate_generator.ranges / 2.0).cpu().numpy()
    limits_low = np.clip(home_ticks_np - half_range_np, 0, cfg.DXL_MAXIMUM_POSITION_VALUE)
    limits_high = np.clip(home_ticks_np + half_range_np, 0, cfg.DXL_MAXIMUM_POSITION_VALUE)

    # 🎯 Estado del Action Hold: el sensado corre cada ciclo (60 Hz), pero el
    # MPC solo recalcula/envía comando cada `action_hold_cycles` ciclos.
    control_counter = 0
    cost_val = 0.0
    t_calc_ms = 0.0

    # ⏱️ Estado del "dwell": instante (perf_counter) en que se entró en
    # tolerancia por primera vez para el waypoint actual. None = aún no llegó.
    wp_reached_time = None

    # Última estimación del observador calculada (para reusar en ciclos donde
    # no se recalcula, ver Perf en el paso 1d más abajo).
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
      # Igual que en cada transición de waypoint: si el primer waypoint
      # queda lejos de donde termina la precarga, trocear el salto también.
      cola_subobjetivos = _generar_subobjetivos(ultima_pos_conocida_mm, waypoints_mm[0])
      if cola_subobjetivos:
        print(f'🧭 Salto grande al WP 1: insertando {len(cola_subobjetivos)} '
              'sub-objetivo(s) intermedio(s).')

      while current_wp_idx < len(waypoints_mm):
        target_real_wp = waypoints_mm[current_wp_idx]
        en_subobjetivo = len(cola_subobjetivos) > 0
        target_wp = cola_subobjetivos[0] if en_subobjetivo else target_real_wp

        # Reset del acumulador anti-deadband al cambiar de target: no
        # arrastrar una corrección pendiente de OTRO waypoint.
        if (
            self._ultimo_target_wp_hist is None
            or not np.allclose(target_wp, self._ultimo_target_wp_hist, atol=1e-6)
        ):
          self.acumulador_hist_ticks = [0.0, 0.0, 0.0, 0.0]
          self._ultimo_target_wp_hist = target_wp.copy()
        # Las semillas fijas y el multi-arranque solo aplican al waypoint
        # REAL, no a los sub-objetivos intermedios de paso.
        seed_para_este_wp = None if en_subobjetivo else _buscar_semilla_para_target(target_real_wp)

        # 🌱 Comando semilla (ver SEED_COMMANDS_TICKS): mandar un comando
        # fijo conocido y esperar a que la posición real converja ahí ANTES
        # de activar el predictor para este waypoint puntual. Se hace una
        # sola vez por waypoint (wp_idx_sembrado evita repetirlo cada ciclo).
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

            # Registrar también los ciclos de precarga en el CSV (antes se
            # dejaban afuera -- generaba un hueco temporal que plot_resultados.py
            # dibujaba como una rampa recta engañosa entre el último punto de
            # antes de la semilla y el primero de después).
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
                + [float('nan')] * 4  # raw_opt_m1-4_ticks: no aplica (no hay CEM acá)
                + [0, 0, 0, 0]  # bloqueado_hist_m1-4: no aplica
                + [0, 0, 0, 0]  # en_limite_m1-4: no aplica
                + [0]  # es_action_hold
                + [float('nan')] * 3  # pred_tout_x/y/z_mm: no aplica
                + [0.0, float('nan')]  # t_calc_ms, cost: no aplica
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

          # Error de la semilla contra el target, ANTES de que el predictor
          # toque nada -- para saber si la combinación conocida-buena
          # todavía alcanza el punto (deriva física) o no.
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
          continue  # arrancar limpio el ciclo normal de control para este waypoint

        t_start = time.perf_counter()
        control_counter += 1

        y_ref_mm = (
            torch.tensor(target_wp, device=self.device)
            .unsqueeze(0)
            .repeat(self.t_out, 1)
        )

        # 1. Leer Sensores Crudos (Simulación / Hardware)
        p_b, q_b, p_e, q_e, real_m, couple_m, tension_m, filtrar_pos_ctrl = (
            self._leer_ciclo_sensores(target_wp, home_m)
        )

        # 1d. Estimación del observador LSTM, para comparar contra
        # OptiTrack/objetivo en el log. Si ya se calculó arriba como p_e
        # (--no-optitrack), se reutiliza (gratis, sin costo extra). Si es
        # solo para comparación (--use-optitrack), Perf: se recalcula nada
        # más al ritmo del Action Hold en vez de cada ciclo de sensado a
        # 60 Hz — es una inferencia LSTM completa que no hace falta correr
        # 4x más seguido de lo necesario. En simulate se recalcula siempre
        # (irrelevante en costo, útil para pruebas de lógica).
        if not self.simulate and not self.use_optitrack:
          pos_obs_mm = np.asarray(p_e, dtype=np.float64) * 1000.0
        elif self.simulate or control_counter % self.action_hold_cycles == 0:
          pos_obs_mm = self._estimar_p_efector_observador() * 1000.0
        # else: se reutiliza el pos_obs_mm calculado en un ciclo anterior

        # 1e. Pose real de OptiTrack para comparación (None si no hay datos
        # frescos, p.ej. Motive no está transmitiendo).
        pos_opti_mm = self._optitrack_pose_relativa_mm()
        if pos_opti_mm is None:
          pos_opti_mm = np.full(3, np.nan)

        # 2. Transformación Cinemática Relativa y Filtrado Causal
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

        # 3. Normalizar Vector [17] y Guardar en Búfer
        vec_scaled = self.scaler.scale(dict_features)
        self.buffer.append(vec_scaled)
        x_hist_tensor = torch.stack(list(self.buffer)[-self.t_in:]).unsqueeze(0)

        # Posición REAL actual (Cartesiano, mm) percibida por OptiTrack/observador
        # (la misma que se muestra como "Ctrl" en el log), para que el MPC calcule
        # su error contra el target usando el sensor real, no una suposición interna.
        pos_filt_mm = pos_filt_m * 1000.0
        current_pos_mm = torch.tensor(pos_filt_mm, device=self.device, dtype=torch.float32)
        ultima_pos_conocida_mm = pos_filt_mm.copy()

        # 4-5. Optimización MPC + envío de comando (Action Hold: solo cada
        # `action_hold_cycles` ciclos se recalcula; en los ciclos intermedios
        # se mantiene el último comando, dando tiempo al cable de tensarse y
        # deformar la silicona).
        es_ciclo_action_hold = (control_counter % self.action_hold_cycles == 0)
        # Diagnóstico por ciclo (para el CSV): sugerencia cruda del optimizador
        # antes de clip/histéresis, y si la histéresis/el límite físico
        # bloquearon la corrección de cada motor -- NaN/0 en los ciclos donde
        # no se recalcula (no action-hold) o el setpoint lock no avanza.
        raw_opt_ticks_row = [float('nan')] * 4
        bloqueado_hist_row = [0, 0, 0, 0]
        en_limite_row = [0, 0, 0, 0]
        pred_tout_mm_row = [float('nan')] * 3
        # (Se probó congelar por completo el optimizador durante el dwell,
        # pero eso resultó PEOR: el robot deriva solo mientras no se manda
        # nada -- real, no artefacto -- y al reactivarse el buffer queda
        # lleno de esa deriva "no comandada", el costo del modelo se dispara
        # y corrige de un golpe violento en vez de suave. Mejor seguir
        # corrigiendo la deriva chica en tiempo real; las otras defensas
        # -persistencia direccional, sin ensanche con error chico, tope de
        # salto- ya evitan el "baile" que motivó el freeze originalmente.)
        if es_ciclo_action_hold:
          u_opt, cost_val, t_calc_ms, top_u_scaled, top_costs, y_pred_final_mm = self.mpc.optimize(
              x_hist_tensor=x_hist_tensor,
              y_ref_mm=y_ref_mm,
              u_current_scaled=u_actual_scaled,
              current_pos_mm=current_pos_mm,
          )
          # Creencia del predictor sobre dónde va a terminar (paso t_out,
          # el que más pesa w_terminal) bajo el comando elegido -- para
          # comparar en el CSV contra lo que realmente mide el sensor unos
          # ciclos después (ver columnas pred_tout_*_mm).
          pred_tout_mm_row = y_pred_final_mm[-1].cpu().numpy().tolist()

          # Top-K candidatos de la pasada de refinamiento, para diagnóstico
          top_ticks = self._ticks_absolutos_desde_escalado(top_u_scaled, home_m).cpu().numpy()
          top_costs_np = top_costs.cpu().numpy()
          top_str = ' | '.join(
              f'#{i + 1} ticks={np.round(top_ticks[i]).astype(int).tolist()} costo={top_costs_np[i]:.3f}'
              for i in range(len(top_costs_np))
          )
          print(f'🏆 Top-{len(top_costs_np)} candidatos: {top_str}')

          if not self.simulate:
            # 🔒 Setpoint Lock: no se adopta el nuevo target del optimizador
            # hasta que la posición real haya recorrido `setpoint_lock_pct`
            # de la distancia entre el origen y el target vigente. El
            # optimizador ya calculó u_opt (sigue "pensando" en segundo
            # plano); si el avance no alcanza, simplemente no se toca el
            # comando y el servo sigue su trayectoria en curso sin interrupción.
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
              # 🌊 Suavizado exponencial (LERP) en espacio escalado: en vez de
              # saltar directo a u_opt, se interpola hacia él. alpha bajo ->
              # transición suave (evita el salto brusco del optimizador).
              u_suavizado = self.ema_alpha * u_opt + (1.0 - self.ema_alpha) * u_actual_scaled

              posiciones_objetivo_ticks = self._ticks_absolutos_desde_escalado(u_suavizado, home_m)
              posiciones_objetivo_crudo = posiciones_objetivo_ticks.cpu().numpy()
              raw_opt_ticks_row = posiciones_objetivo_crudo.tolist()
              posiciones_objetivo = np.clip(
                  posiciones_objetivo_crudo, limits_low, limits_high
              )
              posiciones_objetivo = [int(round(p)) for p in posiciones_objetivo]

              # Diagnóstico: avisar si el target empuja algún motor al límite
              # físico configurado (home ± rango). Es la señal de que el punto
              # puede estar en el borde o fuera de lo que el modelo vio en
              # entrenamiento -- el optimizador queda "pegado" al límite sin
              # una solución estable a la que converger.
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

              # Umbral mínimo de salto (histéresis) CON acumulador anti-
              # deadband: si el cambio pedido este ciclo es chico, no se
              # descarta sin más -- se suma (una fracción, ganancia_integral_hist)
              # al acumulador de ese motor. Si la corrección pedida es
              # consistente ciclo a ciclo, el acumulador cruza el umbral y
              # se libera de una vez (efecto integrador); si es ruido
              # puntual que cambia de signo, se cancela solo. Tope de
              # anti-windup para no crecer sin límite si queda bloqueado
              # por otra razón (p.ej. setpoint lock).
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
                  self.acumulador_hist_ticks[idx_motor] = 0.0  # liberado, resetear
              self.ultimo_comando_ticks = posiciones_finales
              # Nuevo origen para medir el avance hacia ESTE target
              self.target_lock_origen_ticks = list(real_m)

              t_lock_start = time.perf_counter()
              with self.hardware_lock:
                t_esperando_lock_ms = (time.perf_counter() - t_lock_start) * 1000.0
                t_move_start = time.perf_counter()
                self.controlador_dynamixel.move(
                    ids,
                    posiciones_finales,
                    speed=cfg.DEFAULT_SPEED,
                    wait_for_reached=False,  # CRÍTICO: no bloquear el lazo MPC
                )
                t_move_ms = (time.perf_counter() - t_move_start) * 1000.0
              if t_esperando_lock_ms > 200.0 or t_move_ms > 200.0:
                print(f'🐢 [lazo principal] esperó lock={t_esperando_lock_ms:.0f}ms'
                      f' + move()={t_move_ms:.0f}ms (anormal)')

              # Recentrar u_actual_scaled en lo que REALMENTE se mandó (post-umbral),
              # no en la sugerencia cruda del MPC, para que la siguiente búsqueda
              # parta del estado físico real del robot.
              u_actual_scaled = self._escalado_desde_ticks_absolutos(posiciones_finales, home_m)

              # Comando enviado vs. posición real leída en este mismo ciclo (paso 1b),
              # motor por motor, para verificar que el robot sigue las órdenes.
              comando_str = ', '.join(f'm{mid}={p}' for mid, p in zip(ids, posiciones_finales))
              actual_str = ', '.join(f'm{mid}={p}' for mid, p in zip(ids, real_m))
              print(f'🔧 Comando -> [{comando_str}] | Posición motor -> [{actual_str}]')
          else:
            u_actual_scaled = u_opt

        # 6. Registrar Fila en CSV (target, control, observador y OptiTrack)
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

        # 7. Evaluar Llegada al Waypoint (siempre respecto a la fuente de
        # control, cada ciclo -> dist_mm y el dwell necesitan resolución fina).
        dist_mm = err_mm_val

        # Perf: la impresión en consola es cara (I/O de terminal) y no hace
        # falta a 60 Hz -> se muestra solo al ritmo del Action Hold.
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
          # 🧭 Sub-objetivo intermedio: tolerancia laxa, sin dwell -- solo
          # "pasar cerca y seguir" hacia el próximo (o hacia el waypoint
          # real si era el último de la cola). No toca wp_reached_time ni
          # current_wp_idx, eso queda reservado al waypoint real.
          if dist_mm <= TOLERANCIA_SUBOBJETIVO_MM:
            cola_subobjetivos.pop(0)
        else:
          # Banda de histéresis: entrar al dwell exige <= target_tolerance_mm,
          # pero solo se reinicia el conteo si el error se escapa por encima de
          # dwell_reset_tolerance_mm (más ancho). Sin esto, cualquier micro-rebote
          # de ruido (±unos mm, típico del observador LSTM/compliance del cable)
          # apenas por encima de target_tolerance_mm reinicia el conteo entero y
          # el waypoint nunca llega a "alcanzado" aunque esté prácticamente ahí.
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
              # Armar la cola de sub-objetivos para el PRÓXIMO waypoint real,
              # si el salto hacia él es grande (ver DISTANCIA_MAX_SALTO_MM).
              if current_wp_idx < len(waypoints_mm) and ultima_pos_conocida_mm is not None:
                cola_subobjetivos = _generar_subobjetivos(
                    ultima_pos_conocida_mm, waypoints_mm[current_wp_idx]
                )
                if cola_subobjetivos:
                  print(f'🧭 Salto grande al WP {current_wp_idx + 1}: '
                        f'insertando {len(cola_subobjetivos)} sub-objetivo(s) intermedio(s).')
          elif dist_mm > dwell_reset_tolerance_mm:
            # Se escapó de verdad (no ruido) -> recién ahí reiniciar el conteo
            wp_reached_time = None
          # else: target_tolerance_mm < dist_mm <= dwell_reset_tolerance_mm ->
          # se mantiene el conteo en curso (si ya había arrancado) sin sumar ni
          # reiniciar, absorbiendo el ruido de medición dentro de la banda.

        # 8. Sincronización Estricta a 60 Hz (Pacing de Lazo Real)
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
      # Limpieza y cierre seguro del hardware y de los archivos de registro
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
    """Protocolo de validación con repeticiones independientes.

    `fraccion_semilla` (0,1]: qué fracción del camino Home->semilla (en
    espacio de TICKS, no cartesiano) se manda en lazo abierto antes de
    entregarle el control al CEM. Con 1.0 (default) es el bypass completo de
    siempre. Con, por ejemplo, 0.3 o 0.5, se le da al CEM un empujón parcial
    hacia la rama correcta y se deja que complete el resto del camino de
    forma autónoma -- sirve para distinguir si el problema es solo
    "descubrir" la rama desde Home (en ese caso, un empujón parcial alcanza
    para que el CEM converja solo) o si el modelo está mal calibrado en toda
    esa vecindad del espacio de actuadores (en ese caso, ni con el empujón
    parcial converge).

    Cada waypoint se intenta `n_repeticiones` veces, arrancando SIEMPRE
    desde HOME_POSITION (comando directo de motor, sin CEM). Por intento se
    registra cada ciclo (timestamp, wp_idx, intento_idx, target, posición de
    control, posición OptiTrack cruda -- "verdad de terreno" --, error
    instantáneo contra ambas) en memoria, y al cerrar el intento (llegó
    dentro de tolerancia sostenido `hold_time_s`, o se agotó
    `timeout_intento_s` sin llegar) se vuelca todo el bloque al CSV con la
    ÚLTIMA fila marcada 'exito'/'fallo' en la columna `resultado` (el resto
    queda ''). Ningún intento se descarta, incluso si falla.

    A diferencia de run_control_loop (trayectoria continua, pensada para
    operación real visitando varios waypoints en secuencia), acá cada
    intento es una prueba INDEPENDIENTE: se resetea toda la memoria del CEM
    (ver _resetear_estado_mpc_para_nuevo_intento) al arrancar cada uno, para
    que una repetición no "herede" la convergencia de la anterior al mismo
    punto -- si no, un éxito podría deberse a la memoria de std/dirección
    del intento previo, no a que el sistema realmente lo alcance de forma
    confiable desde cero.

    NOTA: a diferencia de run_control_loop, este método no implementa EMA
    (--ema-alpha) ni Setpoint Lock (--setpoint-lock-pct) -- ambos están
    desactivados por defecto (no-op) en la configuración normal, así que no
    hay diferencia de comportamiento si se corre con los parámetros por
    defecto. Si en algún momento se usan esos flags para el experimento,
    hay que agregarlos acá también.
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
    resumen = []  # (wp_idx, intento_idx, resultado, err_final_opti_mm, duracion_s)

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

          # Prueba independiente: sin memoria del intento anterior.
          self._resetear_estado_mpc_para_nuevo_intento()

          filas_intento = []
          # Última predicción VÁLIDA del modelo (t+1 y t+t_out) vista durante
          # este intento -- persiste entre ciclos (a diferencia de
          # pred_tout_mm_row, que se resetea a NaN cada ciclo para el CSV)
          # para poder reportar en el RESUMEN final la predicción del último
          # ciclo de recálculo, útil para comparar en el informe.
          ultimo_pred_t1_mm = [float('nan')] * 3
          ultimo_pred_tout_mm = [float('nan')] * 3

          # 🌱 Comando semilla conocido-bueno (ver SEED_COMMANDS_TICKS): si hay
          # una semilla confirmada para ESTE target y USAR_SEMILLAS_FIJAS está
          # activo, mandar el comando fijo directo (sin CEM) y esperar a que la
          # posición real converja ahí, ANTES de dejar que el CEM tome el
          # control -- igual que en run_control_loop. Se salta el troceo por
          # sub-objetivos (la semilla ya es un salto directo confirmado a mano).
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
          # Valor inicial antes del primer cálculo real (ver misma inicialización
          # en run_control_loop) -- sin esto, el primer ciclo de cada intento
          # revienta con UnboundLocalError si --use-optitrack + action_hold_cycles>1
          # (recién se recalcula en el primer ciclo de Action Hold, no en el 1ro).
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

            # 1. Sensores + estimación del observador (para diagnóstico) +
            # pose OptiTrack cruda (verdad de terreno, None si no hay dato fresco).
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

            # 2. Transformación cinemática relativa + filtrado causal
            meta_ticks_abs = self._ticks_absolutos_desde_escalado(u_actual_scaled, home_m)
            meta_m = meta_ticks_abs.cpu().numpy().tolist()
            dict_features, pos_filt_m = self._procesar_lecturas_cinematicas(
                p_b, q_b, p_e, q_e, real_m, couple_m, tension_m, meta_m, home_m,
                filtrar_posicion=filtrar_pos_ctrl,
            )

            # 3. Normalizar y guardar en búfer
            vec_scaled = self.scaler.scale(dict_features)
            self.buffer.append(vec_scaled)
            x_hist_tensor = torch.stack(list(self.buffer)[-self.t_in:]).unsqueeze(0)

            pos_filt_mm = pos_filt_m * 1000.0
            current_pos_mm = torch.tensor(pos_filt_mm, device=self.device, dtype=torch.float32)
            ultima_pos_conocida_mm = pos_filt_mm.copy()

            # 4-5. CEM (Action Hold) + envío de comando con la misma
            # histéresis anti-deadband y barrera de límites que run_control_loop.
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
              # Predicción del modelo para el paso t+t_out bajo el comando
              # elegido -- para comparar en el print/CSV contra Ctrl/Opti.
              pred_tout_mm_row = y_pred_final_mm[-1].cpu().numpy().tolist()
              # t+1 y t+t_out del último ciclo de recálculo -- persiste hasta
              # el cierre del intento, para el RESUMEN final (ver más abajo).
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

            # 6. Fila del ciclo -- se acumula en memoria; se vuelca completa
            # al terminar el intento (para poder marcar 'resultado' en la
            # última fila sin tener que reescribir el CSV después).
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

            # 7. Evaluar sub-objetivo / llegada (dwell) / timeout.
            # dist_mm (err_mm_val/err_opti_mm_val) es SIEMPRE contra target_wp
            # (el waypoint final) -- correcto para el criterio de éxito/fallo
            # reportado, pero NO sirve para decidir cuándo pasar al siguiente
            # sub-objetivo intermedio (si target_activo != target_wp, el robot
            # puede estar exactamente sobre el sub-objetivo y aun así lejos de
            # target_wp -- la cola nunca avanzaría). Para eso se usa una
            # distancia aparte, contra target_activo, igual que run_control_loop
            # (ahí target_wp SE REASIGNA al sub-objetivo vigente cada ciclo;
            # acá se mantiene fijo en el destino final durante todo el intento,
            # así que hace falta esta distancia separada).
            dist_mm = err_mm_val
            dist_a_target_activo = float(np.linalg.norm(pos_filt_mm - target_activo))

            if es_ciclo_action_hold:
              opti_str = (
                  f'{np.round(pos_opti_mm, 1)} mm' if not np.isnan(pos_opti_mm).any() else 'N/D'
              )
              # Err (ctrl y opti) SIEMPRE se calculan contra target_wp (el
              # waypoint final real) -- por eso el print también tiene que
              # mostrar target_wp como "Target", nunca target_activo. Si
              # target_activo es un sub-objetivo intermedio (target_activo !=
              # target_wp), se muestra aparte y claramente etiquetado, para no
              # confundirlo con el destino contra el que se mide el error
              # (antes el print mostraba target_activo bajo la etiqueta
              # "Target:", dando una distancia visual que no correspondía con
              # el Err calculado -- confuso pero no era un error de cálculo).
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
              # Navegación interna (trocear saltos grandes): distancia al
              # SUB-OBJETIVO vigente, no al destino final -- no es lo que se
              # reporta como resultado del experimento, solo decide cuándo
              # pasar al siguiente punto de la cola.
              if dist_a_target_activo <= TOLERANCIA_SUBOBJETIVO_MM:
                cola_subobjetivos.pop(0)
            else:
              # Criterio de ÉXITO/FALLO real: contra pos_opti (verdad de
              # terreno), no contra pos_ctrl. Si OptiTrack no tiene un frame
              # fresco (err_opti_mm_val = NaN), la comparación numérica da
              # False sola -- ese ciclo no cuenta ni para avanzar el dwell ni
              # para resetearlo, simplemente queda "en pausa" hasta que vuelva
              # a haber dato fresco (no falsea el resultado con dato viejo).
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

            # 8. Pacing a 60 Hz
            t_ejec = time.perf_counter() - t_start
            tiempo_espera = intervalo - t_ejec
            if tiempo_espera > 0:
              time.sleep(tiempo_espera)

          # Cierre del intento: marcar resultado en la ÚLTIMA fila y volcar
          # todo el bloque al logger (recién ahora, ya con 'resultado' fijado).
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

      # Resumen aparte en CSV (una fila por intento, con target + predicciones
      # t+1/t+t_out) -- para tener guardado y poder comparar/graficar en el
      # informe sin tener que reprocesar el log completo de 60Hz.
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
# 7. PUNTO DE ENTRADA
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