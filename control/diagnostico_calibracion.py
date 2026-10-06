"""Quick diagnostic: for HOME + several motor combos (one per motor,

to isolate which cable/motor is suspect), compares the real position measured
by OptiTrack (relative to the base) against what the LSTM observer predicts.
Useful to detect whether a bent cable makes the model predict poorly in
certain configurations.

Adjust HOME/OFFSET/PUNTOS/SETTLE_S below according to what you want to test.

Usage: python continuum_robot/control/diagnostico_calibracion.py
"""
import os
import sys
import json
import time
from collections import deque

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

from continuum_robot.config import robot_config as cfg
from continuum_robot.hardware.dynamixel import create_controller
from NatNetClient import NatNetClient

try:
    from continuum_robot.hardware.galga import PhidgetForceController
except Exception:
    PhidgetForceController = None

# ==============================================================================
# Editable configuration
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OBSERVER_PATH = os.path.join(CURRENT_DIR, 'soft_robot_lstm_v8_12_time.pth')
METADATA_OBS_PATH = os.path.join(CURRENT_DIR, 'dataset_v08_completo_sinRot_filt_params_60.json')

HOME = cfg.HOME_POSITION  # {1: 1871, 2: 1951, 3: 1485, 4: 1712}
MOTOR_IDS = cfg.MOTOR_IDS
OFFSET = 300      # ticks of displacement to isolate each motor
SETTLE_S = 3.0    # seconds to wait after each movement before measuring
MUESTREO_HZ = 50.0
T_IN = 45         # history steps the LSTM expects (same as in MPC.py)

# HOME + one combo per motor (the others stay at HOME) -- this way, if the error
# spikes right when ONE particular motor moves, it is a good sign
# that THAT motor's cable is the problem.
PUNTOS = [
    ("HOME", dict(HOME)),
    ("m1 +", {**HOME, 1: HOME[1] + OFFSET}),
    ("m1 -", {**HOME, 1: HOME[1] - OFFSET}),
    ("m2 +", {**HOME, 2: HOME[2] + OFFSET}),
    ("m3 +", {**HOME, 3: HOME[3] + OFFSET}),
    ("m4 +", {**HOME, 4: HOME[4] + OFFSET}),
]


# ==============================================================================
# LSTM observer (same architecture as MPC.py)
# ==============================================================================
class SoftRobotLSTM(nn.Module):
    def __init__(self, input_size=17, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(out[:, -1, :]))


def cargar_observador(path, device):
    try:
        return torch.jit.load(path, map_location=device)
    except Exception:
        pass
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model = SoftRobotLSTM(
        input_size=checkpoint.get('input_size', 17),
        hidden_size=checkpoint.get('hidden_size', 128),
        num_layers=checkpoint.get('num_layers', 2),
        output_size=checkpoint.get('output_size', 3),
        dropout=checkpoint.get('dropout', 0.2),
    )
    model.load_state_dict(state_dict)
    return model.to(device)


# ==============================================================================
# Shared OptiTrack state (updated asynchronously by NatNet)
# ==============================================================================
p_base, q_base = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
p_efector, q_efector = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]


def on_frame(new_id, position, rotation):
    global p_base, q_base, p_efector, q_efector
    if new_id == 1:
        p_base, q_base = position, rotation
    elif new_id == 2:
        p_efector, q_efector = position, rotation


def posicion_relativa_mm():
    """Same formula as MPC.py: (p_effector - p_base) rotated into the base frame."""
    r_b = R.from_quat(q_base)
    p_rel = r_b.inv().apply(np.array(p_efector) - np.array(p_base))
    return p_rel * 1000.0


def main():
    print(f'Ejecutando en: {DEVICE}')

    with open(METADATA_OBS_PATH, 'r') as f:
        meta_obs = json.load(f)
    x_trans = meta_obs['X_transformer']
    feature_names = list(x_trans.keys())
    mins = torch.tensor([x_trans[k]['min_t'] for k in feature_names], device=DEVICE, dtype=torch.float32)
    maxs = torch.tensor([x_trans[k]['max_t'] for k in feature_names], device=DEVICE, dtype=torch.float32)

    y_trans = meta_obs.get('Y_transformer')
    if y_trans:
        min_y = torch.tensor([y_trans['rel_x']['min_t'], y_trans['rel_y']['min_t'], y_trans['rel_z']['min_t']], device=DEVICE)
        max_y = torch.tensor([y_trans['rel_x']['max_t'], y_trans['rel_y']['max_t'], y_trans['rel_z']['max_t']], device=DEVICE)
    else:
        min_y = max_y = None

    print(f'Cargando observador LSTM desde: {OBSERVER_PATH}')
    observer = cargar_observador(OBSERVER_PATH, DEVICE)
    observer.eval()

    # Causal torque filter (3.5 Hz), same as in MPC.py
    nyquist = 0.5 * MUESTREO_HZ
    b, a = butter(2, 3.5 / nyquist, btype='low', analog=False)
    zi_torque = np.tile(lfilter_zi(b, a), (4, 1)).T

    def filtrar_torque(x):
        x = np.asarray(x, dtype=np.float64).flatten()
        y = np.zeros(4)
        for i in range(4):
            out, zf = lfilter(b, a, [x[i]], zi=zi_torque[:, i])
            y[i] = out[0]
            zi_torque[:, i] = zf
        return y

    print('Conectando Dynamixel...')
    motor = create_controller(simulate=False, port=cfg.SERIAL_PORT, baudrate=cfg.BAUDRATE)
    found = motor.scan(MOTOR_IDS) if hasattr(motor, 'scan') else MOTOR_IDS
    if hasattr(motor, 'enable_torque'):
        motor.enable_torque(found)

    phidget = None
    if PhidgetForceController is not None:
        try:
            phidget = PhidgetForceController()
            print('✓ PhidgetBridge (4 celdas) inicializado correctamente.')
        except Exception as e:
            print(f'⚠️ No se pudo inicializar Phidget: {e} -- tension_m quedará en 0.0 '
                  f'(el modelo entrenó con ~500-2000, esto va a inflar el error).')
    else:
        print('⚠️ Módulo Phidget no disponible -- tension_m quedará en 0.0 '
              '(el modelo entrenó con ~500-2000, esto va a inflar el error).')

    print('Conectando OptiTrack...')
    client = NatNetClient()
    client.set_client_address(cfg.OPTITRACK_HOST)
    client.set_server_address(cfg.OPTITRACK_HOST)
    client.rigid_body_listener = on_frame
    client.run()
    time.sleep(1.0)  # give time for the first frames to arrive

    resultados = []

    try:
        for nombre, combo in PUNTOS:
            objetivo = [combo[mid] for mid in MOTOR_IDS]
            print(f'\n=== {nombre}: moviendo a {dict(zip(MOTOR_IDS, objetivo))} ===')
            motor.move(MOTOR_IDS, objetivo, speed=cfg.DEFAULT_SPEED, wait_for_reached=True)
            print(f'Asentando {SETTLE_S:.1f}s...')
            time.sleep(SETTLE_S)

            # Refresh the LSTM history with real readings from THIS point
            buffer = deque(maxlen=T_IN)
            intervalo = 1.0 / MUESTREO_HZ
            for _ in range(T_IN):
                t0 = time.perf_counter()
                posiciones, torques, _ = motor.sync_get_present_position_and_load(MOTOR_IDS)
                torques_filt = filtrar_torque(torques)
                if phidget is not None:
                    try:
                        tensiones = phidget.leer_fuerzas_gramos()
                    except Exception:
                        tensiones = [0.0] * 4
                else:
                    tensiones = [0.0] * 4
                fila = {'delta_t': intervalo}
                for i in range(1, 5):
                    fila[f'delta_real_m{i}'] = float(posiciones[i - 1] - HOME[i])
                    fila[f'delta_meta_m{i}'] = float(objetivo[i - 1] - HOME[i])
                    fila[f'couple_m{i}'] = torques_filt[i - 1]
                    fila[f'tension_m{i}'] = float(tensiones[i - 1])
                vec = torch.tensor([fila[k] for k in feature_names], device=DEVICE, dtype=torch.float32)
                vec_scaled = -1.0 + 2.0 * (vec - mins) / (maxs - mins + 1e-8)
                buffer.append(vec_scaled)
                dt = time.perf_counter() - t0
                if dt < intervalo:
                    time.sleep(intervalo - dt)

            opti_mm = posicion_relativa_mm()
            with torch.no_grad():
                x_hist = torch.stack(list(buffer)).unsqueeze(0)
                pred = observer(x_hist).squeeze(0)
                if min_y is not None:
                    obs_mm = (min_y + (pred + 1.0) * (max_y - min_y) / 2.0).cpu().numpy() * 1000.0
                else:
                    obs_mm = pred.cpu().numpy() * 1000.0

            error_mm = float(np.linalg.norm(opti_mm - obs_mm))
            print(f'Opti (real):        {np.round(opti_mm, 1)} mm')
            print(f'Obs  (LSTM estima): {np.round(obs_mm, 1)} mm')
            print(f'Error: {error_mm:.2f} mm')
            resultados.append((nombre, opti_mm.copy(), obs_mm.copy(), error_mm))

    finally:
        print('\n=== Resumen ===')
        print(f'{"Punto":8s} {"Opti (mm)":24s} {"Obs (mm)":24s} {"Error (mm)":10s}')
        for nombre, opti_mm, obs_mm, error_mm in resultados:
            print(f'{nombre:8s} {str(np.round(opti_mm, 1)):24s} {str(np.round(obs_mm, 1)):24s} {error_mm:10.2f}')

        try:
            if hasattr(motor, 'disable_torque'):
                motor.disable_torque(MOTOR_IDS)
        except Exception:
            pass
        try:
            motor.close()
        except Exception:
            pass
        try:
            client.shutdown()
        except Exception:
            pass
        try:
            if phidget is not None:
                phidget.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
