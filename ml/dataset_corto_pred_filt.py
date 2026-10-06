"""Processes a 'corto' CSV (recording independent from 'largo') into a .npy
ready to be used as a real VALIDATION set for the predictor (MPCDirectPredictor),
reusing the scaling parameters (min_t/max_t) of the training dataset
-- never recomputing its own limits on 'corto', so as not to
leak information from the test set into training.

Same relative position/rotation formula as MPC.py at inference and as
dataset_pred_filt.py at training: p_rel = R_base.inv().apply(p_efector - p_base).

Usage: adjust PATH_CSV_CORTO / PATH_METADATA_LARGO below and run.
"""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
PATH_CSV_CORTO = 'corto_20260730_195350_730.csv'
PATH_METADATA_LARGO = 'dataset_pred_v08_completo_conRot_directo_filt_params.json'
NOMBRE_SALIDA = 'dataset_corto_v08_completo_conRot_directo_filt'

FS_SISTEMA = 60.0
MAX_DELTA_T = 1.0 / 55.0  # 55 Hz = ~18.18 ms

# =====================================================================
# 2. TRAINING METADATA (scaling and calibration to be reused)
# =====================================================================
with open(PATH_METADATA_LARGO, 'r') as f:
  metadata = json.load(f)

X_TRANS = metadata['X_transformer']
Y_TRANS = metadata['Y_transformer']
ANGULOS_CALIBRACION = metadata['home_motores']
T_IN = metadata['t_in']
T_OUT = metadata['t_out']

# =====================================================================
# 3. LOADING AND READING THE 'CORTO' CSV
# =====================================================================
columnas = [
    'timestamp',
    't_unix',
    't_relativo',
    'combo_id',
    'valid_lecture',
    'meta_m1',
    'meta_m2',
    'meta_m3',
    'meta_m4',
    'real_m1',
    'real_m2',
    'real_m3',
    'real_m4',
    'couple_m1',
    'couple_m2',
    'couple_m3',
    'couple_m4',
    'tension_m1',
    'tension_m2',
    'tension_m3',
    'tension_m4',
    'base_x',
    'base_y',
    'base_z',
    'base_qx',
    'base_qy',
    'base_qz',
    'base_qw',
    'efector_x',
    'efector_y',
    'efector_z',
    'efector_qx',
    'efector_qy',
    'efector_qz',
    'efector_qw',
]

df = pd.read_csv(PATH_CSV_CORTO, names=columnas)
df['delta_t'] = df['t_relativo'].diff().bfill()
df = df.iloc[1:].reset_index(drop=True)
print(f'✓ Muestras de corto cargadas ({len(df)} filas)')

# =====================================================================
# 4. GEOMETRY: MOTOR DELTAS + POSITION/ROTATION RELATIVE TO THE BASE FRAME
# =====================================================================
for i in range(1, 5):
  df[f'delta_real_m{i}'] = df[f'real_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']
  df[f'delta_meta_m{i}'] = df[f'meta_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']

p_base_arr = df[['base_x', 'base_y', 'base_z']].values
p_efector_arr = df[['efector_x', 'efector_y', 'efector_z']].values
q_base_arr = df[['base_qx', 'base_qy', 'base_qz', 'base_qw']].values
q_efector_arr = df[['efector_qx', 'efector_qy', 'efector_qz', 'efector_qw']].values

r_b = R.from_quat(q_base_arr)
r_e = R.from_quat(q_efector_arr)

p_rel = r_b.inv().apply(p_efector_arr - p_base_arr)
df['rel_x'], df['rel_y'], df['rel_z'] = p_rel[:, 0], p_rel[:, 1], p_rel[:, 2]

q_rel = (r_b.inv() * r_e).as_quat()
df['rel_qx'], df['rel_qy'], df['rel_qz'], df['rel_qw'] = (
    q_rel[:, 0],
    q_rel[:, 1],
    q_rel[:, 2],
    q_rel[:, 3],
)

# =====================================================================
# 5. LOW-PASS FILTERING (same cutoffs as dataset_pred_filt.py)
# =====================================================================
def aplicar_filtro_pasabajas(data, cutoff_hz, fs_hz=FS_SISTEMA, order=2):
  nyquist = 0.5 * fs_hz
  b, a = butter(order, cutoff_hz / nyquist, btype='low', analog=False)
  return filtfilt(b, a, data, axis=0)


for col in ['rel_x', 'rel_y', 'rel_z']:
  df[col] = aplicar_filtro_pasabajas(df[col].values, cutoff_hz=6.0)

q_vals = df[['rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']].values.copy()
for i in range(1, len(q_vals)):
  if np.dot(q_vals[i], q_vals[i - 1]) < 0:
    q_vals[i] = -q_vals[i]
q_filt = aplicar_filtro_pasabajas(q_vals, cutoff_hz=6.0)
normas = np.linalg.norm(q_filt, axis=1, keepdims=True)
normas[normas == 0] = 1.0
q_norm = q_filt / normas
df['rel_qx'], df['rel_qy'], df['rel_qz'], df['rel_qw'] = (
    q_norm[:, 0],
    q_norm[:, 1],
    q_norm[:, 2],
    q_norm[:, 3],
)

for i in range(1, 5):
  df[f'couple_m{i}'] = aplicar_filtro_pasabajas(df[f'couple_m{i}'].values, cutoff_hz=3.5)
  df[f'tension_m{i}'] = aplicar_filtro_pasabajas(df[f'tension_m{i}'].values, cutoff_hz=3.5)

print('✓ Filtrado completado.')

# =====================================================================
# 6. SCALING WITH THE TRAINING PARAMETERS (do NOT recompute limits)
# =====================================================================
def escalar_con_parametros(df_raw, columnas_nombres, dict_params):
  scaled = np.zeros((len(df_raw), len(columnas_nombres)))
  for idx, col in enumerate(columnas_nombres):
    min_t = dict_params[col]['min_t']
    max_t = dict_params[col]['max_t']
    val = df_raw[col].values
    scaled[:, idx] = np.clip(-1 + 2 * (val - min_t) / (max_t - min_t), -1.2, 1.2)
  return scaled


cols_x_hist = [
    'delta_real_m1',
    'delta_real_m2',
    'delta_real_m3',
    'delta_real_m4',
    'delta_meta_m1',
    'delta_meta_m2',
    'delta_meta_m3',
    'delta_meta_m4',
    'delta_t',
    'couple_m1',
    'couple_m2',
    'couple_m3',
    'couple_m4',
    'tension_m1',
    'tension_m2',
    'tension_m3',
    'tension_m4',
]
cols_x_fut = ['delta_meta_m1', 'delta_meta_m2', 'delta_meta_m3', 'delta_meta_m4']
cols_y = ['rel_x', 'rel_y', 'rel_z', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']

X_scaled = escalar_con_parametros(df, cols_x_hist, X_TRANS)
X_fut_scaled = escalar_con_parametros(df, cols_x_fut, X_TRANS)
Y_scaled = escalar_con_parametros(df, cols_y, Y_TRANS)

# =====================================================================
# 7. SLIDING WINDOWS (same frequency/constant-setpoint filters
#    as crear_secuencias_directas_mpc in dataset_pred_filt.py)
# =====================================================================
X_hist_list, U_cand_list, Y_fut_list = [], [], []
descartadas_frec = 0
descartadas_consigna = 0
aceptadas = 0

for i in range(len(df) - T_IN - T_OUT):
  window_delta_t = df['delta_t'].iloc[i : i + T_IN + T_OUT].values
  if np.any(window_delta_t > MAX_DELTA_T):
    descartadas_frec += 1
    continue

  u_futuros = X_fut_scaled[i + T_IN : i + T_IN + T_OUT]
  if not np.all(u_futuros == u_futuros[0]):
    descartadas_consigna += 1
    continue

  X_hist_list.append(X_scaled[i : i + T_IN])
  U_cand_list.append(X_fut_scaled[i + T_IN])
  Y_fut_list.append(Y_scaled[i + T_IN : i + T_IN + T_OUT])
  aceptadas += 1

print('📊 Filtro aplicado:')
print(f'   ✅ Muestras aceptadas: {aceptadas}')
print(f'   ❌ Descartadas por cambio de consigna futuro: {descartadas_consigna}')
print(f'   ❌ Descartadas por baja frecuencia (< 55 Hz): {descartadas_frec}')

X_hist = np.array(X_hist_list, dtype=np.float32)
U_cand = np.array(U_cand_list, dtype=np.float32)
Y_fut = np.array(Y_fut_list, dtype=np.float32)

dataset = {'X_hist': X_hist, 'U_cand': U_cand, 'Y_fut': Y_fut}
np.save(f'{NOMBRE_SALIDA}.npy', dataset)

print(
    f'✓ Generado: {NOMBRE_SALIDA}.npy | Historial: {X_hist.shape} | Candidato'
    f' U: {U_cand.shape} | Y Futuro: {Y_fut.shape}'
)
