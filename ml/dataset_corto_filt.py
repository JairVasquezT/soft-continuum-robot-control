"""Procesa un CSV 'corto' (grabación independiente de 'largo') a un .npy
listo para usarse como set de VALIDACIÓN real del observador (SoftRobotLSTM),
reutilizando los parámetros de escalado (min_t/max_t) del dataset de
entrenamiento -- nunca recalculando límites propios sobre 'corto'.

Guarda {'X': X_scaled, 'Y': Y_scaled} con el mismo formato "plano" (sin
ventanear) que produce dataset_filtre.py -- el ventaneo (crear_secuencias,
WINDOW_SIZE) se hace en el script de entrenamiento, igual que con 'largo'.

Misma fórmula de posición/rotación relativa que MPC.py en inferencia y que
dataset_filtre.py en entrenamiento: p_rel = R_base.inv().apply(p_efector - p_base).

Uso: ajustar PATH_CSV_CORTO / PATH_METADATA_LARGO abajo y correr.
"""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================
PATH_CSV_CORTO = 'corto_20260730_195350_730.csv'
PATH_METADATA_LARGO = 'dataset_v01_completo_sinRot_filt_params.json'
NOMBRE_SALIDA = 'dataset_corto_v01_completo_sinRot_filt'

FS_SISTEMA = 60.0

# =====================================================================
# 2. METADATA DE ENTRENAMIENTO (escalado y calibración a reutilizar)
# =====================================================================
with open(PATH_METADATA_LARGO, 'r') as f:
  metadata = json.load(f)

X_TRANS = metadata['X_transformer']
Y_TRANS = metadata['Y_transformer']
ANGULOS_CALIBRACION = metadata['home_motores']

# El orden de las columnas de entrenamiento queda fijado por el orden de
# claves del JSON (Python/json preservan el orden de inserción).
x_keys = list(X_TRANS.keys())
y_keys = list(Y_TRANS.keys())

# =====================================================================
# 3. CARGA Y LECTURA DEL CSV DE CORTO
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
# 4. GEOMETRÍA: DELTAS DE MOTOR + POSICIÓN/ROTACIÓN RELATIVA AL MARCO DE LA BASE
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
# 5. FILTRADO PASABAJAS (mismos cutoffs que dataset_filtre.py)
# =====================================================================
def aplicar_filtro_pasabajas(data, cutoff_hz, fs_hz=FS_SISTEMA, order=2):
  nyquist = 0.5 * fs_hz
  b, a = butter(order, cutoff_hz / nyquist, btype='low', analog=False)
  return filtfilt(b, a, data, axis=0)


for col in ['rel_x', 'rel_y', 'rel_z']:
  df[col] = aplicar_filtro_pasabajas(df[col].values, cutoff_hz=6.0)

if any(k.startswith('rel_q') for k in y_keys):
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
# 6. ESCALADO CON LOS PARÁMETROS DE ENTRENAMIENTO (NO recalcular límites)
# =====================================================================
def escalar_con_parametros(df_raw, columnas_nombres, dict_params):
  scaled = np.zeros((len(df_raw), len(columnas_nombres)), dtype=np.float32)
  for idx, col in enumerate(columnas_nombres):
    min_t = dict_params[col]['min_t']
    max_t = dict_params[col]['max_t']
    val = df_raw[col].values
    scaled[:, idx] = -1 + 2 * (val - min_t) / (max_t - min_t)
  return scaled


X_scaled = escalar_con_parametros(df, x_keys, X_TRANS)
Y_scaled = escalar_con_parametros(df, y_keys, Y_TRANS)

dataset = {'X': X_scaled, 'Y': Y_scaled}
np.save(f'{NOMBRE_SALIDA}.npy', dataset)

print(f'✓ Generado: {NOMBRE_SALIDA}.npy | X {X_scaled.shape} | Y {Y_scaled.shape}')
