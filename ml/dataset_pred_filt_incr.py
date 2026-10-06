import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

# =====================================================================
# 1. PARAMETERS OF THE DIRECT PREDICTIVE MODEL FOR MPC
# =====================================================================
T_IN = 45  # Past history steps (0.75 s at 60 Hz)
T_OUT = 10 # Future prediction steps (0.25 s at 60 Hz)
MAX_DELTA_T = 1.0 / 55.0  # Maximum allowed time (55 Hz = ~18.18 ms)
FS_SISTEMA = 60.0  # Sampling frequency (60 Hz)

PATH_CSV = 'largo_20260730_191948_808.csv'

# =====================================================================
# 2. LOADING AND READING THE CSV
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

df = pd.read_csv(PATH_CSV, names=columnas)

# Computation of the time delta between samples
df['delta_t'] = df['t_relativo'].diff().bfill()
df = df.iloc[1:].reset_index(drop=True)

print(f'✓ Muestras cargadas correctamente ({len(df)} filas)')

# =====================================================================
# 3. CALIBRATION AND MECHANICAL-GEOMETRIC TRANSFORMATION
# =====================================================================
ANGULOS_CALIBRACION = {'m1': 1871.0, 'm2': 1951.0, 'm3': 1485.0, 'm4': 1712.0}
RANGOS_MANUALES = {'m1': 650.0, 'm2': 650.0, 'm3': 750.0, 'm4': 750.0}

LIMITES_ESPACIALES = {
    'rel_x': (-0.220, 0.220),
    'rel_y': (0.180, 0.330),
    'rel_z': (-0.220, 0.220),
}

LIMITES_TORQUE = (-500.0, 500.0)
LIMITES_TENSION = (500.0, 2000.0)

# Relative motor positions
for i in range(1, 5):
  df[f'delta_real_m{i}'] = df[f'real_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']
  df[f'delta_meta_m{i}'] = df[f'meta_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']

# Position and rotation relative to the BASE frame (same computation as MPC.py
# at inference: p_rel = R_base.inv().apply(p_efector - p_base)). Previously this
# was a raw subtraction/quaternion (world frame), without rotating.
p_base_arr = df[['base_x', 'base_y', 'base_z']].values
p_efector_arr = df[['efector_x', 'efector_y', 'efector_z']].values
q_base_arr = df[['base_qx', 'base_qy', 'base_qz', 'base_qw']].values
q_efector_arr = df[['efector_qx', 'efector_qy', 'efector_qz', 'efector_qw']].values

r_b = R.from_quat(q_base_arr)
r_e = R.from_quat(q_efector_arr)

p_rel = r_b.inv().apply(p_efector_arr - p_base_arr)
df['rel_x'] = p_rel[:, 0]
df['rel_y'] = p_rel[:, 1]
df['rel_z'] = p_rel[:, 2]

q_rel = (r_b.inv() * r_e).as_quat()
df['rel_qx'] = q_rel[:, 0]
df['rel_qy'] = q_rel[:, 1]
df['rel_qz'] = q_rel[:, 2]
df['rel_qw'] = q_rel[:, 3]


# =====================================================================
# 4. ZERO-PHASE DIGITAL FILTERING (LOW-PASS + UNWRAP)
# =====================================================================
def aplicar_filtro_pasabajas(data, cutoff_hz=4.0, fs_hz=60.0, order=2):
  nyquist = 0.5 * fs_hz
  normal_cutoff = cutoff_hz / nyquist
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return filtfilt(b, a, data, axis=0)


def filtrar_y_normalizar_cuaterniones(
    df_quat, cutoff_hz=6.0, fs_hz=60.0, order=2
):
  q_vals = df_quat[['rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']].values.copy()
  for i in range(1, len(q_vals)):
    if np.dot(q_vals[i], q_vals[i - 1]) < 0:
      q_vals[i] = -q_vals[i]

  q_filt = aplicar_filtro_pasabajas(
      q_vals, cutoff_hz=cutoff_hz, fs_hz=fs_hz, order=order
  )
  normas = np.linalg.norm(q_filt, axis=1, keepdims=True)
  normas[normas == 0] = 1.0
  return q_filt / normas


print('🧹 Aplicando filtrado pasabajas selectivo...')

for col in ['rel_x', 'rel_y', 'rel_z']:
  df[col] = aplicar_filtro_pasabajas(
      df[col].values, cutoff_hz=6.0, fs_hz=FS_SISTEMA
  )

q_norm = filtrar_y_normalizar_cuaterniones(
    df[['rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']],
    cutoff_hz=6.0,
    fs_hz=FS_SISTEMA,
)
df['rel_qx'], df['rel_qy'], df['rel_qz'], df['rel_qw'] = (
    q_norm[:, 0],
    q_norm[:, 1],
    q_norm[:, 2],
    q_norm[:, 3],
)

for i in range(1, 5):
  df[f'couple_m{i}'] = aplicar_filtro_pasabajas(
      df[f'couple_m{i}'].values, cutoff_hz=3.5, fs_hz=FS_SISTEMA
  )
  df[f'tension_m{i}'] = aplicar_filtro_pasabajas(
      df[f'tension_m{i}'].values, cutoff_hz=3.5, fs_hz=FS_SISTEMA
  )

print('✓ Filtrado completado exitosamente.')


# =====================================================================
# 5. AUTOMATIC NORMALIZATION TO THE RANGE [-1, 1]
# =====================================================================
def calcular_limites_y_escalar(data_frame, nombres_columnas):
  data_array = data_frame[nombres_columnas].values
  scaled_data = np.zeros_like(data_array, dtype=float)
  dict_parametros = {}

  for col in range(data_array.shape[1]):
    col_name = nombres_columnas[col]

    if 'delta_real' in col_name or 'delta_meta' in col_name:
      m_id = col_name.split('_')[-1]
      radio = RANGOS_MANUALES[m_id]
      min_t, max_t = -radio, radio

    elif col_name.startswith('couple_'):
      min_t, max_t = LIMITES_TORQUE

    elif col_name.startswith('tension_'):
      min_t, max_t = LIMITES_TENSION

    elif col_name in LIMITES_ESPACIALES:
      min_t, max_t = LIMITES_ESPACIALES[col_name]

    elif col_name.startswith('rel_q'):
      min_t, max_t = -1.0, 1.0

    elif col_name == 'delta_t':
      min_t, max_t = 0.016, 0.033

    else:
      col_min = float(np.min(data_array[:, col]))
      col_max = float(np.max(data_array[:, col]))
      rango_actual = col_max - col_min

      if rango_actual == 0:
        scaled_data[:, col] = data_array[:, col]
        dict_parametros[col_name] = {'min_t': col_min, 'max_t': col_max}
        continue

      min_t = col_min - (0.1 * rango_actual / 0.8)
      max_t = col_max + (0.1 * rango_actual / 0.8)

    scaled_val = -1 + 2 * (data_array[:, col] - min_t) / (max_t - min_t)
    scaled_data[:, col] = np.clip(scaled_val, -1.2, 1.2)
    dict_parametros[col_name] = {'min_t': float(min_t), 'max_t': float(max_t)}

  return scaled_data, dict_parametros


# =====================================================================
# 6. STRUCTURING OF DIRECT SEQUENCES WITH STRICT FILTERING
# =====================================================================
def crear_secuencias_directas_mpc(
    df_raw, columnas_x_hist, columnas_x_fut, columnas_y, t_in=45, t_out=10
):
  """Generates sequences for direct prediction.

  Filters windows only if: 1. The frequency drops below 55 Hz in the window
  (delta_t > 18.18 ms). 2. The target setpoint (U) changes during the 10 future
  steps.
  """
  X_scaled, params_x_hist = calcular_limites_y_escalar(df_raw, columnas_x_hist)
  X_fut_scaled, params_x_fut = calcular_limites_y_escalar(
      df_raw, columnas_x_fut
  )
  Y_scaled, params_y = calcular_limites_y_escalar(df_raw, columnas_y)

  X_hist_list, U_cand_list, Y_fut_list, Y_t0_list = [], [], [], []

  descartadas_frec = 0
  descartadas_consigna = 0
  aceptadas = 0

  n_muestras = len(df_raw)

  # Continuous sweep over the entire time series
  for i in range(n_muestras - t_in - t_out):

    # 1. Frequency filter (< 55 Hz over the full t_in + t_out window)
    window_delta_t = df_raw['delta_t'].iloc[i : i + t_in + t_out].values
    if np.any(window_delta_t > MAX_DELTA_T):
      descartadas_frec += 1
      continue

    # 2. Constant-Setpoint Filter over the 10 FUTURE STEPS (steps t_in to t_in + t_out)
    u_futuros = X_fut_scaled[i + t_in : i + t_in + t_out]  # Shape: (10, 4)

    # We check whether the command changes within the future horizon
    if not np.all(u_futuros == u_futuros[0]):
      descartadas_consigna += 1
      continue  # Only the future command transition is discarded

    # Saving the valid sample
    x_hist = X_scaled[i : i + t_in]  # 45 past steps (may include previous commands)
    u_cand = X_fut_scaled[i + t_in]  # Candidate target command
    y_fut = Y_scaled[i + t_in : i + t_in + t_out]  # 10 future state steps
    # State (position/orientation) scaled at t0 -- the last step of the
    # history, i.e. the row right BEFORE the first step of y_fut.
    # It is stored on the SAME scale as y_fut (same params_y) so that
    # incremental displacements (y_fut - y_t0) can be built without
    # rescaling anything -- see train_pred_incr.py.
    y_t0 = Y_scaled[i + t_in - 1]

    X_hist_list.append(x_hist)
    U_cand_list.append(u_cand)
    Y_fut_list.append(y_fut)
    Y_t0_list.append(y_t0)
    aceptadas += 1

  print(f'📊 Filtro aplicado:')
  print(f'   ✅ Muestras aceptadas: {aceptadas}')
  print(
      f'   ❌ Descartadas por cambio de consigna futuro ({t_out}p):'
      f' {descartadas_consigna}'
  )
  print(f'   ❌ Descartadas por baja frecuencia (< 55 Hz): {descartadas_frec}')

  X_hist_arr = np.array(X_hist_list, dtype=np.float32)
  U_cand_arr = np.array(U_cand_list, dtype=np.float32)
  Y_fut_arr = np.array(Y_fut_list, dtype=np.float32)
  Y_t0_arr = np.array(Y_t0_list, dtype=np.float32)

  params_x_combined = {**params_x_hist, **params_x_fut}

  return (
      X_hist_arr,
      U_cand_arr,
      Y_fut_arr,
      Y_t0_arr,
      params_x_combined,
      params_y,
  )


def procesar_y_guardar_directo(
    cols_x_hist, cols_x_fut, cols_y, nombre_archivo_base
):
  nombre_archivo = f'{nombre_archivo_base}_directo_filt'

  X_hist, U_cand, Y_fut, Y_t0, params_x, params_y = crear_secuencias_directas_mpc(
      df,
      cols_x_hist,
      cols_x_fut,
      cols_y,
      t_in=T_IN,
      t_out=T_OUT,
  )

  # Save a dictionary with the optimized main tensors. Y_t0 is
  # new: the scaled position/orientation at t0 (last step of the
  # history) per sample, to be able to train on incremental
  # displacements relative to t0 (see train_pred_incr.py) without breaking
  # compatibility with the existing scripts (which only read X_hist/U_cand/
  # Y_fut and ignore extra keys).
  dataset = {'X_hist': X_hist, 'U_cand': U_cand, 'Y_fut': Y_fut, 'Y_t0': Y_t0}
  np.save(f'{nombre_archivo}.npy', dataset)

  metadata = {
      'X_transformer': params_x,
      'Y_transformer': params_y,
      'home_motores': ANGULOS_CALIBRACION,
      't_in': T_IN,
      't_out': T_OUT,
      'shapes': {
          'X_hist': list(X_hist.shape),
          'U_cand': list(U_cand.shape),
          'Y_fut': list(Y_fut.shape),
          'Y_t0': list(Y_t0.shape),
      },
  }
  with open(f'{nombre_archivo}_params.json', 'w') as f:
    json.dump(metadata, f, indent=4)

  print(
      f'✓ Creado {nombre_archivo}.npy | Historial: {X_hist.shape} | Candidato'
      f' U: {U_cand.shape} | Y Futuro: {Y_fut.shape} | Y t0: {Y_t0.shape}\n'
  )


# =====================================================================
# 7. DEFINITION OF REQUIRED CONFIGURATIONS
# =====================================================================
cols_x_futuro = [
    'delta_meta_m1',
    'delta_meta_m2',
    'delta_meta_m3',
    'delta_meta_m4',
]

cols_base = [
    'delta_real_m1',
    'delta_real_m2',
    'delta_real_m3',
    'delta_real_m4',
    'delta_meta_m1',
    'delta_meta_m2',
    'delta_meta_m3',
    'delta_meta_m4',
    'delta_t',
]
cols_torque = ['couple_m1', 'couple_m2', 'couple_m3', 'couple_m4']
cols_tension = ['tension_m1', 'tension_m2', 'tension_m3', 'tension_m4']

configuraciones_X_hist = {
    'real_meta_10': cols_base,  # 9 inputs
    'real_meta_torque_10': cols_base + cols_torque,  # 13 inputs
    'real_meta_tension_10': cols_base + cols_tension,  # 13 inputs
    'completo_10': cols_base + cols_torque + cols_tension,  # 17 inputs
}

cols_y_sinRot = ['rel_x', 'rel_y', 'rel_z']
cols_y_conRot = [
    'rel_x',
    'rel_y',
    'rel_z',
    'rel_qx',
    'rel_qy',
    'rel_qz',
    'rel_qw',
]

configuraciones_Y = {'sinRot': cols_y_sinRot, 'conRot': cols_y_conRot}

# =====================================================================
# 8. EXECUTION AND GENERATION OF DATASETS
# =====================================================================
print(
    f'\n🚀 Generando datasets para MPC Predictivo Directo (T_IN={T_IN},'
    f' T_OUT={T_OUT})...\n'
)

v_idx = 1
for x_tag, x_hist_cols in configuraciones_X_hist.items():
  for y_tag, y_cols in configuraciones_Y.items():
    nombre_base = f'dataset_pred_v{v_idx:02d}_{x_tag}_{y_tag}'
    procesar_y_guardar_directo(
        x_hist_cols, cols_x_futuro, y_cols, nombre_base
    )
    v_idx += 1

print('🎉 Proceso completado exitosamente.')