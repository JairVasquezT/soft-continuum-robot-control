import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# =====================================================================
# 1. LOADING AND READING THE CSV
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

df = pd.read_csv('largo_20260730_165024_062.csv', names=columnas)

# Computation of the time delta between samples
df['delta_t'] = df['t_relativo'].diff()
df['delta_t'] = df['delta_t'].bfill()
df = df.iloc[1:].reset_index(drop=True)

print(f'✓ Muestras cargadas correctamente ({len(df)} filas)')

# =====================================================================
# 2. CALIBRATION SECTION (FIXED CENTERS AND RANGES)
# =====================================================================
ANGULOS_CALIBRACION = {'m1': 1871.0, 'm2': 1951.0, 'm3': 1485.0, 'm4': 1712.0}

RANGOS_MANUALES = {'m1': 650.0, 'm2': 650.0, 'm3': 750.0, 'm4': 750.0}

LIMITES_ESPACIALES = {
    'rel_x': (-0.220, 0.220),
    'rel_y': (0.180, 0.330),
    'rel_z': (-0.220, 0.220),
}

LIMITES_TORQUE = (-400.0, 400.0)
LIMITES_TENSION = (500.0, 2000.0)

# Apply motor deltas relative to the HOME position
for i in range(1, 5):
  df[f'delta_real_m{i}'] = df[f'real_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']
  df[f'delta_meta_m{i}'] = df[f'meta_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']

# =====================================================================
# 3. RELATIVE POSITION TRANSFORMATION
# =====================================================================
df['rel_x'] = df['efector_x'] - df['base_x']
df['rel_y'] = df['efector_y'] - df['base_y']
df['rel_z'] = df['efector_z'] - df['base_z']

# =====================================================================
# 4. RELATIVE ROTATION TRANSFORMATION (QUATERNIONS)
# =====================================================================
wb, xb, yb, zb = (
    df['base_qw'],
    -df['base_qx'],
    -df['base_qy'],
    -df['base_qz'],
)
we, xe, ye, ze = (
    df['efector_qw'],
    df['efector_qx'],
    df['efector_qy'],
    df['efector_qz'],
)
df['rel_qw'] = wb * we - xb * xe - yb * ye - zb * ze
df['rel_qx'] = wb * xe + xb * we + yb * ze - zb * ye
df['rel_qy'] = wb * ye - xb * ze + yb * we + zb * xe
df['rel_qz'] = wb * ze + xb * ye - yb * xe + zb * we


# =====================================================================
# 4.5. ZERO-PHASE LOW-PASS FILTERING (NEW CLEANING SECTION)
# =====================================================================
def aplicar_filtro_pasabajas(data, cutoff_hz=6.0, fs_hz=60.0, order=2):
  """Applies a bidirectional low-pass Butterworth filter (filtfilt).

  Removes jitter and velocity spikes without introducing time lag.
  """
  nyquist = 0.5 * fs_hz
  normal_cutoff = cutoff_hz / nyquist
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return filtfilt(b, a, data, axis=0)


FS_SISTEMA = 60.0  # Estimated sampling frequency of the system (60 Hz)

print('🧹 Aplicando filtrado pasabajas de fase cero...')

# A) Filter Y outputs (OptiTrack: Positions and Orientations)
cols_y_a_filtrar = [
    'rel_x',
    'rel_y',
    'rel_z',
    'rel_qx',
    'rel_qy',
    'rel_qz',
    'rel_qw',
]
for col in cols_y_a_filtrar:
  df[col] = aplicar_filtro_pasabajas(
      df[col].values, cutoff_hz=6.0, fs_hz=FS_SISTEMA
  )

# B) Filter X force sensors (Torques and Tensions)
for i in range(1, 5):
  df[f'couple_m{i}'] = aplicar_filtro_pasabajas(
      df[f'couple_m{i}'].values, cutoff_hz=4.0, fs_hz=FS_SISTEMA
  )
  df[f'tension_m{i}'] = aplicar_filtro_pasabajas(
      df[f'tension_m{i}'].values, cutoff_hz=4.0, fs_hz=FS_SISTEMA
  )

# NOTE: The motor positions (delta_real_m, delta_meta_m) and delta_t
# are NOT filtered, in order to keep the exact switching instant intact.

print('✓ Filtrado completado en OptiTrack, Torques y Tensiones.')


# =====================================================================
# 5. HYBRID SCALING FUNCTION
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
      min_t, max_t = 45.0, 60.0

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

    scaled_data[:, col] = -1 + 2 * (data_array[:, col] - min_t) / (
        max_t - min_t
    )
    dict_parametros[col_name] = {'min_t': float(min_t), 'max_t': float(max_t)}

  return scaled_data, dict_parametros


def procesar_y_guardar(columnas_x, columnas_y, nombre_archivo):
  X_scaled, X_params = calcular_limites_y_escalar(df, columnas_x)
  Y_scaled, Y_params = calcular_limites_y_escalar(df, columnas_y)

  dataset = {'X': X_scaled, 'Y': Y_scaled}
  np.save(f'{nombre_archivo}.npy', dataset)

  metadata = {
      'X_transformer': X_params,
      'Y_transformer': Y_params,
      'home_motores': ANGULOS_CALIBRACION,
  }
  with open(f'{nombre_archivo}_params.json', 'w') as f:
    json.dump(metadata, f, indent=4)

  print(f'✓ Generado: {nombre_archivo}.npy')


# =====================================================================
# 6. GENERATION OF ALL POSSIBLE COMBINATIONS (v01 to v16)
# =====================================================================
X_base = [
    'delta_real_m1',
    'delta_real_m2',
    'delta_real_m3',
    'delta_real_m4',
    'delta_t',
]
X_meta = ['delta_meta_m1', 'delta_meta_m2', 'delta_meta_m3', 'delta_meta_m4']
X_torque = ['couple_m1', 'couple_m2', 'couple_m3', 'couple_m4']
X_tension = ['tension_m1', 'tension_m2', 'tension_m3', 'tension_m4']

Y_sin_rot = ['rel_x', 'rel_y', 'rel_z']
Y_con_rot = ['rel_x', 'rel_y', 'rel_z', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']

configuraciones_X = {
    'real': X_base,
    'real_meta': X_base + X_meta,
    'real_torque': X_base + X_torque,
    'real_tension': X_base + X_tension,
    'real_torque_tension': X_base + X_torque + X_tension,
    'real_meta_torque': X_base + X_meta + X_torque,
    'real_meta_tension': X_base + X_meta + X_tension,
    'completo': X_base + X_meta + X_torque + X_tension,
}

configuraciones_Y = {'sinRot': Y_sin_rot, 'conRot': Y_con_rot}

orden_datasets = [
    ('real', 'sinRot'),
    ('real_meta', 'sinRot'),
    ('real_torque', 'sinRot'),
    ('real_tension', 'sinRot'),
    ('real_torque_tension', 'sinRot'),
    ('real_meta_torque', 'sinRot'),
    ('real_meta_tension', 'sinRot'),
    ('completo', 'sinRot'),
    ('real', 'conRot'),
    ('real_meta', 'conRot'),
    ('real_torque', 'conRot'),
    ('real_tension', 'conRot'),
    ('real_torque_tension', 'conRot'),
    ('real_meta_torque', 'conRot'),
    ('real_meta_tension', 'conRot'),
    ('completo', 'conRot'),
]

print('\nProcesando matrices de entrenamiento...')
v_idx = 1
for x_tag, y_tag in orden_datasets:
  x_cols = configuraciones_X[x_tag]
  y_cols = configuraciones_Y[y_tag]
  nombre_archivo = f'dataset_v{v_idx:02d}_{x_tag}_{y_tag}'
  procesar_y_guardar(x_cols, y_cols, nombre_archivo)
  v_idx += 1

# =====================================================================
# 7. 3D VISUALIZATION (Subsampling: 1 out of every 4 samples)
# =====================================================================
df_sub = df.iloc[::4]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    df_sub['rel_x'],
    df_sub['rel_y'],
    df_sub['rel_z'],
    c=df_sub['t_relativo'],
    cmap='magma',
    s=6,
)

ax.scatter(
    [0],
    [0],
    [0],
    color='red',
    s=150,
    marker='^',
    label='Base del Robot (Origen 0,0,0)',
)

ax.set_title('Espacio de Trabajo Tridimensional del Efector (Datos Filtrados)')
ax.set_xlabel('Eje X (m)')
ax.set_ylabel('Eje Y (m - Altura)')
ax.set_zlabel('Eje Z (m)')
ax.legend()
fig.colorbar(sc, ax=ax, label='Tiempo Relativo (s)')
plt.show()