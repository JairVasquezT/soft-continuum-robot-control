import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

# =====================================================================
# 1. CARGA Y LECTURA DEL CSV
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

df = pd.read_csv('logs_trayectoria_final_largo.csv', names=columnas)

#Submuestreo a 30Hz
#df = df.iloc[::2].reset_index(drop=True)

# Cálculo del tiempo delta entre muestras
df['delta_t'] = df['t_relativo'].diff().bfill()
df = df.iloc[1:].reset_index(drop=True)

print(f'✓ Muestras cargadas correctamente ({len(df)} filas)')

# =====================================================================
# 2. SECCIÓN DE CALIBRACIÓN (CENTROS FIJOS Y RANGOS)
# =====================================================================
ANGULOS_CALIBRACION = {'m1': 1871.0, 'm2': 1951.0, 'm3': 1485.0, 'm4': 1712.0}
RANGOS_MANUALES = {'m1': 650.0, 'm2': 650.0, 'm3': 750.0, 'm4': 750.0}

LIMITES_ESPACIALES = {
    'rel_x': (-0.220, 0.220),
    'rel_y': (0.180, 0.330),
    'rel_z': (-0.220, 0.220),
}

LIMITES_TORQUE = (-500.0, 500.0)
LIMITES_TENSION = (100.0, 2500.0)

# Posiciones relativas de motores (NO SE FILTRAN para preservar la dinámica de conmutación)
for i in range(1, 5):
  df[f'delta_real_m{i}'] = df[f'real_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']
  df[f'delta_meta_m{i}'] = df[f'meta_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']

# =====================================================================
# 3-4. TRANSFORMACIÓN DE POSICIÓN Y ROTACIÓN RELATIVA AL MARCO DE LA BASE
# =====================================================================
# Misma fórmula que usa MPC.py en inferencia (_procesar_lecturas_cinematicas):
# p_rel = R_base.inv().apply(p_efector - p_base)
# q_rel = (R_base.inv() * R_efector).as_quat()
# Antes esto era una resta/cuaternión "en crudo" (marco del mundo), sin rotar
# al marco de la base -- desalineado con lo que MPC.py calcula en tiempo real.
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
# 4.5. FILTRADO ESPECIALIZADO DE FASE CERO
# =====================================================================
def aplicar_filtro_pasabajas(data, cutoff_hz=4.0, fs_hz=60.0, order=2):
  """Aplica un filtro Butterworth pasabajas bidireccional (filtfilt)."""
  nyquist = 0.5 * fs_hz
  normal_cutoff = cutoff_hz / nyquist
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return filtfilt(b, a, data, axis=0)


def filtrar_y_normalizar_cuaterniones(
    df_quat, cutoff_hz=6.0, fs_hz=60.0, order=2
):
  """Filtra cuaterniones preservando la continuidad de signo y la norma 1."""
  q_vals = df_quat[['rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']].values.copy()

  # 1. Desenrollado de signo (Sign Unwrap) para evitar saltos entre q y -q
  for i in range(1, len(q_vals)):
    if np.dot(q_vals[i], q_vals[i - 1]) < 0:
      q_vals[i] = -q_vals[i]

  # 2. Aplicar filtro lineal a las 4 componentes
  q_filt = aplicar_filtro_pasabajas(
      q_vals, cutoff_hz=cutoff_hz, fs_hz=fs_hz, order=order
  )

  # 3. Re-normalización geométrica (||q|| = 1)
  normas = np.linalg.norm(q_filt, axis=1, keepdims=True)
  normas[normas == 0] = 1.0
  return q_filt / normas


FS_SISTEMA = 60.0  # Frecuencia de muestreo (60 Hz)

print('🧹 Aplicando filtrado pasabajas selectivo...')

# A) Filtrar Posiciones Relativas (OptiTrack X, Y, Z a 6 Hz)
for col in ['rel_x', 'rel_y', 'rel_z']:
  df[col] = aplicar_filtro_pasabajas(
      df[col].values, cutoff_hz=6.0, fs_hz=FS_SISTEMA
  )

# B) Filtrar Cuaterniones con Desenrollado y Re-normalización
q_normalizados = filtrar_y_normalizar_cuaterniones(
    df[['rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']],
    cutoff_hz=6.0,
    fs_hz=FS_SISTEMA,
)
df['rel_qx'], df['rel_qy'], df['rel_qz'], df['rel_qw'] = (
    q_normalizados[:, 0],
    q_normalizados[:, 1],
    q_normalizados[:, 2],
    q_normalizados[:, 3],
)

# C) Filtrar Sensores de Fuerza y Tensión (Torques y Tensiones a 3.5 Hz)
for i in range(1, 5):
  df[f'couple_m{i}'] = aplicar_filtro_pasabajas(
      df[f'couple_m{i}'].values, cutoff_hz=3.5, fs_hz=FS_SISTEMA
  )
  df[f'tension_m{i}'] = aplicar_filtro_pasabajas(
      df[f'tension_m{i}'].values, cutoff_hz=3.5, fs_hz=FS_SISTEMA
  )

# NOTA: Posiciones de motores (delta_real_m, delta_meta_m) no se filtran.

print(
    '✓ Filtrado completado: Posiciones 3D, Cuaterniones (normalizados), Torques'
    ' y Tensiones.'
)


# =====================================================================
# 5. FUNCIÓN DE ESCALADO HÍBRIDA CON SUFIJO _FILT
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
      min_t, max_t = 0.016, 0.033 #60Hz
      #min_t, max_t = 0.032, 0.054 #30Hz

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


def procesar_y_guardar(columnas_x, columnas_y, nombre_archivo_base):
  # Añadir el sufijo '_filt' a los archivos generados
  nombre_archivo = f'{nombre_archivo_base}_filt'

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

  print(f'✓ Generado: {nombre_archivo}.npy y {nombre_archivo}_params.json')


# =====================================================================
# 6. GENERACIÓN DE TODAS LAS COMBINACIONES FILTRADAS (v01 a v16)
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

print('\nProcesando matrices de entrenamiento filtradas...')
v_idx = 1
for x_tag, y_tag in orden_datasets:
  x_cols = configuraciones_X[x_tag]
  y_cols = configuraciones_Y[y_tag]
  nombre_base = f'dataset_v{v_idx:02d}_{x_tag}_{y_tag}'
  procesar_y_guardar(x_cols, y_cols, nombre_base)
  v_idx += 1

# =====================================================================
# 7. VISUALISATION 3D (Sous-échantillonnage : 1 donnée sur 4)
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
    label='Base du Robot (Origine 0,0,0)',
)

ax.set_title(
    "Espace de Travail Tridimensionnel de l'Effecteur (Données Filtrées)"
)
ax.set_xlabel('Axe X (m)')
ax.set_ylabel('Axe Y (m - Hauteur)')
ax.set_zlabel('Axe Z (m)')
ax.legend()
fig.colorbar(sc, ax=ax, label='Temps Relatif (s)')
plt.show()

