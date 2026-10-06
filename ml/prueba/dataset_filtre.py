import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

# =====================================================================
# 1. LOADING AND READING THE CSV
# =====================================================================
columnas = [
    'timestamp',
    't_unix',
    't_relativo',
    'combo_id',
    'meta_m1',
    'meta_m2',
    'meta_m3',
    'meta_m4',
    'real_m1',
    'real_m2',
    'real_m3',
    'real_m4',
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

df = pd.read_csv('largo_20260710_130354_284.csv', names=columnas)

#Subsampling to 30Hz
#df = df.iloc[::2].reset_index(drop=True)

# Computation of the time delta between samples
df['delta_t'] = df['t_relativo'].diff().bfill()
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

LIMITES_TORQUE = (-500.0, 500.0)
LIMITES_TENSION = (500.0, 2000.0)

# Relative motor positions (NOT FILTERED, to preserve the switching dynamics)
for i in range(1, 5):
  df[f'delta_real_m{i}'] = df[f'real_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']
  df[f'delta_meta_m{i}'] = df[f'meta_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']

# =====================================================================
# 3-4. POSITION AND ROTATION TRANSFORMATION RELATIVE TO THE BASE FRAME
# =====================================================================
# Same formula used by MPC.py at inference (_procesar_lecturas_cinematicas):
# p_rel = R_base.inv().apply(p_efector - p_base)
# q_rel = (R_base.inv() * R_efector).as_quat()
# Previously this was a "raw" subtraction/quaternion (world frame), without rotating
# to the base frame -- misaligned with what MPC.py computes in real time.
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
# 4.5. SPECIALIZED ZERO-PHASE FILTERING
# =====================================================================
def aplicar_filtro_pasabajas(data, cutoff_hz=4.0, fs_hz=60.0, order=2):
  """Applies a bidirectional low-pass Butterworth filter (filtfilt)."""
  nyquist = 0.5 * fs_hz
  normal_cutoff = cutoff_hz / nyquist
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return filtfilt(b, a, data, axis=0)


def filtrar_y_normalizar_cuaterniones(
    df_quat, cutoff_hz=6.0, fs_hz=60.0, order=2
):
  """Filters quaternions preserving sign continuity and unit norm."""
  q_vals = df_quat[['rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']].values.copy()

  # 1. Sign unwrapping (Sign Unwrap) to avoid jumps between q and -q
  for i in range(1, len(q_vals)):
    if np.dot(q_vals[i], q_vals[i - 1]) < 0:
      q_vals[i] = -q_vals[i]

  # 2. Apply the linear filter to the 4 components
  q_filt = aplicar_filtro_pasabajas(
      q_vals, cutoff_hz=cutoff_hz, fs_hz=fs_hz, order=order
  )

  # 3. Geometric re-normalization (||q|| = 1)
  normas = np.linalg.norm(q_filt, axis=1, keepdims=True)
  normas[normas == 0] = 1.0
  return q_filt / normas


FS_SISTEMA = 60.0  # Sampling frequency (60 Hz)

print('🧹 Aplicando filtrado pasabajas selectivo...')

# A) Filter Relative Positions (OptiTrack X, Y, Z at 6 Hz)
for col in ['rel_x', 'rel_y', 'rel_z']:
  df[col] = aplicar_filtro_pasabajas(
      df[col].values, cutoff_hz=6.0, fs_hz=FS_SISTEMA
  )

# B) Filter Quaternions with Unwrapping and Re-normalization
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


# NOTE: Motor positions (delta_real_m, delta_meta_m) are not filtered.

print(
    '✓ Filtrado completado: Posiciones 3D, Cuaterniones (normalizados), Torques'
    ' y Tensiones.'
)


# =====================================================================
# 5. HYBRID SCALING FUNCTION WITH _FILT SUFFIX
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
  # Add the '_filt' suffix to the generated files
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
# 6. GENERATION OF ALL FILTERED COMBINATIONS (v01 to v16)
# =====================================================================
X_base = [
    'delta_real_m1',
    'delta_real_m2',
    'delta_real_m3',
    'delta_real_m4',
    'delta_t',
]
X_meta = ['delta_meta_m1', 'delta_meta_m2', 'delta_meta_m3', 'delta_meta_m4']

Y_sin_rot = ['rel_x', 'rel_y', 'rel_z']
Y_con_rot = ['rel_x', 'rel_y', 'rel_z', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']

configuraciones_X = {
    'real': X_base,
    'real_meta': X_base + X_meta,
}

configuraciones_Y = {'sinRot': Y_sin_rot, 'conRot': Y_con_rot}

orden_datasets = [
    ('real', 'sinRot'),
    ('real_meta', 'sinRot'),
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
# 7. 3D VISUALIZATION (Subsampling: 1 sample out of 4)
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

# =====================================================================
# 8. SAMPLING FREQUENCY ANALYSIS (0 - 30 Hz)
# =====================================================================
# Calculation of sampling frequency per sample: Fs = 1 / delta_t
frecuencias = 1.0 / df['delta_t'].replace(0, np.nan)
frecuencias = frecuencias.dropna()

# Global capture metrics
total_donnees = len(df)
temps_total_s = df['t_relativo'].iloc[-1] - df['t_relativo'].iloc[0]

# Filter in range [0, 30] Hz
freq_filtradas = frecuencias[(frecuencias >= 0.0) & (frecuencias <= 30.0)]
frecuencia_promedio = freq_filtradas.mean()

# Histogram configuration (30 bins)
num_bins = 30
counts, bin_edges = np.histogram(freq_filtradas, bins=num_bins, range=(0.0, 30.0))

fig, ax = plt.subplots(figsize=(12, 6))

# Gradient color per bar
cm = plt.colormaps['viridis']
norm = plt.Normalize(vmin=0.0, vmax=30.0)

for i in range(num_bins):
    bin_min = bin_edges[i]
    bin_max = bin_edges[i + 1]
    count = int(counts[i])

    # Draw bar
    ax.bar(
        bin_min,
        count,
        width=bin_max - bin_min,
        align='edge',
        color=cm(norm((bin_min + bin_max) / 2.0)),
        edgecolor='black',
        alpha=0.8,
    )

    # Label count on top of bar if count > 0
    if count > 0:
        ax.text(
            (bin_min + bin_max) / 2.0,
            count + (max(counts) * 0.01),
            f'{count}',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=90 if num_bins > 20 else 0,
        )

# Marker line for the mean frequency
ax.axvline(
    x=frecuencia_promedio,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Moyenne : {frecuencia_promedio:.2f} Hz'
)

# Formatting graph labels (French)
ax.set_title(
    f"Distribution de la Fréquence d'Échantillonnage (0 - 30 Hz)\n"
    f'Total : {total_donnees:,} échantillons | Durée : {temps_total_s:.2f} s | Moyenne : {frecuencia_promedio:.2f} Hz',
    fontsize=12,
    pad=15,
)
ax.set_xlabel("Fréquence d'Échantillonnage (Hz)", fontsize=11)
ax.set_ylabel("Nombre d'Échantillons", fontsize=11)
ax.set_xlim(0.0, 30.0)
ax.set_xticks(bin_edges)
plt.xticks(rotation=45, fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()