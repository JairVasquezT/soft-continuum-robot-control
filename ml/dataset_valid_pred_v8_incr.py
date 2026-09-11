import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn

# =====================================================================
# 1. CONFIGURACIÓN Y ARCHIVOS
# =====================================================================
PATH_CSV_TEST = 'corto_20260730_195350_730.csv'
PATH_METADATA = 'dataset_pred_v08_completo_10_conRot_directo_filt_params.json'
PATH_MODEL_WEIGHTS = 'best_mpc_pinn_predictor_incr.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🚀 Evaluando en dispositivo: {DEVICE}')

# Cargar metadatos
with open(PATH_METADATA, 'r') as f:
  metadata = json.load(f)

X_TRANS = metadata['X_transformer']
Y_TRANS = metadata['Y_transformer']
ANGULOS_CALIBRACION = metadata['home_motores']
T_IN = metadata['t_in']
T_OUT = metadata['t_out']
MAX_DELTA_T = 1.0 / 55.0  # 55 Hz = ~18.18 ms

RANGOS_MANUALES = {'m1': 650.0, 'm2': 650.0, 'm3': 750.0, 'm4': 750.0}
LIMITES_ESPACIALES = {
    'rel_x': (-0.220, 0.220),
    'rel_y': (0.180, 0.330),
    'rel_z': (-0.220, 0.220),
}
LIMITES_TORQUE = (-500.0, 500.0)
LIMITES_TENSION = (500.0, 2000.0)

# =====================================================================
# 2. PROCESAMIENTO Y FILTRADO DEL CSV DE PRUEBA
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

df = pd.read_csv(PATH_CSV_TEST, names=columnas)
df['delta_t'] = df['t_relativo'].diff().bfill()
df = df.iloc[1:].reset_index(drop=True)

# Geometría y Motores
for i in range(1, 5):
  df[f'delta_real_m{i}'] = df[f'real_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']
  df[f'delta_meta_m{i}'] = df[f'meta_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']

# Posición relativa al marco de la BASE
p_base_arr = df[['base_x', 'base_y', 'base_z']].values
p_efector_arr = df[['efector_x', 'efector_y', 'efector_z']].values
q_base_arr = df[['base_qx', 'base_qy', 'base_qz', 'base_qw']].values

r_b = R.from_quat(q_base_arr)
p_rel = r_b.inv().apply(p_efector_arr - p_base_arr)

df['rel_x'] = p_rel[:, 0]
df['rel_y'] = p_rel[:, 1]
df['rel_z'] = p_rel[:, 2]


# Filtro Pasabajas
def aplicar_filtro_pasabajas(data, cutoff_hz=6.0, fs_hz=60.0, order=2):
  nyquist = 0.5 * fs_hz
  normal_cutoff = cutoff_hz / nyquist
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return filtfilt(b, a, data, axis=0)


for col in ['rel_x', 'rel_y', 'rel_z']:
  df[col] = aplicar_filtro_pasabajas(df[col].values, cutoff_hz=6.0, fs_hz=60.0)

for i in range(1, 5):
  df[f'couple_m{i}'] = aplicar_filtro_pasabajas(
      df[f'couple_m{i}'].values, cutoff_hz=3.5, fs_hz=60.0
  )
  df[f'tension_m{i}'] = aplicar_filtro_pasabajas(
      df[f'tension_m{i}'].values, cutoff_hz=3.5, fs_hz=60.0
  )


# Escalado estricto usando PARÁMETROS DE ENTRENAMIENTO
def escalar_con_parametros(df_raw, columnas_nombres, dict_params):
  scaled_matrix = np.zeros((len(df_raw), len(columnas_nombres)))
  for idx, col in enumerate(columnas_nombres):
    min_t = dict_params[col]['min_t']
    max_t = dict_params[col]['max_t']
    val = df_raw[col].values
    scaled = -1 + 2 * (val - min_t) / (max_t - min_t)
    scaled_matrix[:, idx] = np.clip(scaled, -1.2, 1.2)
  return scaled_matrix


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

cols_u_fut = [
    'delta_meta_m1',
    'delta_meta_m2',
    'delta_meta_m3',
    'delta_meta_m4',
]

# 🔴 CAMBIO AQUÍ: Solo 3 salidas
cols_y = ['rel_x', 'rel_y', 'rel_z']

X_scaled = escalar_con_parametros(df, cols_x_hist, X_TRANS)
U_scaled = escalar_con_parametros(df, cols_u_fut, X_TRANS)
Y_scaled = escalar_con_parametros(df, cols_y, Y_TRANS)

# Generar Secuencias
X_hist_list, U_cand_list, Y_fut_list, Y_t0_list = [], [], [], []
for i in range(len(df) - T_IN - T_OUT):
  window_delta_t = df['delta_t'].iloc[i : i + T_IN + T_OUT].values
  if np.any(window_delta_t > MAX_DELTA_T):
    continue

  u_futuros = U_scaled[i + T_IN : i + T_IN + T_OUT]
  if not np.all(u_futuros == u_futuros[0]):
    continue

  X_hist_list.append(X_scaled[i : i + T_IN])
  U_cand_list.append(U_scaled[i + T_IN])
  Y_fut_list.append(Y_scaled[i + T_IN : i + T_IN + T_OUT])
  # Posición escalada en t0 (último paso del historial) -- el modelo
  # incremental predice desplazamientos respecto a ESTA fila, igual que en
  # dataset_pred_filt.py/train_pred_incr.py.
  Y_t0_list.append(Y_scaled[i + T_IN - 1])

X_test = torch.tensor(np.array(X_hist_list), dtype=torch.float32)
U_test = torch.tensor(np.array(U_cand_list), dtype=torch.float32)
Y_test_scaled = torch.tensor(np.array(Y_fut_list), dtype=torch.float32)
Y_t0_scaled = np.array(Y_t0_list)  # [N, 3] -- posición absoluta escalada en t0

print(
    f'✓ Dataset de Prueba Cargado: {len(X_test)} muestras válidas generadas.\n'
)


class MPCDirectPredictor(nn.Module):

  def __init__(
      self,
      input_hist_dim,
      u_cand_dim=4,
      hidden_size=128,
      num_layers=2,
      t_out=10,
      output_dim=3,  # 🔴 CAMBIO AQUÍ: 3 dimensiones por defecto
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


checkpoint = torch.load(PATH_MODEL_WEIGHTS, map_location=DEVICE)
model = MPCDirectPredictor(
    input_hist_dim=checkpoint['input_hist_dim'],
    u_cand_dim=checkpoint['u_cand_dim'],
    hidden_size=checkpoint['hidden_size'],
    num_layers=checkpoint['num_layers'],
    t_out=checkpoint['t_out'],
    output_dim=checkpoint['output_dim'],
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# =====================================================================
# 4. INFERENCIA Y EVALUACIÓN EN UNIDADES REALES (MILÍMETROS)
# =====================================================================
with torch.no_grad():
  Y_pred_scaled = model(X_test.to(DEVICE), U_test.to(DEVICE)).cpu().numpy()

Y_true_scaled = Y_test_scaled.numpy()


# Función de Desescalado -- para una POSICIÓN ABSOLUTA en [-1,1] (lleva el
# offset/bias "+1"+min).
def desescalar_var(val_scaled, col_name, dict_params):
  min_t = dict_params[col_name]['min_t']
  max_t = dict_params[col_name]['max_t']
  return min_t + (val_scaled + 1.0) * (max_t - min_t) / 2.0


# Función de Desescalado -- para un DESPLAZAMIENTO (diferencia entre dos
# valores ya en [-1,1]), SIN el offset/bias: un delta no tiene "origen"
# propio (ver misma fórmula en train_pred_incr.py/PINNLossMPC).
def desescalar_delta_var(delta_scaled, col_name, dict_params):
  min_t = dict_params[col_name]['min_t']
  max_t = dict_params[col_name]['max_t']
  return delta_scaled * (max_t - min_t) / 2.0


# Este modelo predice DESPLAZAMIENTOS INCREMENTALES (ΔX, ΔY, ΔZ) respecto a
# t0, no posición absoluta -- hay que desescalar el delta y sumarle la
# posición REAL en t0 para reconstruir la trayectoria absoluta comparable
# contra Y_true (que sigue siendo posición absoluta, tal cual sale del
# dataset).
X_t0_mm = desescalar_var(Y_t0_scaled[:, 0], 'rel_x', Y_TRANS) * 1000.0  # [N]
Y_t0_mm = desescalar_var(Y_t0_scaled[:, 1], 'rel_y', Y_TRANS) * 1000.0
Z_t0_mm = desescalar_var(Y_t0_scaled[:, 2], 'rel_z', Y_TRANS) * 1000.0

dX_pred_mm = desescalar_delta_var(Y_pred_scaled[:, :, 0], 'rel_x', Y_TRANS) * 1000.0  # [N, T_OUT]
dY_pred_mm = desescalar_delta_var(Y_pred_scaled[:, :, 1], 'rel_y', Y_TRANS) * 1000.0
dZ_pred_mm = desescalar_delta_var(Y_pred_scaled[:, :, 2], 'rel_z', Y_TRANS) * 1000.0

# Convertir a Metros -> Milímetros (posición ABSOLUTA reconstruida = t0 + delta)
X_pred_mm = X_t0_mm[:, None] + dX_pred_mm
Y_pred_mm = Y_t0_mm[:, None] + dY_pred_mm
Z_pred_mm = Z_t0_mm[:, None] + dZ_pred_mm

X_true_mm = desescalar_var(Y_true_scaled[:, :, 0], 'rel_x', Y_TRANS) * 1000.0
Y_true_mm = desescalar_var(Y_true_scaled[:, :, 1], 'rel_y', Y_TRANS) * 1000.0
Z_true_mm = desescalar_var(Y_true_scaled[:, :, 2], 'rel_z', Y_TRANS) * 1000.0

mae_x = np.mean(np.abs(X_pred_mm - X_true_mm))
mae_y = np.mean(np.abs(Y_pred_mm - Y_true_mm))
mae_z = np.mean(np.abs(Z_pred_mm - Z_true_mm))

dist_3d_mm = np.sqrt(
    (X_pred_mm - X_true_mm) ** 2
    + (Y_pred_mm - Y_true_mm) ** 2
    + (Z_pred_mm - Z_true_mm) ** 2
)
mae_3d_global = np.mean(dist_3d_mm)

print('====================================================')
print('📊 RESULTADOS DE EVALUACIÓN DEL PREDICTOR EN DATASET CORTO')
print('====================================================')
print(f'Error MAE Eje X: {mae_x:.3f} mm')
print(f'Error MAE Eje Y: {mae_y:.3f} mm')
print(f'Error MAE Eje Z: {mae_z:.3f} mm')
print(f'📐 ERROR DE DISTANCIA EUCLÍDEA 3D PROMEDIO: {mae_3d_global:.3f} mm\n')

print('--- ERROR PROMEDIO 3D PASO A PASO EN EL HORIZONTE FUTURO ---')
for paso in range(T_OUT):
  err_paso = np.mean(dist_3d_mm[:, paso])
  tiempo_futuro_ms = (paso + 1) * (1000.0 / 60.0)
  print(
      f'Paso t+{paso+1:02d} ({tiempo_futuro_ms:.1f} ms): Error 3D ='
      f' {err_paso:.3f} mm'
  )

# =====================================================================
# 4.5. ERROR ANGULAR ENTRE EL VECTOR DE DESPLAZAMIENTO PREDICHO Y EL REAL
# =====================================================================
# Para cada transición consecutiva del horizonte (t+k -> t+k+1), compara la
# DIRECCIÓN del desplazamiento 3D predicho contra la real, vía el ángulo
# entre ambos vectores (0° = misma dirección, 180° = opuestos). Es la
# contraparte, en grados y como métrica de evaluación (no de entrenamiento),
# de la similitud coseno que usa loss_direction_xz en train_pred.py -- pero
# acá en 3D completo, no solo X-Z, para tener el panorama completo.
pos_pred_mm = np.stack([X_pred_mm, Y_pred_mm, Z_pred_mm], axis=-1)  # [N, T_OUT, 3]
pos_true_mm = np.stack([X_true_mm, Y_true_mm, Z_true_mm], axis=-1)  # [N, T_OUT, 3]

dir_pred = pos_pred_mm[:, 1:, :] - pos_pred_mm[:, :-1, :]  # [N, T_OUT-1, 3]
dir_true = pos_true_mm[:, 1:, :] - pos_true_mm[:, :-1, :]  # [N, T_OUT-1, 3]

norm_pred = np.linalg.norm(dir_pred, axis=-1)
norm_true = np.linalg.norm(dir_true, axis=-1)
EPS_MAGNITUD_MM = 0.5  # transiciones donde el desplazamiento real es < 0.5mm
# no tienen una dirección bien definida (ruido) -- se excluyen del promedio.
validas = (norm_pred > EPS_MAGNITUD_MM) & (norm_true > EPS_MAGNITUD_MM)

cos_theta = np.sum(dir_pred * dir_true, axis=-1) / (norm_pred * norm_true + 1e-9)
cos_theta = np.clip(cos_theta, -1.0, 1.0)
angulo_deg = np.degrees(np.arccos(cos_theta))

print('\n--- ERROR ANGULAR (3D) ENTRE DESPLAZAMIENTO PREDICHO Y REAL, POR TRANSICIÓN ---')
for k in range(T_OUT - 1):
  mask_k = validas[:, k]
  n_validas = mask_k.sum()
  if n_validas == 0:
    print(f'Transición t+{k+1:02d} -> t+{k+2:02d}: sin transiciones válidas '
          f'(desplazamiento real < {EPS_MAGNITUD_MM}mm en todas)')
    continue
  angulo_medio = np.mean(angulo_deg[mask_k, k])
  print(f'Transición t+{k+1:02d} -> t+{k+2:02d}: ángulo medio = {angulo_medio:6.2f}° '
        f'({n_validas}/{len(mask_k)} muestras válidas)')

angulo_medio_global = np.mean(angulo_deg[validas])
print(f'\n📐 ÁNGULO MEDIO GLOBAL (todas las transiciones válidas): '
      f'{angulo_medio_global:.2f}°')

# Filtrar solo transiciones con desplazamiento por encima del ruido de medición.
# 3mm por paso (16.7ms) implicaría ~180mm/s, más rápido que cualquier tramo de
# este CSV de test (desplazamiento real máximo observado: 1.84mm/paso, media
# 0.63mm) -- con ese umbral no queda ninguna muestra. 1.0mm sigue estando muy
# por encima del piso de ruido típico de OptiTrack (sub-milimétrico) y deja
# ~17% de las transiciones (las de movimiento más franco) para promediar.
UMBRAL_DESPLAZAMIENTO_MM = 1.0

# pos_t1/pos_t2 = posición REAL en cada paso consecutivo del horizonte
# (pos_true_mm[:, 1:, :] - pos_true_mm[:, :-1, :] es exactamente dir_true,
# ya en mm, así que desplazamiento == norm_true calculado arriba).
pos_t1 = pos_true_mm[:, :-1, :]
pos_t2 = pos_true_mm[:, 1:, :]
desplazamiento = np.linalg.norm(pos_t2 - pos_t1, axis=-1)  # ya en mm

mask_movimiento_real = desplazamiento > UMBRAL_DESPLAZAMIENTO_MM

angulo_filtrado = angulo_deg[mask_movimiento_real]
print(f'Ángulo medio (solo desplazamientos > {UMBRAL_DESPLAZAMIENTO_MM}mm): '
      f'{angulo_filtrado.mean():.2f}° ({mask_movimiento_real.sum()} muestras)')

# =====================================================================
# 5. VISUALIZACIÓN: ROLLOUT CADA 10 VENTANAS SOBRE LA TRAYECTORIA REAL
# =====================================================================
PASO_MUESTRAS = 10
N_MAX = 4000  # Límite a los primeros 1000 datos

# Acortar los datos reales a los primeros N_MAX puntos
real_continuo = np.stack([X_true_mm[:N_MAX, 0], Y_true_mm[:N_MAX, 0], Z_true_mm[:N_MAX, 0]], axis=1)
tiempo_continuo = np.arange(len(real_continuo))

# Asegurar que las anclas de predicción no superen el rango de 1000 datos
limite_anclas = min(len(X_test), N_MAX) - T_OUT
anclas = list(range(0, max(0, limite_anclas), PASO_MUESTRAS))
cmap = plt.get_cmap('plasma')

fig2 = plt.figure(figsize=(16, 8))

# Subplot 3D
ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
ax1.plot(
    real_continuo[:, 0], real_continuo[:, 1], real_continuo[:, 2],
    color='black', linewidth=1, alpha=0.4, label='Real (continuo)',
)
for j, i in enumerate(anclas):
    color = cmap(j / max(1, len(anclas) - 1))
    origen = real_continuo[i:i + 1]
    rollout = np.stack([X_pred_mm[i, :], Y_pred_mm[i, :], Z_pred_mm[i, :]], axis=1)
    tramo = np.vstack([origen, rollout])
    ax1.plot(tramo[:, 0], tramo[:, 1], tramo[:, 2], color=color, linewidth=2)
    ax1.scatter(*origen[0], color=color, s=15)
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title(f'Real (negro) + {len(anclas)} rollouts de {T_OUT} pasos (primeros {N_MAX} datos)')
ax1.legend()

# Subplots 1D por eje
ejes = [('X', X_true_mm[:N_MAX, 0], X_pred_mm), 
        ('Y', Y_true_mm[:N_MAX, 0], Y_pred_mm),
        ('Z', Z_true_mm[:N_MAX, 0], Z_pred_mm)]

for k, (nombre_eje, real_eje, pred_eje) in enumerate(ejes):
    ax = fig2.add_subplot(3, 2, 2 * (k + 1))
    ax.plot(tiempo_continuo, real_eje, color='black', linewidth=1, alpha=0.5,
            label='Real' if k == 0 else None)
    for j, i in enumerate(anclas):
        color = cmap(j / max(1, len(anclas) - 1))
        t_rollout = np.arange(i, i + T_OUT + 1)
        valores_rollout = np.concatenate([[real_eje[i]], pred_eje[i, :]])
        ax.plot(t_rollout, valores_rollout, color=color, linewidth=1.5)
    ax.set_ylabel(f'{nombre_eje} (mm)')
    if k == 0:
        ax.set_title(f'Rollouts de {T_OUT} pasos vs. trayectoria real, por eje (primeros {N_MAX} datos)')
        ax.legend()
    if k == 2:
        ax.set_xlabel('Muestra temporal')

plt.tight_layout()
plt.show()