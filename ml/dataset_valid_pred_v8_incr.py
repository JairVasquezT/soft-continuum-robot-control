import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn

# =====================================================================
# 1. CONFIGURATION AND FILES
# =====================================================================
PATH_CSV_TEST = 'corto_20260730_195350_730.csv'
PATH_METADATA = 'dataset_pred_v08_completo_10_conRot_directo_filt_params.json'
PATH_MODEL_WEIGHTS = 'best_mpc_pinn_predictor_incr.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🚀 Evaluando en dispositivo: {DEVICE}')

# Load metadata
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
# 2. PROCESSING AND FILTERING OF THE TEST CSV
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

# Geometry and Motors
for i in range(1, 5):
  df[f'delta_real_m{i}'] = df[f'real_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']
  df[f'delta_meta_m{i}'] = df[f'meta_m{i}'] - ANGULOS_CALIBRACION[f'm{i}']

# Position relative to the BASE frame
p_base_arr = df[['base_x', 'base_y', 'base_z']].values
p_efector_arr = df[['efector_x', 'efector_y', 'efector_z']].values
q_base_arr = df[['base_qx', 'base_qy', 'base_qz', 'base_qw']].values

r_b = R.from_quat(q_base_arr)
p_rel = r_b.inv().apply(p_efector_arr - p_base_arr)

df['rel_x'] = p_rel[:, 0]
df['rel_y'] = p_rel[:, 1]
df['rel_z'] = p_rel[:, 2]


# Low-pass Filter
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


# Strict scaling using TRAINING PARAMETERS
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

# 🔴 CHANGE HERE: Only 3 outputs
cols_y = ['rel_x', 'rel_y', 'rel_z']

X_scaled = escalar_con_parametros(df, cols_x_hist, X_TRANS)
U_scaled = escalar_con_parametros(df, cols_u_fut, X_TRANS)
Y_scaled = escalar_con_parametros(df, cols_y, Y_TRANS)

# Generate Sequences
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
  # Scaled position at t0 (last history step) -- the model
  # incremental predicts displacements relative to THIS row, same as in
  # dataset_pred_filt.py/train_pred_incr.py.
  Y_t0_list.append(Y_scaled[i + T_IN - 1])

X_test = torch.tensor(np.array(X_hist_list), dtype=torch.float32)
U_test = torch.tensor(np.array(U_cand_list), dtype=torch.float32)
Y_test_scaled = torch.tensor(np.array(Y_fut_list), dtype=torch.float32)
Y_t0_scaled = np.array(Y_t0_list)  # [N, 3] -- scaled absolute position at t0

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
      output_dim=3,  # 🔴 CHANGE HERE: 3 dimensions by default
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
# 4. INFERENCE AND EVALUATION IN REAL UNITS (MILLIMETERS)
# =====================================================================
with torch.no_grad():
  Y_pred_scaled = model(X_test.to(DEVICE), U_test.to(DEVICE)).cpu().numpy()

Y_true_scaled = Y_test_scaled.numpy()


# Descaling Function -- for an ABSOLUTE POSITION in [-1,1] (takes the
# offset/bias "+1"+min).
def desescalar_var(val_scaled, col_name, dict_params):
  min_t = dict_params[col_name]['min_t']
  max_t = dict_params[col_name]['max_t']
  return min_t + (val_scaled + 1.0) * (max_t - min_t) / 2.0


# Descaling Function -- for a DISPLACEMENT (difference between two
# values already in [-1,1]), WITHOUT the offset/bias: a delta has no "origin"
# own (see same formula in train_pred_incr.py/PINNLossMPC).
def desescalar_delta_var(delta_scaled, col_name, dict_params):
  min_t = dict_params[col_name]['min_t']
  max_t = dict_params[col_name]['max_t']
  return delta_scaled * (max_t - min_t) / 2.0


# This model predicts INCREMENTAL DISPLACEMENTS (ΔX, ΔY, ΔZ) relative to
# t0, not absolute position -- the delta must be descaled and added to the
# REAL position at t0 to reconstruct the comparable absolute trajectory
# against Y_true (which is still absolute position, exactly as it comes out of the
# dataset).
X_t0_mm = desescalar_var(Y_t0_scaled[:, 0], 'rel_x', Y_TRANS) * 1000.0  # [N]
Y_t0_mm = desescalar_var(Y_t0_scaled[:, 1], 'rel_y', Y_TRANS) * 1000.0
Z_t0_mm = desescalar_var(Y_t0_scaled[:, 2], 'rel_z', Y_TRANS) * 1000.0

dX_pred_mm = desescalar_delta_var(Y_pred_scaled[:, :, 0], 'rel_x', Y_TRANS) * 1000.0  # [N, T_OUT]
dY_pred_mm = desescalar_delta_var(Y_pred_scaled[:, :, 1], 'rel_y', Y_TRANS) * 1000.0
dZ_pred_mm = desescalar_delta_var(Y_pred_scaled[:, :, 2], 'rel_z', Y_TRANS) * 1000.0

# Convert to Meters -> Millimeters (reconstructed ABSOLUTE position = t0 + delta)
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
# 4.5. ANGULAR ERROR BETWEEN THE PREDICTED AND THE REAL DISPLACEMENT VECTOR
# =====================================================================
# For each consecutive transition of the horizon (t+k -> t+k+1), compares the
# DIRECTION of the predicted 3D displacement against the real one, via the angle
# between both vectors (0° = same direction, 180° = opposite). It is the
# counterpart, in degrees and as an evaluation metric (not a training one),
# of the cosine similarity used by loss_direction_xz in train_pred.py -- but
# here in full 3D, not just X-Z, to get the complete picture.
pos_pred_mm = np.stack([X_pred_mm, Y_pred_mm, Z_pred_mm], axis=-1)  # [N, T_OUT, 3]
pos_true_mm = np.stack([X_true_mm, Y_true_mm, Z_true_mm], axis=-1)  # [N, T_OUT, 3]

dir_pred = pos_pred_mm[:, 1:, :] - pos_pred_mm[:, :-1, :]  # [N, T_OUT-1, 3]
dir_true = pos_true_mm[:, 1:, :] - pos_true_mm[:, :-1, :]  # [N, T_OUT-1, 3]

norm_pred = np.linalg.norm(dir_pred, axis=-1)
norm_true = np.linalg.norm(dir_true, axis=-1)
EPS_MAGNITUD_MM = 0.5  # transitions where the real displacement is < 0.5mm
# do not have a well-defined direction (noise) -- they are excluded from the average.
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

# Filter only transitions with displacement above the measurement noise.
# 3mm per step (16.7ms) would imply ~180mm/s, faster than any segment of
# this test CSV (maximum observed real displacement: 1.84mm/step, mean
# 0.63mm) -- with that threshold no sample remains. 1.0mm is still well
# above the typical OptiTrack noise floor (sub-millimeter) and leaves
# ~17% of the transitions (those with the clearest movement) to average.
UMBRAL_DESPLAZAMIENTO_MM = 1.0

# pos_t1/pos_t2 = REAL position at each consecutive step of the horizon
# (pos_true_mm[:, 1:, :] - pos_true_mm[:, :-1, :] is exactly dir_true,
# already in mm, so displacement == norm_true computed above).
pos_t1 = pos_true_mm[:, :-1, :]
pos_t2 = pos_true_mm[:, 1:, :]
desplazamiento = np.linalg.norm(pos_t2 - pos_t1, axis=-1)  # already in mm

mask_movimiento_real = desplazamiento > UMBRAL_DESPLAZAMIENTO_MM

angulo_filtrado = angulo_deg[mask_movimiento_real]
print(f'Ángulo medio (solo desplazamientos > {UMBRAL_DESPLAZAMIENTO_MM}mm): '
      f'{angulo_filtrado.mean():.2f}° ({mask_movimiento_real.sum()} muestras)')

# =====================================================================
# 5. VISUALIZATION: ROLLOUT EVERY 10 WINDOWS OVER THE REAL TRAJECTORY
# =====================================================================
PASO_MUESTRAS = 10
N_MAX = 4000  # Limit to the first 1000 samples

# Truncate the real data to the first N_MAX points
real_continuo = np.stack([X_true_mm[:N_MAX, 0], Y_true_mm[:N_MAX, 0], Z_true_mm[:N_MAX, 0]], axis=1)
tiempo_continuo = np.arange(len(real_continuo))

# Ensure the prediction anchors do not exceed the 1000-sample range
limite_anclas = min(len(X_test), N_MAX) - T_OUT
anclas = list(range(0, max(0, limite_anclas), PASO_MUESTRAS))
cmap = plt.get_cmap('plasma')

fig2 = plt.figure(figsize=(16, 8))

# 3D Subplot
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

# 1D Subplots per axis
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