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
PATH_METADATA = 'dataset_pred_v07_completo_10_sinRot_directo_filt_params.json'
PATH_MODEL_WEIGHTS = 'best_mpc_pinn_predictor_3.pth'

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
LIMITES_TENSION = (100.0, 2500.0)

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
# NOTE: the whole window is NO longer discarded if the command changes within
# of the future horizon (before: 'if not np.all(u_futuros == u_futuros[0]):
# continue') -- that discarded too much in segments with command changes
# frequent (e.g. logs_trayectoria_final.csv, where each candidate of
# exploration lasts only 0.25s = 15 samples, very close to T_OUT). Instead of
# that, horizonte_comando is saved: how many consecutive future steps
# from t+1 keep the same command -- same as horizonte_valido in
# dataset_pred_filt.py, but only with the command condition (there is no
# valid_lecture/delta_t per step here). It is used in section 4.5 so as not to
# compute the direction error beyond the point where the command changed.
X_hist_list, U_cand_list, Y_fut_list, Horizonte_comando_list = [], [], [], []
for i in range(len(df) - T_IN - T_OUT):
  window_delta_t = df['delta_t'].iloc[i : i + T_IN + T_OUT].values
  if np.any(window_delta_t > MAX_DELTA_T):
    continue

  u_futuros = U_scaled[i + T_IN : i + T_IN + T_OUT]
  horizonte_comando = 0
  for t in range(T_OUT):
    if np.allclose(u_futuros[t], u_futuros[0], atol=1e-6):
      horizonte_comando += 1
    else:
      break

  X_hist_list.append(X_scaled[i : i + T_IN])
  U_cand_list.append(U_scaled[i + T_IN])
  Y_fut_list.append(Y_scaled[i + T_IN : i + T_IN + T_OUT])
  Horizonte_comando_list.append(horizonte_comando)

X_test = torch.tensor(np.array(X_hist_list), dtype=torch.float32)
U_test = torch.tensor(np.array(U_cand_list), dtype=torch.float32)
Y_test_scaled = torch.tensor(np.array(Y_fut_list), dtype=torch.float32)
Horizonte_comando = np.array(Horizonte_comando_list, dtype=np.int32)  # [N]

print(
    f'✓ Dataset de Prueba Cargado: {len(X_test)} muestras válidas generadas '
    f'(horizonte de comando constante: media={Horizonte_comando.mean():.2f}/{T_OUT}).\n'
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
# In batches -- in a single pass, with thousands of samples (more now that
# the entire window is not discarded due to a command change) the LSTM is left
# without GPU memory.
BATCH_SIZE_INFERENCIA = 2048
Y_pred_chunks = []
with torch.no_grad():
  for ini in range(0, len(X_test), BATCH_SIZE_INFERENCIA):
    fin = ini + BATCH_SIZE_INFERENCIA
    pred_chunk = model(X_test[ini:fin].to(DEVICE), U_test[ini:fin].to(DEVICE))
    Y_pred_chunks.append(pred_chunk.cpu().numpy())
Y_pred_scaled = np.concatenate(Y_pred_chunks, axis=0)

Y_true_scaled = Y_test_scaled.numpy()


# Descaling Function
def desescalar_var(val_scaled, col_name, dict_params):
  min_t = dict_params[col_name]['min_t']
  max_t = dict_params[col_name]['max_t']
  return min_t + (val_scaled + 1.0) * (max_t - min_t) / 2.0


# Convert to Meters -> Millimeters
X_pred_mm = desescalar_var(Y_pred_scaled[:, :, 0], 'rel_x', Y_TRANS) * 1000.0
Y_pred_mm = desescalar_var(Y_pred_scaled[:, :, 1], 'rel_y', Y_TRANS) * 1000.0
Z_pred_mm = desescalar_var(Y_pred_scaled[:, :, 2], 'rel_z', Y_TRANS) * 1000.0

X_true_mm = desescalar_var(Y_true_scaled[:, :, 0], 'rel_x', Y_TRANS) * 1000.0
Y_true_mm = desescalar_var(Y_true_scaled[:, :, 1], 'rel_y', Y_TRANS) * 1000.0
Z_true_mm = desescalar_var(Y_true_scaled[:, :, 2], 'rel_z', Y_TRANS) * 1000.0

# Command-horizon mask: step j (0-indexed) only counts for
# the position error if the command stayed constant up to that point (same
# criterion as section 4.5) -- e.g. if you commanded +30 on all 4 motors and
# held for all 10 steps, all 10 are used; if from step 8 onward you commanded
# -30, only the first 7 steps are used (Horizonte_comando=7) for the
# position error of that sample.
mask_paso_comando = np.arange(T_OUT)[None, :] < Horizonte_comando[:, None]  # [N, T_OUT]
n_pasos_validos_pos = mask_paso_comando.sum()

mae_x = np.abs(X_pred_mm - X_true_mm)[mask_paso_comando].mean()
mae_y = np.abs(Y_pred_mm - Y_true_mm)[mask_paso_comando].mean()
mae_z = np.abs(Z_pred_mm - Z_true_mm)[mask_paso_comando].mean()

dist_3d_mm = np.sqrt(
    (X_pred_mm - X_true_mm) ** 2
    + (Y_pred_mm - Y_true_mm) ** 2
    + (Z_pred_mm - Z_true_mm) ** 2
)
mae_3d_global = dist_3d_mm[mask_paso_comando].mean()

print('====================================================')
print('📊 RESULTADOS DE EVALUACIÓN DEL PREDICTOR EN DATASET CORTO')
print('====================================================')
print(f'(solo pasos con comando constante desde t+1: {n_pasos_validos_pos} de '
      f'{X_pred_mm.size} pasos totales)')
print(f'Error MAE Eje X: {mae_x:.3f} mm')
print(f'Error MAE Eje Y: {mae_y:.3f} mm')
print(f'Error MAE Eje Z: {mae_z:.3f} mm')
print(f'📐 ERROR DE DISTANCIA EUCLÍDEA 3D PROMEDIO: {mae_3d_global:.3f} mm\n')

print('--- ERROR PROMEDIO 3D PASO A PASO EN EL HORIZONTE FUTURO (solo comando constante) ---')
for paso in range(T_OUT):
  mask_col = mask_paso_comando[:, paso]
  n_validas_paso = mask_col.sum()
  tiempo_futuro_ms = (paso + 1) * (1000.0 / 60.0)
  if n_validas_paso == 0:
    print(f'Paso t+{paso+1:02d} ({tiempo_futuro_ms:.1f} ms): sin muestras con comando '
          f'constante hasta acá')
    continue
  err_paso = np.mean(dist_3d_mm[mask_col, paso])
  print(
      f'Paso t+{paso+1:02d} ({tiempo_futuro_ms:.1f} ms): Error 3D ='
      f' {err_paso:.3f} mm ({n_validas_paso}/{len(mask_col)} muestras)'
  )

# =====================================================================
# 4.5. ANGULAR ERROR BETWEEN THE PREDICTED AND THE REAL DISPLACEMENT VECTOR
# =====================================================================
# Compares the DIRECTION of the predicted 3D displacement against the real one between
# points separated by SALTO_ANGULO steps (e.g. t+1 vs t+5 with jump=4),
# NOT consecutive transitions (t+k -> t+k+1): one step apart the
# real displacement is so small that the direction is pure noise/false
# errors -- with more separation the vector is longer and the direction more
# stable/significant (same criterion as salto_direccion in
# train_pred.py). The angle is the angle between both vectors (0° = same
# direction, 180° = opposite); it is the counterpart, in degrees and as a metric
# of evaluation (not training), of the cosine similarity used by
# loss_direction_3d in train_pred.py -- in full 3D, not just X-Z.
#
# Only computed for transitions (k, k+SALTO_ANGULO) where the command
# stayed constant over that WHOLE segment -- if the command changed before the
# step k+SALTO_ANGULO, the model never saw that change (u_cand is fixed for
# whole prediction) and comparing its direction there would be unfair/meaningless
# sense. mask_horizonte_dir[n, k] = True if step k+SALTO_ANGULO (the
# farthest end of the transition) is still within the segment of
# constant command of sample n (Horizonte_comando[n]).
SALTO_ANGULO = 4

pos_pred_mm = np.stack([X_pred_mm, Y_pred_mm, Z_pred_mm], axis=-1)  # [N, T_OUT, 3]
pos_true_mm = np.stack([X_true_mm, Y_true_mm, Z_true_mm], axis=-1)  # [N, T_OUT, 3]

dir_pred = pos_pred_mm[:, SALTO_ANGULO:, :] - pos_pred_mm[:, :-SALTO_ANGULO, :]  # [N, T_OUT-SALTO_ANGULO, 3]
dir_true = pos_true_mm[:, SALTO_ANGULO:, :] - pos_true_mm[:, :-SALTO_ANGULO, :]  # [N, T_OUT-SALTO_ANGULO, 3]

norm_pred = np.linalg.norm(dir_pred, axis=-1)
norm_true = np.linalg.norm(dir_true, axis=-1)
EPS_MAGNITUD_MM = 0.5  # transitions where the real displacement is < 0.5mm
# do not have a well-defined direction (noise) -- they are excluded from the average.
pasos_extremo = np.arange(SALTO_ANGULO, T_OUT)  # k+SALTO_ANGULO for each column
mask_horizonte_dir = pasos_extremo[None, :] < Horizonte_comando[:, None]  # [N, T_OUT-SALTO_ANGULO]
validas = (norm_pred > EPS_MAGNITUD_MM) & (norm_true > EPS_MAGNITUD_MM) & mask_horizonte_dir

cos_theta = np.sum(dir_pred * dir_true, axis=-1) / (norm_pred * norm_true + 1e-9)
cos_theta = np.clip(cos_theta, -1.0, 1.0)
angulo_deg = np.degrees(np.arccos(cos_theta))

print(f'\n--- ERROR ANGULAR (3D) ENTRE DESPLAZAMIENTO PREDICHO Y REAL, SALTO={SALTO_ANGULO} PASOS ---')
for k in range(T_OUT - SALTO_ANGULO):
  mask_k = validas[:, k]
  n_validas = mask_k.sum()
  if n_validas == 0:
    print(f'Transición t+{k+1:02d} -> t+{k+1+SALTO_ANGULO:02d}: sin transiciones válidas '
          f'(desplazamiento real < {EPS_MAGNITUD_MM}mm en todas)')
    continue
  angulo_medio = np.mean(angulo_deg[mask_k, k])
  print(f'Transición t+{k+1:02d} -> t+{k+1+SALTO_ANGULO:02d}: ángulo medio = {angulo_medio:6.2f}° '
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

# pos_t1/pos_t2 = REAL position at each pair separated by SALTO_ANGULO steps
# (pos_true_mm[:, SALTO_ANGULO:, :] - pos_true_mm[:, :-SALTO_ANGULO, :] is
# exactly dir_true, already in mm, so displacement == norm_true
# computed above).
pos_t1 = pos_true_mm[:, :-SALTO_ANGULO, :]
pos_t2 = pos_true_mm[:, SALTO_ANGULO:, :]
desplazamiento = np.linalg.norm(pos_t2 - pos_t1, axis=-1)  # already in mm

mask_movimiento_real = (desplazamiento > UMBRAL_DESPLAZAMIENTO_MM) & mask_horizonte_dir

angulo_filtrado = angulo_deg[mask_movimiento_real]
print(f'Ángulo medio (solo desplazamientos > {UMBRAL_DESPLAZAMIENTO_MM}mm, comando '
      f'constante en el tramo): {angulo_filtrado.mean():.2f}° '
      f'({mask_movimiento_real.sum()} muestras)')

# =====================================================================
# 4.6. DISCRIMINATION DIAGNOSTIC: does the predictor distinguish between
# different candidate commands, or collapsed to predicting almost the same without
# import u_cand? Fixed history window taken from the CSV (rows
# R_INICIO_HIST..t0) -- this model is NOT incremental, it predicts position
# ABSOLUTE directly, so nothing needs to be added to the output.
# =====================================================================
R_INICIO_HIST = 12715
r_home = R_INICIO_HIST + T_IN - 1

print(f'\n📍 Ventana de historial: filas {R_INICIO_HIST}-{r_home} (t0 = fila {r_home})')

historial_fijo = torch.tensor(
    X_scaled[R_INICIO_HIST : R_INICIO_HIST + T_IN], dtype=torch.float32
).unsqueeze(0).to(DEVICE)


def escalar_u_cand(delta_ticks):
  """Scales a candidate command in raw ticks (delta relative to Home)
  to the same [-1, 1] range used during training."""
  radios = np.array([RANGOS_MANUALES[f'm{i}'] for i in range(1, 5)])
  return np.clip(delta_ticks / radios, -1.0, 1.0)


candidatos_ticks = np.array([
    [500, 500, 500, 500],
    [-500, -500, -500, -500],
    [500, -500, 500, -500],
    [0, 0, 0, 0],  # neutral reference candidate
])

print(f'\nComparación de posición final predicha (t+{T_OUT}) para distintos u_cand:\n')
for u_ticks in candidatos_ticks:
  u_scaled = escalar_u_cand(u_ticks)
  u_tensor = torch.tensor(u_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

  with torch.no_grad():
    pred = model(historial_fijo, u_tensor)  # [1, T_OUT, output_dim], scaled absolute position

  pos_final_scaled = pred[0, -1, :3].cpu().numpy()
  x_mm = desescalar_var(pos_final_scaled[0], 'rel_x', Y_TRANS) * 1000.0
  y_mm = desescalar_var(pos_final_scaled[1], 'rel_y', Y_TRANS) * 1000.0
  z_mm = desescalar_var(pos_final_scaled[2], 'rel_z', Y_TRANS) * 1000.0

  print(f'u_cand (ticks)={u_ticks} -> posición final t+{T_OUT} (mm): '
        f'x={x_mm:.1f}, y={y_mm:.1f}, z={z_mm:.1f}')

# =====================================================================
# 5. VISUALIZATION: ROLLOUT EVERY 10 WINDOWS OVER THE REAL TRAJECTORY
# =====================================================================
# Each sample predicts T_OUT steps ahead, not a single point. Here we
# plots a prediction fan every PASO_MUESTRAS windows (instead of
# a large separation to avoid overlap) to closely follow the whole
# movement -- with PASO_MUESTRAS < T_OUT consecutive fans
# overlap on purpose, forming a "band" that shows the consistency of
# the prediction along the whole path.
PASO_MUESTRAS = 30

real_continuo = np.stack([X_true_mm[:, 0], Y_true_mm[:, 0], Z_true_mm[:, 0]], axis=1)
tiempo_continuo = np.arange(len(real_continuo))

anclas = list(range(0, len(X_test) - T_OUT, PASO_MUESTRAS))
cmap = plt.get_cmap('plasma')

fig2 = plt.figure(figsize=(16, 8))

ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
ax1.plot(
    real_continuo[:, 0], real_continuo[:, 1], real_continuo[:, 2],
    color='black', linewidth=1, alpha=0.4, label='Real (continuo)',
)
for j, i in enumerate(anclas):
  color = cmap(j / max(1, len(anclas) - 1))
  origen = real_continuo[i:i + 1]
  rollout = np.stack([X_pred_mm[i, :], Y_pred_mm[i, :], Z_pred_mm[i, :]], axis=1)
  # t0 only as a point (not joined with a line) -- the line connects only
  # t+1..t+T_OUT, so that the rollout trend is visible without the jump
  # initial t0->t+1 dominating the trace.
  ax1.plot(rollout[:, 0], rollout[:, 1], rollout[:, 2], color=color, linewidth=2)
  ax1.scatter(*origen[0], color=color, s=15, marker='o')
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title(f'Real (negro) + {len(anclas)} rollouts de {T_OUT} pasos (color = orden temporal)')
ax1.legend()

ejes = [('X', X_true_mm[:, 0], X_pred_mm), ('Y', Y_true_mm[:, 0], Y_pred_mm),
        ('Z', Z_true_mm[:, 0], Z_pred_mm)]
for k, (nombre_eje, real_eje, pred_eje) in enumerate(ejes):
  ax = fig2.add_subplot(3, 2, 2 * (k + 1))
  ax.plot(tiempo_continuo, real_eje, color='black', linewidth=1, alpha=0.5,
           label='Real' if k == 0 else None)
  for j, i in enumerate(anclas):
    color = cmap(j / max(1, len(anclas) - 1))
    # t0 only as a point; the line connects only t+1..t+T_OUT.
    ax.scatter(i, real_eje[i], color=color, s=12, zorder=3)
    t_rollout = np.arange(i + 1, i + T_OUT + 1)
    ax.plot(t_rollout, pred_eje[i, :], color=color, linewidth=1.5)
  ax.set_ylabel(f'{nombre_eje} (mm)')
  if k == 0:
    ax.set_title(f'Rollouts de {T_OUT} pasos vs. trayectoria real, por eje')
    ax.legend()
  if k == 2:
    ax.set_xlabel('Muestra temporal')

plt.tight_layout()
plt.show()