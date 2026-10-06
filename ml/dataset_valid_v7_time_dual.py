import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =====================================================================
# 1. CONTROL PANEL (Configured for V7 Dual-Stream)
# =====================================================================
VERSION_MODELO = "v7_dual"
WINDOW_SIZE = 45  # Window synchronized to 45 samples (~0.75 s at 60 Hz)
BATCH_SIZE_EVAL = 512

PATH_CSV = "corto_20260730_164214_477.csv"
PATH_PESOS = "soft_robot_lstm_v7_5_45win.pth"  # V7 model weights
PATH_JSON = "dataset_v07_real_meta_tension_sinRot_filt_params.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(
    f'🔥 Evaluando CSV con dispositivo: {device} | Modo:'
    f' {VERSION_MODELO.upper()}'
)

cfg = {"input_size": 13, "output_size": 3, "y_keys": ["rel_x", "rel_y", "rel_z"]}

# =====================================================================
# 2. LOADING MASTER PARAMETERS FROM THE V7 JSON
# =====================================================================
with open(PATH_JSON, 'r') as f:
  norm_params = json.load(f)

home = norm_params['home_motores']

# =====================================================================
# 3. MECHANICAL AND GEOMETRIC PROCESSING
# =====================================================================
columnas_validas = [
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

df = pd.read_csv(PATH_CSV, names=columnas_validas, header=0)
df.columns = df.columns.str.strip()

df['delta_t'] = df['t_relativo'].astype(float).diff().bfill()
df = df.iloc[1:].reset_index(drop=True)

for i in range(1, 5):
  df[f'delta_real_m{i}'] = df[f'real_m{i}'].astype(float) - home[f'm{i}']
  df[f'delta_meta_m{i}'] = df[f'meta_m{i}'].astype(float) - home[f'm{i}']
  df[f'couple_m{i}'] = df[f'couple_m{i}'].astype(float)
  df[f'tension_m{i}'] = df[f'tension_m{i}'].astype(float)

p_base = df[['base_x', 'base_y', 'base_z']].astype(float).values
p_efector = df[['efector_x', 'efector_y', 'efector_z']].astype(float).values

r_base = R.from_quat(
    df[['base_qx', 'base_qy', 'base_qz', 'base_qw']].astype(float).values
)
r_efector = R.from_quat(
    df[['efector_qx', 'efector_qy', 'efector_qz', 'efector_qw']]
    .astype(float)
    .values
)

p_relativo = r_base.inv().apply(p_efector - p_base)
r_relativo = r_base.inv() * r_efector
q_relativo = r_relativo.as_quat()

df['rel_x'], df['rel_y'], df['rel_z'] = (
    p_relativo[:, 0],
    p_relativo[:, 1],
    p_relativo[:, 2],
)
df['rel_qx'], df['rel_qy'], df['rel_qz'], df['rel_qw'] = (
    q_relativo[:, 0],
    q_relativo[:, 1],
    q_relativo[:, 2],
    q_relativo[:, 3],
)

# =====================================================================
# 4. AUTOMATIC NORMALIZATION (13 V7 FEATURES)
# =====================================================================
x_keys = list(norm_params['X_transformer'].keys())
y_keys = cfg['y_keys']

X_raw_matrix = df[x_keys].astype(float).values
Y_raw_matrix = df[y_keys].astype(float).values


def normalizar_con_limites_json(matrix, params_section, keys_list):
  matrix_scaled = np.zeros_like(matrix, dtype=np.float32)
  for idx, key in enumerate(keys_list):
    min_t = params_section[key]['min_t']
    max_t = params_section[key]['max_t']
    matrix_scaled[:, idx] = 2 * (matrix[:, idx] - min_t) / (max_t - min_t) - 1
  return matrix_scaled


X_scaled = normalizar_con_limites_json(
    X_raw_matrix, norm_params['X_transformer'], x_keys
)
Y_scaled = normalizar_con_limites_json(
    Y_raw_matrix, norm_params['Y_transformer'], y_keys
)

# =====================================================================
# 5. TEMPORAL WINDOWING AND DATALOADER
# =====================================================================
X_seq, Y_seq = [], []
for i in range(len(X_scaled) - WINDOW_SIZE):
  X_seq.append(X_scaled[i : i + WINDOW_SIZE])
  Y_seq.append(Y_scaled[i + WINDOW_SIZE])

X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32)

eval_dataset = TensorDataset(X_tensor, Y_tensor)
eval_loader = DataLoader(
    eval_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False
)


# =====================================================================
# 6. DUAL-STREAM LSTM ARCHITECTURE (V7)
# =====================================================================
class V7DualStreamLSTM(nn.Module):

  def __init__(
      self,
      kin_input_size=9,
      tens_input_size=4,
      kin_hidden=96,
      tens_hidden=48,
      output_size=3,
      dropout=0.2,
  ):
    super(V7DualStreamLSTM, self).__init__()

    # Kinematic Branch: 9 inputs (delta_real_m1..4, delta_t, delta_meta_m1..4)
    self.lstm_kin = nn.LSTM(
        kin_input_size, kin_hidden, num_layers=2, batch_first=True
    )

    # Dynamic Branch: 4 inputs (tension_m1..4)
    self.lstm_tens = nn.LSTM(
        tens_input_size, tens_hidden, num_layers=1, batch_first=True
    )

    self.dropout = nn.Dropout(dropout)

    # Fusion Layer
    self.fc = nn.Sequential(
        nn.Linear(kin_hidden + tens_hidden, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, output_size),
    )

  def forward(self, x_kin, x_tens):
    out_kin, _ = self.lstm_kin(x_kin)
    out_tens, _ = self.lstm_tens(x_tens)

    last_kin = out_kin[:, -1, :]
    last_tens = out_tens[:, -1, :]

    fusion = torch.cat((last_kin, last_tens), dim=1)
    return self.fc(fusion)


# Instantiation and weight loading
model = V7DualStreamLSTM(
    kin_input_size=9,
    tens_input_size=4,
    kin_hidden=96,
    tens_hidden=48,
    output_size=3,
    dropout=0.2,
).to(device)

model.load_state_dict(torch.load(PATH_PESOS, map_location=device))
model.eval()

# =====================================================================
# 7. BATCH INFERENCE WITH V7 SEPARATION (9 KINEMATIC + 4 TENSIONS)
# =====================================================================
preds_list = []
real_list = []

with torch.no_grad():
  for x_batch, y_batch in eval_loader:
    x_batch = x_batch.to(device)

    # Feature separation for V7
    x_kin = x_batch[:, :, :9]  # Kinematics (0..8)
    x_tens = x_batch[:, :, 9:]  # Tensions (9..12)

    preds_batch = model(x_kin, x_tens)

    preds_list.append(preds_batch.cpu().numpy())
    real_list.append(y_batch.numpy())

preds_scaled = np.vstack(preds_list)
real_scaled = np.vstack(real_list)


def desnormalizar_matriz(matrix_sc, params_section, keys_list):
  matrix_phys = np.zeros_like(matrix_sc)
  for idx, key in enumerate(keys_list):
    min_t = params_section[key]['min_t']
    max_t = params_section[key]['max_t']
    matrix_phys[:, idx] = min_t + ((matrix_sc[:, idx] + 1) / 2) * (max_t - min_t)
  return matrix_phys


Y_real_phys = desnormalizar_matriz(
    real_scaled, norm_params['Y_transformer'], y_keys
)
Y_pred_phys = desnormalizar_matriz(
    preds_scaled, norm_params['Y_transformer'], y_keys
)

# Conversion to millimeters
Y_real_mm = Y_real_phys[:, :3] * 1000.0
Y_pred_mm = Y_pred_phys[:, :3] * 1000.0

# =====================================================================
# 8. COMPUTATION OF ERROR METRICS
# =====================================================================
mae_ejes = np.mean(np.abs(Y_real_mm - Y_pred_mm), axis=0)
error_euclidiano_3d = np.sqrt(np.sum((Y_real_mm - Y_pred_mm) ** 2, axis=1))
mae_3d_promedio = np.mean(error_euclidiano_3d)

print('\n=======================================================')
print(
    f'📊 RESULTADOS DE VALIDACIÓN EN DATASET OCULTO ({VERSION_MODELO.upper()})'
)
print('=======================================================')
print(f'Error MAE Eje X: {mae_ejes[0]:.3f} mm')
print(f'Error MAE Eje Y: {mae_ejes[1]:.3f} mm')
print(f'Error MAE Eje Z: {mae_ejes[2]:.3f} mm')
print(f'📐 ERROR DE DISTANCIA EUCLÍDEA 3D PROMEDIO: {mae_3d_promedio:.3f} mm')
print('=======================================================\n')

# =====================================================================
# 9. COMPARATIVE GRAPHICAL VISUALIZATION
# =====================================================================
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(
    Y_real_mm[:, 0],
    Y_real_mm[:, 1],
    Y_real_mm[:, 2],
    color='black',
    label='Real (OptiTrack)',
    linewidth=2.5,
)
ax1.plot(
    Y_pred_mm[:, 0],
    Y_pred_mm[:, 1],
    Y_pred_mm[:, 2],
    color='crimson',
    linestyle='--',
    label='Predicho (V7 Dual-Stream)',
    linewidth=2,
)
ax1.set_title(
    f'Trayectoria 3D Real vs Predicción ({VERSION_MODELO.upper()})',
    fontsize=12,
    fontweight='bold',
)
ax1.set_xlabel('Eje X (mm)')
ax1.set_ylabel('Eje Y (mm)')
ax1.set_zlabel('Eje Z (mm)')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(1, 2, 2)
time_axis = np.arange(len(Y_real_mm))
ax2.plot(
    time_axis, Y_real_mm[:, 0], color='black', label='X Real', linewidth=1.5
)
ax2.plot(
    time_axis,
    Y_pred_mm[:, 0],
    color='crimson',
    linestyle='--',
    label='X Predicho',
)
ax2.plot(
    time_axis,
    Y_real_mm[:, 1],
    color='darkblue',
    alpha=0.4,
    label='Y Real',
)
ax2.plot(
    time_axis,
    Y_pred_mm[:, 1],
    color='dodgerblue',
    linestyle=':',
    label='Y Predicho',
)
ax2.plot(
    time_axis,
    Y_real_mm[:, 2],
    color='darkgreen',
    alpha=0.4,
    label='Z Real',
)
ax2.plot(
    time_axis,
    Y_pred_mm[:, 2],
    color='limegreen',
    linestyle=':',
    label='Z Predicho',
)

ax2.set_title(
    'Seguimiento Temporal Desglosado por Eje (mm)',
    fontsize=12,
    fontweight='bold',
)
ax2.set_xlabel('Muestras Temporales')
ax2.set_ylabel('Posición (mm)')
ax2.grid(True)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()