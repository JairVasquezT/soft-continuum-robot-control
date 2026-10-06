import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation as R

# =====================================================================
# 1. CONTROL PANEL (Configure the variant to evaluate here)
# =====================================================================
VERSION_MODELO = "v2"       # Options: "v1", "v2", "v3", "v4"
WINDOW_SIZE = 20            # Temporal window optimized at 12.5 Hz

PATH_CSV = "corto_20260710_140945_078.csv"
PATH_PESOS = "soft_robot_lstm_v2_2.pth"  # Or 'soft_robot_lstm_v2.pth'
PATH_JSON = "dataset_v2_conMeta_sinRot_params.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Procesando CSV con dispositivo: {device} | Modo: {VERSION_MODELO.upper()}")

# Architecture control dictionary following your paper taxonomy
config_arquitectura = {
    "v1": {"input_size": 4, "output_size": 3, "y_keys": ['rel_x', 'rel_y', 'rel_z']},
    "v2": {"input_size": 8, "output_size": 3, "y_keys": ['rel_x', 'rel_y', 'rel_z']},
    "v3": {"input_size": 4, "output_size": 7, "y_keys": ['rel_x', 'rel_y', 'rel_z', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']},
    "v4": {"input_size": 8, "output_size": 7, "y_keys": ['rel_x', 'rel_y', 'rel_z', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']}
}
cfg = config_arquitectura[VERSION_MODELO]

# =====================================================================
# 2. LOADING MASTER PARAMETERS FROM THE JSON
# =====================================================================
with open(PATH_JSON, 'r') as f:
    norm_params = json.load(f)

# Extract the robot's physical "zero" saved in the training JSON
home = norm_params["home_motores"]

# =====================================================================
# 3. RAW MECHANICAL AND GEOMETRIC PROCESSING OF THE CSV
# =====================================================================
print(f"📖 Leyendo archivo de telemetría original: {PATH_CSV}")

columnas_validas = [
    'timestamp', 't_unix', 't_relativo', 'combo_id',
    'meta_m1', 'meta_m2', 'meta_m3', 'meta_m4', 
    'real_m1', 'real_m2', 'real_m3', 'real_m4',
    'base_x', 'base_y', 'base_z', 'base_qx', 'base_qy', 'base_qz', 'base_qw',
    'efector_x', 'efector_y', 'efector_z', 'efector_qx', 'efector_qy', 'efector_qz', 'efector_qw'
]

df = pd.read_csv(PATH_CSV, names=columnas_validas, header=0)

df = df.iloc[1:].reset_index(drop=True)

df.columns = df.columns.str.strip()

# A. Compute motor deltas by subtracting their Home calibration
df['delta_real_m1'] = df['real_m1'].astype(float) - home['m1']
df['delta_real_m2'] = df['real_m2'].astype(float) - home['m2']
df['delta_real_m3'] = df['real_m3'].astype(float) - home['m3']
df['delta_real_m4'] = df['real_m4'].astype(float) - home['m4']

df['delta_meta_m1'] = df['meta_m1'].astype(float) - home['m1']
df['delta_meta_m2'] = df['meta_m2'].astype(float) - home['m2']
df['delta_meta_m3'] = df['meta_m3'].astype(float) - home['m3']
df['delta_meta_m4'] = df['meta_m4'].astype(float) - home['m4']

# B. Geometric Coordinate Transformation of OptiTrack (Base -> End-effector)
# Convert the positions to numpy arrays
p_base = df[['base_x', 'base_y', 'base_z']].values
p_efector = df[['efector_x', 'efector_y', 'efector_z']].values

# Extract the quaternions and create the three-dimensional rotation objects
r_base = R.from_quat(df[['base_qx', 'base_qy', 'base_qz', 'base_qw']].values)
r_efector = R.from_quat(df[['efector_qx', 'efector_qy', 'efector_qz', 'efector_qw']].values)

# Exact relative position: P_rel = (R_base)^(-1) * (P_efector - P_base)
p_relativo = r_base.inv().apply(p_efector - p_base)

# Exact relative rotation: R_rel = (R_base)^(-1) * R_effector
r_relativo = r_base.inv() * r_efector
q_relativo = r_relativo.as_quat() # Returns a matrix with [qx, qy, qz, qw]

# Save the computations back into the intermediate DataFrame
df['rel_x'], df['rel_y'], df['rel_z'] = p_relativo[:, 0], p_relativo[:, 1], p_relativo[:, 2]
df['rel_qx'], df['rel_qy'], df['rel_qz'], df['rel_qw'] = q_relativo[:, 0], q_relativo[:, 1], q_relativo[:, 2], q_relativo[:, 3]

# =====================================================================
# 4. ISOLATION AND HISTORICAL SCALING [-1, 1]
# =====================================================================
x_keys = list(norm_params['X_transformer'].keys())
y_keys = cfg['y_keys']

X_raw_matrix = df[x_keys].values
Y_raw_matrix = df[y_keys].values

def normalizar_con_limites_json(matrix, params_section, keys_list):
    matrix_scaled = np.zeros_like(matrix, dtype=np.float32)
    for idx, key in enumerate(keys_list):
        min_t = params_section[key]['min_t']
        max_t = params_section[key]['max_t']
        # Exact fit to the symmetric range [-1, 1]
        matrix_scaled[:, idx] = 2 * (matrix_matrix := matrix[:, idx] - min_t) / (max_t - min_t) - 1
    return matrix_scaled

X_scaled = normalizar_con_limites_json(X_raw_matrix, norm_params['X_transformer'], x_keys)
Y_scaled = normalizar_con_limites_json(Y_raw_matrix, norm_params['Y_transformer'], y_keys)

# =====================================================================
# 5. CREATION OF CONTINUOUS TEMPORAL WINDOWS
# =====================================================================
X_seq, Y_seq = [], []
for i in range(len(X_scaled) - WINDOW_SIZE):
    X_seq.append(X_scaled[i : i + WINDOW_SIZE])
    Y_seq.append(Y_scaled[i + WINDOW_SIZE])

X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)
Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32).to(device)

# =====================================================================
# 6. LSTM ARCHITECTURE DEFINITION AND WEIGHT LOADING
# =====================================================================
class SoftRobotLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.0):
        super(SoftRobotLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        last_step = out[:, -1, :]
        out_regularized = self.dropout(last_step)
        return self.fc(out_regularized)

model = SoftRobotLSTM(input_size=cfg['input_size'], output_size=cfg['output_size'], dropout=0.2).to(device)
model.load_state_dict(torch.load(PATH_PESOS, map_location=device))
model.eval()

# =====================================================================
# 7. INFERENCE AND MATHEMATICAL DENORMALIZATION
# =====================================================================
with torch.no_grad():
    preds_scaled = model(X_tensor).cpu().numpy()
real_scaled = Y_tensor.cpu().numpy()

def desnormalizar_matriz(matrix_sc, params_section, keys_list):
    matrix_phys = np.zeros_like(matrix_sc)
    for idx, key in enumerate(keys_list):
        min_t = params_section[key]['min_t']
        max_t = params_section[key]['max_t']
        matrix_phys[:, idx] = min_t + ((matrix_sc[:, idx] + 1) / 2) * (max_t - min_t)
    return matrix_phys

Y_real_phys = desnormalizar_matriz(real_scaled, norm_params['Y_transformer'], y_keys)
Y_pred_phys = desnormalizar_matriz(preds_scaled, norm_params['Y_transformer'], y_keys)

# Unit algebraic filter for quaternions if you evaluate V3 or V4
if cfg['output_size'] == 7:
    q_vectors = Y_pred_phys[:, 3:7]
    Y_pred_phys[:, 3:7] = q_vectors / np.linalg.norm(q_vectors, axis=1, keepdims=True)

# Metric Cartesian conversion: we go from Meters to Millimeters for engineering analysis
Y_real_mm = Y_real_phys[:, :3] * 1000.0
Y_pred_mm = Y_pred_phys[:, :3] * 1000.0

# =====================================================================
# 8. COMPUTATION OF SCIENTIFIC ERROR METRICS
# =====================================================================
mae_ejes = np.mean(np.abs(Y_real_mm - Y_pred_mm), axis=0)
error_euclidiano_3d = np.sqrt(np.sum((Y_real_mm - Y_pred_mm)**2, axis=1))
mae_3d_promedio = np.mean(error_euclidiano_3d)

print("\n=======================================================")
print(f"📊 RESULTADOS DE VALIDACIÓN EN DATASET OCULTO ({VERSION_MODELO.upper()})")
print("=======================================================")
print(f"Error MAE Eje X: {mae_ejes[0]:.3f} mm")
print(f"Error MAE Eje Y: {mae_ejes[1]:.3f} mm")
print(f"Error MAE Eje Z: {mae_ejes[2]:.3f} mm")
print(f"📐 ERROR DE DISTANCIA EUCLÍDEA 3D PROMEDIO: {mae_3d_promedio:.3f} mm")
print("=======================================================\n")

'''
# =====================================================================
# 9. GENERACIÓN DE GRÁFICOS COMPUESTOS EXPERIMENTALES
# =====================================================================
fig = plt.figure(figsize=(14, 6))

# Gráfico Izquierdo: Reconstrucción espacial 3D del espacio de trabajo
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(Y_real_mm[:, 0], Y_real_mm[:, 1], Y_real_mm[:, 2], color='black', label='Real (OptiTrack)', linewidth=2.5)
ax1.plot(Y_pred_mm[:, 0], Y_pred_mm[:, 1], Y_pred_mm[:, 2], color='crimson', linestyle='--', label='Predicho (LSTM)', linewidth=2)
ax1.set_title(f'Trayectoria en Espacio 3D Real (Modo {VERSION_MODELO.upper()})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Eje X (mm)')
ax1.set_ylabel('Eje Y (mm)')
ax1.set_zlabel('Eje Z (mm)')
ax1.legend()
ax1.grid(True)

# Gráfico Derecho: Desglose por componente a lo largo del tiempo
ax2 = fig.add_subplot(1, 2, 2)
time_axis = np.arange(len(Y_real_mm))
ax2.plot(time_axis, Y_real_mm[:, 0], color='black', label='X Real', linewidth=1.5)
ax2.plot(time_axis, Y_pred_mm[:, 0], color='crimson', linestyle='--', label='X Predicho')
ax2.plot(time_axis, Y_real_mm[:, 1], color='darkblue', alpha=0.4, label='Y Real')
ax2.plot(time_axis, Y_pred_mm[:, 1], color='dodgerblue', linestyle=':', label='Y Predicho')
ax2.plot(time_axis, Y_real_mm[:, 2], color='darkgreen', alpha=0.4, label='Z Real')
ax2.plot(time_axis, Y_pred_mm[:, 2], color='limegreen', linestyle=':', label='Z Predicho')

ax2.set_title('Seguimiento Temporal Desglosado por Eje (mm)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Muestras / Secuencias Temporales')
ax2.set_ylabel('Posición (mm)')
ax2.grid(True)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

'''

# =====================================================================
# 9. GENERATION OF COMPOSITE EXPERIMENTAL PLOTS (CROPPED WINDOW)
# =====================================================================
# Define the temporal window of interest
INICIO_FRAME = 1100
FIN_FRAME = 1250

# Make sure we do not exceed the actual bounds of the matrix
FIN_FRAME = min(FIN_FRAME, len(Y_real_mm))

# Extract the data subsets to plot only that window
y_real_ventana = Y_real_mm[INICIO_FRAME:FIN_FRAME]
y_pred_ventana = Y_pred_mm[INICIO_FRAME:FIN_FRAME]

fig = plt.figure(figsize=(14, 6))

# Left Plot: 3D spatial reconstruction of the selected window
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(y_real_ventana[:, 0], y_real_ventana[:, 1], y_real_ventana[:, 2], color='black', label='Real (OptiTrack)', linewidth=2.5)
ax1.plot(y_pred_ventana[:, 0], y_pred_ventana[:, 1], y_pred_ventana[:, 2], color='crimson', linestyle='--', label='Predicho (LSTM)', linewidth=2)
ax1.set_title(f'Trayectoria 3D (Frames {INICIO_FRAME} a {FIN_FRAME})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Eje X (mm)')
ax1.set_ylabel('Eje Y (mm)')
ax1.set_zlabel('Eje Z (mm)')
ax1.legend()
ax1.grid(True)

# Right Plot: Per-component breakdown over time within the window
ax2 = fig.add_subplot(1, 2, 2)
# The time axis will reflect the real frame number of the original experiment
time_axis = np.arange(INICIO_FRAME, FIN_FRAME)

ax2.plot(time_axis, y_real_ventana[:, 0], color='black', label='X Real', linewidth=1.5)
ax2.plot(time_axis, y_pred_ventana[:, 0], color='crimson', linestyle='--', label='X Predicho')
ax2.plot(time_axis, y_real_ventana[:, 1], color='darkblue', alpha=0.4, label='Y Real')
ax2.plot(time_axis, y_pred_ventana[:, 1], color='dodgerblue', linestyle=':', label='Y Predicho')
ax2.plot(time_axis, y_real_ventana[:, 2], color='darkgreen', alpha=0.4, label='Z Real')
ax2.plot(time_axis, y_pred_ventana[:, 2], color='limegreen', linestyle=':', label='Z Predicho')

ax2.set_title(f'Seguimiento Temporal Desglosado (Frames {INICIO_FRAME} a {FIN_FRAME})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Número de Frame Original')
ax2.set_ylabel('Posición (mm)')
ax2.grid(True)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()