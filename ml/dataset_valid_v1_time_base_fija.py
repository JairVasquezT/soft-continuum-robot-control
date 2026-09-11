import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation as R

# =====================================================================
# 1. PANEL DE CONTROL
# =====================================================================
VERSION_MODELO = "v1"       # Opciones: "v1", "v2", "v3", "v4"
WINDOW_SIZE = 120            # Ventana temporal
BATCH_SIZE_EVAL = 512       # 💡 Lote de evaluación para evitar el CUDA OOM

PATH_CSV = "corto_20260727_174651_814.csv"
PATH_PESOS = "soft_robot_lstm_v1_8_time.pth" 
PATH_JSON = "dataset_v1_sinMeta_sinRot_params.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Procesando CSV con dispositivo: {device} | Modo: {VERSION_MODELO.upper()}")

config_arquitectura = {
    "v1": {"input_size": 5, "output_size": 3, "y_keys": ['rel_x', 'rel_y', 'rel_z']},
    "v2": {"input_size": 9, "output_size": 3, "y_keys": ['rel_x', 'rel_y', 'rel_z']},
    "v3": {"input_size": 5, "output_size": 7, "y_keys": ['rel_x', 'rel_y', 'rel_z', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']},
    "v4": {"input_size": 9, "output_size": 7, "y_keys": ['rel_x', 'rel_y', 'rel_z', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_qw']}
}
cfg = config_arquitectura[VERSION_MODELO]

# =====================================================================
# 2. CARGA DE PARÁMETROS MAESTROS DESDE EL JSON
# =====================================================================
with open(PATH_JSON, 'r') as f:
    norm_params = json.load(f)

home = norm_params["home_motores"]

# =====================================================================
# 3. PROCESAMIENTO MECÁNICO Y GEOMÉTRICO
# =====================================================================
print(f"📖 Leyendo archivo de telemetría original: {PATH_CSV}")

columnas_validas = [
    'timestamp', 't_unix', 't_relativo', 'combo_id', 'valid_lecture',
    'meta_m1', 'meta_m2', 'meta_m3', 'meta_m4', 
    'real_m1', 'real_m2', 'real_m3', 'real_m4',
    'base_x', 'base_y', 'base_z', 'base_qx', 'base_qy', 'base_qz', 'base_qw',
    'efector_x', 'efector_y', 'efector_z', 'efector_qx', 'efector_qy', 'efector_qz', 'efector_qw'
]

df = pd.read_csv(PATH_CSV, names=columnas_validas, header=0)
df.columns = df.columns.str.strip()

# Diferencial de tiempo dt
df['delta_t'] = df['t_relativo'].astype(float).diff().bfill()
df = df.iloc[1:].reset_index(drop=True)

# Deltas de motores
for i in range(1, 5):
    df[f'delta_real_m{i}'] = df[f'real_m{i}'].astype(float) - home[f'm{i}']
    df[f'delta_meta_m{i}'] = df[f'meta_m{i}'].astype(float) - home[f'm{i}']

# =====================================================================
# 3. TRANSFORMACIÓN CON BASE FIJA (Tomando solo el primer frame)
# =====================================================================
# A. Tomar la posición y cuaternión de la base ÚNICAMENTE del primer frame
p_base_fija = df[['base_x', 'base_y', 'base_z']].iloc[0].astype(float).values
q_base_fijo = df[['base_qx', 'base_qy', 'base_qz', 'base_qw']].iloc[0].astype(float).values

# B. Posición del efector variable en el tiempo
p_efector = df[['efector_x', 'efector_y', 'efector_z']].astype(float).values

# C. Crear la rotación fija de la base y las rotaciones dinámicas del efector
r_base_fija = R.from_quat(q_base_fijo)
r_efector = R.from_quat(df[['efector_qx', 'efector_qy', 'efector_qz', 'efector_qw']].astype(float).values)

# D. Transformaciones geométricas usando la base fija
# P_rel = (R_base_fija)^(-1) * (P_efector - P_base_fija)
p_relativo = r_base_fija.inv().apply(p_efector - p_base_fija)

# R_rel = (R_base_fija)^(-1) * R_efector
r_relativo = r_base_fija.inv() * r_efector
q_relativo = r_relativo.as_quat()

# E. Asignar de nuevo al DataFrame
df['rel_x'], df['rel_y'], df['rel_z'] = p_relativo[:, 0], p_relativo[:, 1], p_relativo[:, 2]
df['rel_qx'], df['rel_qy'], df['rel_qz'], df['rel_qw'] = q_relativo[:, 0], q_relativo[:, 1], q_relativo[:, 2], q_relativo[:, 3]

# =====================================================================
# 4. NORMALIZACIÓN [-1, 1]
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

X_scaled = normalizar_con_limites_json(X_raw_matrix, norm_params['X_transformer'], x_keys)
Y_scaled = normalizar_con_limites_json(Y_raw_matrix, norm_params['Y_transformer'], y_keys)

# =====================================================================
# 5. CREACIÓN DE VENTANAS TEMPORALES
# =====================================================================
X_seq, Y_seq = [], []
for i in range(len(X_scaled) - WINDOW_SIZE):
    X_seq.append(X_scaled[i : i + WINDOW_SIZE])
    Y_seq.append(Y_scaled[i + WINDOW_SIZE])

X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32)

# 💡 USO DE DATALOADER EN CPU/GPU PARA EVITAR EXPLOSIÓN DE MEMORIA
eval_dataset = TensorDataset(X_tensor, Y_tensor)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)

# =====================================================================
# 6. ARQUITECTURA DE LA RED
# =====================================================================
class SoftRobotLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.0):
        super(SoftRobotLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)  # PyTorch inicializa los estados h0, c0 en 0 automáticamente
        last_step = out[:, -1, :]
        out_regularized = self.dropout(last_step)
        return self.fc(out_regularized)

model = SoftRobotLSTM(input_size=cfg['input_size'], output_size=cfg['output_size'], dropout=0.2).to(device)
model.load_state_dict(torch.load(PATH_PESOS, map_location=device))
model.eval()

# =====================================================================
# 7. INFERENCIA EN MINI-BATCHES (PROTEGIDO CONTRA OOM)
# =====================================================================
preds_list = []
real_list = []

with torch.no_grad():
    for x_batch, y_batch in eval_loader:
        x_batch = x_batch.to(device)
        preds_batch = model(x_batch)
        
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

Y_real_phys = desnormalizar_matriz(real_scaled, norm_params['Y_transformer'], y_keys)
Y_pred_phys = desnormalizar_matriz(preds_scaled, norm_params['Y_transformer'], y_keys)

if cfg['output_size'] == 7:
    q_vectors = Y_pred_phys[:, 3:7]
    Y_pred_phys[:, 3:7] = q_vectors / np.linalg.norm(q_vectors, axis=1, keepdims=True)

Y_real_mm = Y_real_phys[:, :3] * 1000.0
Y_pred_mm = Y_pred_phys[:, :3] * 1000.0

# =====================================================================
# 8. CÁLCULO DE MÉTRICAS DE ERROR
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

# =====================================================================
# 9. GRÁFICOS
# =====================================================================
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(Y_real_mm[:, 0], Y_real_mm[:, 1], Y_real_mm[:, 2], color='black', label='Real (OptiTrack)', linewidth=2.5)
ax1.plot(Y_pred_mm[:, 0], Y_pred_mm[:, 1], Y_pred_mm[:, 2], color='crimson', linestyle='--', label='Predicho (LSTM)', linewidth=2)
ax1.set_title(f'Trayectoria en Espacio 3D Real (Modo {VERSION_MODELO.upper()})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Eje X (mm)')
ax1.set_ylabel('Eje Y (mm)')
ax1.set_zlabel('Eje Z (mm)')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(1, 2, 2)
time_axis = np.arange(len(Y_real_mm))
ax2.plot(time_axis, Y_real_mm[:, 0], color='black', label='X Real', linewidth=1.5)
ax2.plot(time_axis, Y_pred_mm[:, 0], color='crimson', linestyle='--', label='X Predicho')
ax2.plot(time_axis, Y_real_mm[:, 1], color='darkblue', alpha=0.4, label='Y Real')
ax2.plot(time_axis, Y_pred_mm[:, 1], color='dodgerblue', linestyle=':', label='Y Predicho')
ax2.plot(time_axis, Y_real_mm[:, 2], color='darkgreen', alpha=0.4, label='Z Real')
ax2.plot(time_axis, Y_pred_mm[:, 2], color='limegreen', linestyle=':', label='Z Predicho')

ax2.set_title('Seguimiento Temporal Desglosado por Eje (mm)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Muestras Temporales')
ax2.set_ylabel('Posición (mm)')
ax2.grid(True)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()