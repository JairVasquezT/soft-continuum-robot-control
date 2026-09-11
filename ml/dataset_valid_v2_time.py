import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =====================================================================
# 1. PANEL DE CONTROL (Configura aquí la variante a evaluar)
# =====================================================================
VERSION_MODELO = "v2"  # Opciones: "v1", "v2", "v3", "v4"
WINDOW_SIZE = 45  # Ventana temporal optimizada
BATCH_SIZE_EVAL = 512  # [NUEVO] Tamaño del lote para inferencia eficiente

PATH_CSV = "corto_20260730_164214_477.csv"
PATH_PESOS = "soft_robot_lstm_v2_7_time.pth"  # O 'soft_robot_lstm_v2.pth'
PATH_JSON = "dataset_v02_real_meta_sinRot_params.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(
    f"🔥 Procesando CSV con dispositivo: {device} | Modo: {VERSION_MODELO.upper()}"
)

# Diccionario de control de la arquitectura
config_arquitectura = {
    "v1": {
        "input_size": 5,
        "output_size": 3,
        "y_keys": ["rel_x", "rel_y", "rel_z"],
    },
    "v2": {
        "input_size": 9,
        "output_size": 3,
        "y_keys": ["rel_x", "rel_y", "rel_z"],
    },
    "v3": {
        "input_size": 5,
        "output_size": 7,
        "y_keys": [
            "rel_x",
            "rel_y",
            "rel_z",
            "rel_qx",
            "rel_qy",
            "rel_qz",
            "rel_qw",
        ],
    },
    "v4": {
        "input_size": 9,
        "output_size": 7,
        "y_keys": [
            "rel_x",
            "rel_y",
            "rel_z",
            "rel_qx",
            "rel_qy",
            "rel_qz",
            "rel_qw",
        ],
    },
}
cfg = config_arquitectura[VERSION_MODELO]

# =====================================================================
# 2. CARGA DE PARÁMETROS MAESTROS DESDE EL JSON
# =====================================================================
with open(PATH_JSON, "r") as f:
  norm_params = json.load(f)

home = norm_params["home_motores"]

# =====================================================================
# 3. PROCESAMIENTO MECÁNICO Y GEOMÉTRICO EN CRUDO DEL CSV
# =====================================================================
print(f"📖 Leyendo archivo de telemetría original: {PATH_CSV}")

columnas_validas = [
    "timestamp",
    "t_unix",
    "t_relativo",
    "combo_id",
    "valid_lecture",
    "meta_m1",
    "meta_m2",
    "meta_m3",
    "meta_m4",
    "real_m1",
    "real_m2",
    "real_m3",
    "real_m4",
    "couple_m1",
    "couple_m2",
    "couple_m3",
    "couple_m4",
    "tension_m1",
    "tension_m2",
    "tension_m3",
    "tension_m4",
    "base_x",
    "base_y",
    "base_z",
    "base_qx",
    "base_qy",
    "base_qz",
    "base_qw",
    "efector_x",
    "efector_y",
    "efector_z",
    "efector_qx",
    "efector_qy",
    "efector_qz",
    "efector_qw",
]

df = pd.read_csv(PATH_CSV, names=columnas_validas, header=0)
df.columns = df.columns.str.strip()

# Differencial de tiempo (delta_t)
df["delta_t"] = df["t_relativo"].astype(float).diff()
df["delta_t"] = df["delta_t"].bfill()

# Recortar la primera fila para asegurar consistencia matemática
df = df.iloc[1:].reset_index(drop=True)

# Deltas de los motores restando calibración Home
df["delta_real_m1"] = df["real_m1"].astype(float) - home["m1"]
df["delta_real_m2"] = df["real_m2"].astype(float) - home["m2"]
df["delta_real_m3"] = df["real_m3"].astype(float) - home["m3"]
df["delta_real_m4"] = df["real_m4"].astype(float) - home["m4"]

df["delta_meta_m1"] = df["meta_m1"].astype(float) - home["m1"]
df["delta_meta_m2"] = df["meta_m2"].astype(float) - home["m2"]
df["delta_meta_m3"] = df["meta_m3"].astype(float) - home["m3"]
df["delta_meta_m4"] = df["meta_m4"].astype(float) - home["m4"]

# Transformación Geométrica de Coordenadas de OptiTrack (Base -> Efector)
p_base = df[["base_x", "base_y", "base_z"]].astype(float).values
p_efector = df[["efector_x", "efector_y", "efector_z"]].astype(float).values

r_base = R.from_quat(
    df[["base_qx", "base_qy", "base_qz", "base_qw"]].astype(float).values
)
r_efector = R.from_quat(
    df[["efector_qx", "efector_qy", "efector_qz", "efector_qw"]]
    .astype(float)
    .values
)

p_relativo = r_base.inv().apply(p_efector - p_base)
r_relativo = r_base.inv() * r_efector
q_relativo = r_relativo.as_quat()

df["rel_x"], df["rel_y"], df["rel_z"] = (
    p_relativo[:, 0],
    p_relativo[:, 1],
    p_relativo[:, 2],
)
df["rel_qx"], df["rel_qy"], df["rel_qz"], df["rel_qw"] = (
    q_relativo[:, 0],
    q_relativo[:, 1],
    q_relativo[:, 2],
    q_relativo[:, 3],
)

# =====================================================================
# 4. AISLAMIENTO Y ESCALADO HISTÓRICO [-1, 1]
# =====================================================================
x_keys = list(norm_params["X_transformer"].keys())
y_keys = cfg["y_keys"]

X_raw_matrix = df[x_keys].astype(float).values
Y_raw_matrix = df[y_keys].astype(float).values


def normalizar_con_limites_json(matrix, params_section, keys_list):
  matrix_scaled = np.zeros_like(matrix, dtype=np.float32)
  for idx, key in enumerate(keys_list):
    min_t = params_section[key]["min_t"]
    max_t = params_section[key]["max_t"]
    matrix_scaled[:, idx] = 2 * (matrix[:, idx] - min_t) / (max_t - min_t) - 1
  return matrix_scaled


X_scaled = normalizar_con_limites_json(
    X_raw_matrix, norm_params["X_transformer"], x_keys
)
Y_scaled = normalizar_con_limites_json(
    Y_raw_matrix, norm_params["Y_transformer"], y_keys
)

# =====================================================================
# 5. CREACIÓN DE VENTANAS TEMPORALES CONTINUAS Y DATALOADER
# =====================================================================
X_seq, Y_seq = [], []
for i in range(len(X_scaled) - WINDOW_SIZE):
  X_seq.append(X_scaled[i : i + WINDOW_SIZE])
  Y_seq.append(Y_scaled[i + WINDOW_SIZE])

# [NUEVO] Creación de Dataset y DataLoader manteniendo el orden temporal (shuffle=False)
X_tensor_all = torch.tensor(np.array(X_seq), dtype=torch.float32)
Y_tensor_all = torch.tensor(np.array(Y_seq), dtype=torch.float32)

eval_dataset = TensorDataset(X_tensor_all, Y_tensor_all)
eval_loader = DataLoader(
    eval_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False
)


# =====================================================================
# 6. DEFINICIÓN DE ARQUITECTURA LSTM Y CARGA DE PESOS
# =====================================================================
class SoftRobotLSTM(nn.Module):

  def __init__(
      self,
      input_size,
      hidden_size=128,
      num_layers=2,
      output_size=3,
      dropout=0.0,
  ):
    super(SoftRobotLSTM, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
        x.device
    )  # Vinculado dinámicamente al device del tensor de entrada
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    out, _ = self.lstm(x, (h0, c0))
    last_step = out[:, -1, :]
    out_regularized = self.dropout(last_step)
    return self.fc(out_regularized)


model = SoftRobotLSTM(
    input_size=cfg["input_size"], output_size=cfg["output_size"], dropout=0.2
).to(device)
model.load_state_dict(torch.load(PATH_PESOS, map_location=device))
model.eval()

# =====================================================================
# 7. INFERENCIA BATCH-BY-BATCH Y DESNORMALIZACIÓN
# =====================================================================
preds_list = []
real_list = []

# [NUEVO] Inferencia por lotes (BATCH_SIZE_EVAL = 512)
with torch.no_grad():
  for batch_x, batch_y in eval_loader:
    batch_x = batch_x.to(device)
    outputs = model(batch_x)

    preds_list.append(outputs.cpu().numpy())
    real_list.append(batch_y.numpy())

# Concatenar todos los lotes procesados
preds_scaled = np.vstack(preds_list)
real_scaled = np.vstack(real_list)


def desnormalizar_matriz(matrix_sc, params_section, keys_list):
  matrix_phys = np.zeros_like(matrix_sc)
  for idx, key in enumerate(keys_list):
    min_t = params_section[key]["min_t"]
    max_t = params_section[key]["max_t"]
    matrix_phys[:, idx] = min_t + ((matrix_sc[:, idx] + 1) / 2) * (max_t - min_t)
  return matrix_phys


Y_real_phys = desnormalizar_matriz(
    real_scaled, norm_params["Y_transformer"], y_keys
)
Y_pred_phys = desnormalizar_matriz(
    preds_scaled, norm_params["Y_transformer"], y_keys
)

# Filtro algebraico para cuaterniones en V3 y V4
if cfg["output_size"] == 7:
  q_vectors = Y_pred_phys[:, 3:7]
  Y_pred_phys[:, 3:7] = q_vectors / np.linalg.norm(
      q_vectors, axis=1, keepdims=True
  )

# Conversión a Milímetros
Y_real_mm = Y_real_phys[:, :3] * 1000.0
Y_pred_mm = Y_pred_phys[:, :3] * 1000.0

# =====================================================================
# 8. CÁLCULO DE MÉTRICAS CIENTÍFICAS DE ERROR
# =====================================================================
mae_ejes = np.mean(np.abs(Y_real_mm - Y_pred_mm), axis=0)
error_euclidiano_3d = np.sqrt(np.sum((Y_real_mm - Y_pred_mm) ** 2, axis=1))
mae_3d_promedio = np.mean(error_euclidiano_3d)

print("\n=======================================================")
print(f"📊 RESULTATS DE VALIDATION ({VERSION_MODELO.upper()})")
print("=======================================================")
print(f"Erreur MAE Eje X: {mae_ejes[0]:.3f} mm")
print(f"Erreur MAE Eje Y: {mae_ejes[1]:.3f} mm")
print(f"Erreur MAE Eje Z: {mae_ejes[2]:.3f} mm")
print(f"📐 ERREUR DE DISTANCE EUCLÍDEA 3D MOYENNE: {mae_3d_promedio:.3f} mm")
print("=======================================================\n")

# =====================================================================
# 9. GENERACIÓN DE GRÁFICOS COMPUESTOS EXPERIMENTALES
# =====================================================================
fig = plt.figure(figsize=(14, 6))

# Gráfico Izquierdo: Reconstrucción espacial 3D
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot(
    Y_real_mm[:, 0],
    Y_real_mm[:, 1],
    Y_real_mm[:, 2],
    color="black",
    label="Real (OptiTrack)",
    linewidth=2.5,
)
ax1.plot(
    Y_pred_mm[:, 0],
    Y_pred_mm[:, 1],
    Y_pred_mm[:, 2],
    color="crimson",
    linestyle="--",
    label="Predicho (PINN-LSTM)",
    linewidth=2,
)
ax1.set_title(
    f"Trayectoria en Espacio 3D Real (Modo {VERSION_MODELO.upper()})",
    fontsize=12,
    fontweight="bold",
)
ax1.set_xlabel("Eje X (mm)")
ax1.set_ylabel("Eje Y (mm)")
ax1.set_zlabel("Eje Z (mm)")
ax1.legend()
ax1.grid(True)

# Gráfico Derecho: Comportamiento por eje a lo largo del tiempo
ax2 = fig.add_subplot(1, 2, 2)
time_axis = np.arange(len(Y_real_mm))
ax2.plot(
    time_axis, Y_real_mm[:, 0], color="black", label="X Real", linewidth=1.5
)
ax2.plot(
    time_axis,
    Y_pred_mm[:, 0],
    color="crimson",
    linestyle="--",
    label="X Predicho",
)
ax2.plot(
    time_axis,
    Y_real_mm[:, 1],
    color="darkblue",
    alpha=0.4,
    label="Y Real",
)
ax2.plot(
    time_axis,
    Y_pred_mm[:, 1],
    color="dodgerblue",
    linestyle=":",
    label="Y Predicho",
)
ax2.plot(
    time_axis,
    Y_real_mm[:, 2],
    color="darkgreen",
    alpha=0.4,
    label="Z Real",
)
ax2.plot(
    time_axis,
    Y_pred_mm[:, 2],
    color="limegreen",
    linestyle=":",
    label="Z Predicho",
)

ax2.set_title(
    "Seguimiento Temporal Desglosado por Eje (mm)",
    fontsize=12,
    fontweight="bold",
)
ax2.set_xlabel("Muestras / Secuencias Temporales")
ax2.set_ylabel("Posición (mm)")
ax2.grid(True)
ax2.legend(loc="upper right", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()