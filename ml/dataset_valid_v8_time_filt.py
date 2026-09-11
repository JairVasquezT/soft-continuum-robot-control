import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------
# NOTA: Se eliminó la importación que causaba el ModuleNotFoundError
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='soft_robot_lstm_v8_18_vel.pth')
parser.add_argument('--window_size', type=int, default=90)
# hidden_size/num_layers: solo se usan como FALLBACK si el checkpoint no trae
# esa metadata embebida (los .pth de train_v8_time_100_2.py sí la traen).
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--title', type=str, default='')
args = parser.parse_args()

# =====================================================================
# 1. PANEL DE CONTROL (Configurado para Modelo V8 Single-Stream)
# =====================================================================
VERSION_MODELO = "v8"  # Seleccionado modo V8 Completo (17 Entradas)
WINDOW_SIZE = args.window_size  # Ventana temporal de 90 muestras (1.5s a 60 Hz) -- debe
# coincidir EXACTO con el WINDOW_SIZE usado al entrenar (train_v8_time_100.py)
BATCH_SIZE_EVAL = 512  # Inferencia por lotes protegida contra CUDA OOM
FS_SISTEMA = 60.0  # Frecuencia de muestreo del sistema (60 Hz)

#logs_trayectoria_final_corto
#corto_20260730_195350_730
PATH_CSV = "logs_trayectoria_final_corto.csv"
PATH_PESOS = args.model
PATH_JSON = "dataset_v08_completo_sinRot_filt_params.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"🔥 Procesando CSV con dispositivo: {device} | Modo:"
    f" {VERSION_MODELO.upper()}"
)

# Diccionario de control de la arquitectura ampliado con V8
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
    "v8": {
        "input_size": 17,
        "output_size": 3,
        "y_keys": ["rel_x", "rel_y", "rel_z"],
    },  # Mode V8 Single-Stream
}
cfg = config_arquitectura[VERSION_MODELO]

# =====================================================================
# 2. CARGA DE PARÁMETROS MAESTROS DESDE EL JSON
# =====================================================================
with open(PATH_JSON, "r") as f:
  norm_params = json.load(f)

home = norm_params["home_motores"]

# =====================================================================
# 3. PROCESAMIENTO MECÁNICO Y GEOMÉTRICO (17 FEATURES)
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

# 1. SUBMUESTREO A 30 Hz (Toma 1 de cada 2 filas)
#df = df.iloc[::2].reset_index(drop=True)

# Diferencial de tiempo dt
df["delta_t"] = df["t_relativo"].astype(float).diff().bfill()
df = df.iloc[1:].reset_index(drop=True)

# Deltas de los 4 motores restando la posición Home
for i in range(1, 5):
  df[f"delta_real_m{i}"] = df[f"real_m{i}"].astype(float) - home[f"m{i}"]
  df[f"delta_meta_m{i}"] = df[f"meta_m{i}"].astype(float) - home[f"m{i}"]
  df[f"couple_m{i}"] = df[f"couple_m{i}"].astype(float)
  df[f"tension_m{i}"] = df[f"tension_m{i}"].astype(float)

# Transformación Cinemática de OptiTrack (Base -> Efector)
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
# 3.5. FILTRADO DIGITAL DE FASE CERO (PASABAJAS + UNWRAP)
# =====================================================================
def aplicar_filtro_pasabajas(data, cutoff_hz, fs_hz=60.0, order=2):
  nyquist = 0.5 * fs_hz
  normal_cutoff = cutoff_hz / nyquist
  b, a = butter(order, normal_cutoff, btype="low", analog=False)
  return filtfilt(b, a, data, axis=0)


def filtrar_y_normalizar_cuaterniones(
    df_quat, cutoff_hz=6.0, fs_hz=60.0, order=2
):
  q_vals = df_quat[["rel_qx", "rel_qy", "rel_qz", "rel_qw"]].values.copy()
  for i in range(1, len(q_vals)):
    if np.dot(q_vals[i], q_vals[i - 1]) < 0:
      q_vals[i] = -q_vals[i]
  q_filt = aplicar_filtro_pasabajas(
      q_vals, cutoff_hz=cutoff_hz, fs_hz=fs_hz, order=order
  )
  normas = np.linalg.norm(q_filt, axis=1, keepdims=True)
  normas[normas == 0] = 1.0
  return q_filt / normas


print("🧹 Aplicando filtrado pasabajas al conjunto de Test V8 Single-Stream...")

for col in ["rel_x", "rel_y", "rel_z"]:
  df[col] = aplicar_filtro_pasabajas(
      df[col].values, cutoff_hz=6.0, fs_hz=FS_SISTEMA
  )

q_norm = filtrar_y_normalizar_cuaterniones(
    df[["rel_qx", "rel_qy", "rel_qz", "rel_qw"]],
    cutoff_hz=6.0,
    fs_hz=FS_SISTEMA,
)
df["rel_qx"], df["rel_qy"], df["rel_qz"], df["rel_qw"] = (
    q_norm[:, 0],
    q_norm[:, 1],
    q_norm[:, 2],
    q_norm[:, 3],
)

for i in range(1, 5):
  df[f"couple_m{i}"] = aplicar_filtro_pasabajas(
      df[f"couple_m{i}"].values, cutoff_hz=3.5, fs_hz=FS_SISTEMA
  )
  df[f"tension_m{i}"] = aplicar_filtro_pasabajas(
      df[f"tension_m{i}"].values, cutoff_hz=3.5, fs_hz=FS_SISTEMA
  )

print("✓ Filtrado V8 Single-Stream completado exitosamente.")

# =====================================================================
# 4. NORMALIZACIÓN AUTOMÁTICA EN BASE AL JSON
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
    denom = max_t - min_t if (max_t - min_t) != 0 else 1.0
    matrix_scaled[:, idx] = 2 * (matrix[:, idx] - min_t) / denom - 1

  return matrix_scaled


X_scaled = normalizar_con_limites_json(
    X_raw_matrix, norm_params["X_transformer"], x_keys
)
Y_scaled = normalizar_con_limites_json(
    Y_raw_matrix, norm_params["Y_transformer"], y_keys
)

print("\n--- DIAGNÓSTICO DE NORMALIZACIÓN ---")
print("Mínimos normalizados (X - 17 vars):", np.min(X_scaled, axis=0))
print("Máximos normalizados (X - 17 vars):", np.max(X_scaled, axis=0))
print("Mínimos normalizados (Y - 3 ejes):", np.min(Y_scaled, axis=0))
print("Máximos normalizados (Y - 3 ejes):", np.max(Y_scaled, axis=0))
print("------------------------------------\n")

# =====================================================================
# 5. VENTANADO TEMPORAL Y DATALOADER
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
# 6. ARQUITECTURA DE LA RED E INICIALIZACIÓN (SINGLE-STREAM)
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
    out, _ = self.lstm(x)
    last_step = out[:, -1, :]
    out_regularized = self.dropout(last_step)
    return self.fc(out_regularized)


checkpoint = torch.load(PATH_PESOS, map_location=device)
# Soporta tanto checkpoints "envueltos" (dict con model_state_dict + metadata,
# como los que guarda train_v8_time_100_2.py) como state_dict plano (formato
# viejo, sin envolver).
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
  state_dict = checkpoint["model_state_dict"]
  input_size_ckpt = checkpoint.get("input_size", cfg["input_size"])
  hidden_size_ckpt = checkpoint.get("hidden_size", args.hidden_size)
  num_layers_ckpt = checkpoint.get("num_layers", args.num_layers)
  output_size_ckpt = checkpoint.get("output_size", cfg["output_size"])
else:
  state_dict = checkpoint
  input_size_ckpt = cfg["input_size"]
  hidden_size_ckpt = args.hidden_size
  num_layers_ckpt = args.num_layers
  output_size_ckpt = cfg["output_size"]

def _construir_y_cargar(hidden_size, num_layers):
  m = SoftRobotLSTM(
      input_size=input_size_ckpt,
      hidden_size=hidden_size,
      num_layers=num_layers,
      output_size=output_size_ckpt,
      dropout=0.2,
  ).to(device)
  m.load_state_dict(state_dict)
  return m


try:
  model = _construir_y_cargar(hidden_size_ckpt, num_layers_ckpt)
except RuntimeError as e:
  # La metadata del checkpoint (hidden_size/num_layers) puede estar mal --
  # algunos .pth de la fase 2 del sweep (mayor_capacidad/mas_profunda)
  # quedaron guardados con 'hidden_size'/'num_layers' fijos en 128/2 sin
  # importar la arquitectura real con la que se entrenaron (bug corregido
  # en train_v8_time_100_2.py, pero estos .pth ya existentes conservan la
  # metadata vieja). Los PESOS sí son correctos -- reintentar con la
  # arquitectura que se pasó explícitamente por --hidden_size/--num_layers
  # (la fuente de verdad real: viene del mismo plan de configuraciones que
  # generó el checkpoint, ver run_experimentos_ventana.py).
  if (hidden_size_ckpt, num_layers_ckpt) != (args.hidden_size, args.num_layers):
    print(
        f'⚠️ La metadata del checkpoint (hidden={hidden_size_ckpt}, '
        f'layers={num_layers_ckpt}) no coincide con los pesos guardados. '
        f'Reintentando con --hidden_size={args.hidden_size} '
        f'--num_layers={args.num_layers}...'
    )
    model = _construir_y_cargar(args.hidden_size, args.num_layers)
  else:
    raise e
model.eval()

# =====================================================================
# 7. INFERENCIA EN LOTE Y DESNORMALIZACIÓN
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
    min_t = params_section[key]["min_t"]
    max_t = params_section[key]["max_t"]
    matrix_phys[:, idx] = min_t + ((matrix_sc[:, idx] + 1) / 2) * (
        max_t - min_t
    )
  return matrix_phys


Y_real_phys = desnormalizar_matriz(
    real_scaled, norm_params["Y_transformer"], y_keys
)
Y_pred_phys = desnormalizar_matriz(
    preds_scaled, norm_params["Y_transformer"], y_keys
)

if cfg["output_size"] == 7:
  q_vectors = Y_pred_phys[:, 3:7]
  Y_pred_phys[:, 3:7] = q_vectors / np.linalg.norm(
      q_vectors, axis=1, keepdims=True
  )

# Conversión a milímetros
Y_real_mm = Y_real_phys[:, :3] * 1000.0
Y_pred_mm = Y_pred_phys[:, :3] * 1000.0

print("🔍 DIAGNÓSTICO DE SALIDA:")
print("Primeras 5 predicciones de la red (mm):\n", Y_pred_mm[:5])
print("Primeros 5 valores reales de OptiTrack (mm):\n", Y_real_mm[:5])
print("Varianza de las predicciones en X:", np.var(Y_pred_mm[:, 0]))

# =====================================================================
# 8. CÁLCULO DE MÉTRICAS DE ERROR
# =====================================================================
mae_ejes = np.mean(np.abs(Y_real_mm - Y_pred_mm), axis=0)
error_euclidiano_3d = np.sqrt(np.sum((Y_real_mm - Y_pred_mm) ** 2, axis=1))
mae_3d_promedio = np.mean(error_euclidiano_3d)

print("=======================================================")
print(
    f"📊 RESULTADOS DE VALIDACIÓN EN DATASET OCULTO ({VERSION_MODELO.upper()})"
)
print("=======================================================")
print(f"Error MAE Eje X: {mae_ejes[0]:.3f} mm")
print(f"Error MAE Eje Y: {mae_ejes[1]:.3f} mm")
print(f"Error MAE Eje Z: {mae_ejes[2]:.3f} mm")
print(f"📐 ERROR DE DISTANCIA EUCLÍDEA 3D PROMEDIO: {mae_3d_promedio:.3f} mm")
print("=======================================================\n")

# =====================================================================
# 9. VISUALIZACIÓN GRÁFICA COMPARATIVA
# =====================================================================
fig = plt.figure(figsize=(14, 6))
if args.title:
  try:
    fig.canvas.manager.set_window_title(args.title)
  except Exception:
    pass

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot(
    Y_real_mm[:, 0],
    Y_real_mm[:, 1],
    Y_real_mm[:, 2],
    color="black",
    label="Réel (OptiTrack)",
    linewidth=2.5,
)
ax1.plot(
    Y_pred_mm[:, 0],
    Y_pred_mm[:, 1],
    Y_pred_mm[:, 2],
    color="crimson",
    linestyle="--",
    label="Prédit (V8 Single-Stream)",
    linewidth=2,
)
titulo_extra = f" -- {args.title}" if args.title else ""
ax1.set_title(
    f"Trajectoire 3D Réelle vs Prédite ({VERSION_MODELO.upper()}){titulo_extra}",
    fontsize=12,
    fontweight="bold",
)
ax1.set_xlabel("Axe X (mm)")
ax1.set_ylabel("Axe Y (mm)")
ax1.set_zlabel("Axe Z (mm)")
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(1, 2, 2)
time_axis = np.arange(len(Y_real_mm))
ax2.plot(
    time_axis, Y_real_mm[:, 0], color="black", label="X Réel", linewidth=1.5
)
ax2.plot(
    time_axis,
    Y_pred_mm[:, 0],
    color="crimson",
    linestyle="--",
    label="X Prédit",
)
ax2.plot(
    time_axis,
    Y_real_mm[:, 1],
    color="darkblue",
    alpha=0.4,
    label="Y Réel",
)
ax2.plot(
    time_axis,
    Y_pred_mm[:, 1],
    color="dodgerblue",
    linestyle=":",
    label="Y Prédit",
)
ax2.plot(
    time_axis,
    Y_real_mm[:, 2],
    color="darkgreen",
    alpha=0.4,
    label="Z Réel",
)
ax2.plot(
    time_axis,
    Y_pred_mm[:, 2],
    color="limegreen",
    linestyle=":",
    label="Z Prédit",
)

ax2.set_title(
    "Suivi Temporel Détaillé par Axe (mm)",
    fontsize=12,
    fontweight="bold",
)
ax2.set_xlabel("Échantillons Temporels")
ax2.set_ylabel("Position (mm)")
ax2.grid(True)
ax2.legend(loc="upper right", bbox_to_anchor=(1, 1))

plt.tight_layout()

# =====================================================================
# 10. RÉSUMÉ DE L'ERREUR PAR BLOCS (fenêtre séparée)
# =====================================================================
# Erreur signée (prédit - réel) par axe, sur les positions déjà rotées par
# rapport à la base (Y_real_mm/Y_pred_mm viennent de rel_x/rel_y/rel_z, pas
# du repère monde) -- moyenne ± écart-type par bloc pour voir si l'erreur a
# un biais systématique (dérive) ou si c'est juste du bruit stable dans le
# temps.
def graficar_error_por_bloques(error_x, error_y, error_z, tamano_bloque=200):
  n_bloques = len(error_x) // tamano_bloque
  fig_err, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
  ejes = {'X': error_x, 'Y': error_y, 'Z': error_z}
  colores = {'X': '#2563eb', 'Y': '#16a34a', 'Z': '#dc2626'}

  for ax, (nombre, error) in zip(axes, ejes.items()):
    medias = [np.mean(error[i*tamano_bloque:(i+1)*tamano_bloque]) for i in range(n_bloques)]
    stds = [np.std(error[i*tamano_bloque:(i+1)*tamano_bloque]) for i in range(n_bloques)]
    x = np.arange(n_bloques)
    ax.plot(x, medias, color=colores[nombre], linewidth=1.2)
    ax.fill_between(x, np.array(medias)-np.array(stds), np.array(medias)+np.array(stds),
                     color=colores[nombre], alpha=0.2)
    ax.set_ylabel(f'Erreur {nombre} (mm)')
    ax.grid(alpha=0.25)

  axes[-1].set_xlabel(f'Bloc de {tamano_bloque} échantillons (couvre l\'ensemble du test)')
  fig_err.suptitle("Erreur par composante, moyenne ± écart-type par bloc")
  fig_err.tight_layout()
  return fig_err


error_x_signed = Y_pred_mm[:, 0] - Y_real_mm[:, 0]
error_y_signed = Y_pred_mm[:, 1] - Y_real_mm[:, 1]
error_z_signed = Y_pred_mm[:, 2] - Y_real_mm[:, 2]
fig_error_bloques = graficar_error_por_bloques(error_x_signed, error_y_signed, error_z_signed)
if args.title:
  try:
    fig_error_bloques.canvas.manager.set_window_title(f'Erreur par blocs -- {args.title}')
  except Exception:
    pass

plt.show()