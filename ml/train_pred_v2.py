import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

# =====================================================================
# 1. CONFIGURACIÓN Y HIPERPARÁMETROS
# =====================================================================
# Variante "liviana" de train_pred.py: en vez de 'completo' (17 entradas:
# real+meta+torque+tension) + conRot (7 salidas con cuaternión), usa
# 'real_meta' (9 entradas: tiempo + ángulos reales + ángulos objetivo) +
# sinRot (3 salidas, solo posición) -- dataset_pred_v01_real_meta_sinRot
# (v01 en la numeración de dataset_pred_filt.py: x_tag='real_meta' es el
# primero del loop, y_tag='sinRot' también, de ahí v01; NO es un error de
# nombre -- dataset_pred_v08 = 'completo'+conRot, el que usa train_pred.py).
#
# NOTA: 'corto' (grabación independiente) NO se usa acá para validar -- se
# reserva 100% como test ciego final, sin tocarse nunca durante el
# entrenamiento. La validación sale de 'largo' mismo, con el mismo split
# por bloques distribuidos que train_v8_time_100_2.py (ver sección 4).
PATH_DATASET = 'dataset_pred_v01_real_meta_sinRot_directo_filt.npy'
PATH_METADATA = 'dataset_pred_v01_real_meta_sinRot_directo_filt_params.json'
MODEL_SAVE_PATH = 'best_mpc_pinn_predictor_v2_realmeta.pth'

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🚀 Entrenando en dispositivo: {DEVICE}')

# Cargar parámetros de escalado desde el JSON
with open(PATH_METADATA, 'r') as f:
  metadata = json.load(f)
params_y = metadata['Y_transformer']


# =====================================================================
# 2. FUNCIÓN DE PÉRDIDA INFORMADA POR LA FÍSICA (PINN LOSS)
# =====================================================================
class PINNLossMPC(nn.Module):

  def __init__(
      self,
      params_y,
      w_mse=1.0,
      w_sph_max=10.0,
      w_sph_min=10.0,
      w_cyl_max=10.0,
      w_speed=5.0,
      w_smooth=2.0,
      w_quat=5.0,
      max_step_dist=0.015,  # Máximo desplazamiento permitido por paso (1.5 cm)
  ):
    super(PINNLossMPC, self).__init__()
    self.params_y = params_y
    self.w_mse = w_mse
    self.w_sph_max = w_sph_max
    self.w_sph_min = w_sph_min
    self.w_cyl_max = w_cyl_max
    self.w_speed = w_speed
    self.w_smooth = w_smooth
    self.w_quat = w_quat
    self.max_step_dist = max_step_dist

    self.mse = nn.MSELoss()

    # Extraer límites para desescalar de [-1, 1] a Metros Reales
    self.min_x = params_y['rel_x']['min_t']
    self.max_x = params_y['rel_x']['max_t']
    self.min_y = params_y['rel_y']['min_t']
    self.max_y = params_y['rel_y']['max_t']
    self.min_z = params_y['rel_z']['min_t']
    self.max_z = params_y['rel_z']['max_t']

  def desescalar_xyz(self, y_scaled):
    # y_scaled shape: [batch, 10, num_vars]
    x_real = (
        self.min_x + (y_scaled[:, :, 0] + 1.0) * (self.max_x - self.min_x) / 2.0
    )
    y_real = (
        self.min_y + (y_scaled[:, :, 1] + 1.0) * (self.max_y - self.min_y) / 2.0
    )
    z_real = (
        self.min_z + (y_scaled[:, :, 2] + 1.0) * (self.max_z - self.min_z) / 2.0
    )
    return x_real, y_real, z_real

  def forward(self, y_pred, y_true):
    # 1. Pérdida Principal MSE (Puntos objetivo)
    loss_mse = self.mse(y_pred, y_true)

    # 2. Desescalar predicciones a Metros Reales
    x_real, y_real, z_real = self.desescalar_xyz(y_pred)

    # Distancias Geométricas
    dist_esferica_cuadrada = x_real**2 + y_real**2 + z_real**2
    dist_cilindrica_cuadrada = x_real**2 + z_real**2

    # A) Penalizaciones Geométricas Espaciales
    pen_sph_max = torch.relu(dist_esferica_cuadrada - 0.335**2)
    pen_sph_min = torch.relu(0.22**2 - dist_esferica_cuadrada)
    pen_cyl_max = torch.relu(dist_cilindrica_cuadrada - 0.23**2)

    loss_geom = (
        self.w_sph_max * torch.mean(pen_sph_max)
        + self.w_sph_min * torch.mean(pen_sph_min)
        + self.w_cyl_max * torch.mean(pen_cyl_max)
    )

    # B) Restricción de Velocidad Máxima por Paso (Continuidad Física)
    # Calcutar diferencia entre pasos consecutivos (t+1 - t, t+2 - t+1, ...)
    pos_real = torch.stack([x_real, y_real, z_real], dim=-1)  # [batch, 10, 3]
    paso_diffs = pos_real[:, 1:, :] - pos_real[:, :-1, :]  # [batch, 9, 3]
    dist_por_paso = torch.norm(paso_diffs, dim=-1)  # [batch, 9]

    pen_velocidad = torch.relu(dist_por_paso - self.max_step_dist)
    loss_speed = self.w_speed * torch.mean(pen_velocidad)

    # C) Restricción de Suavizado Temporal (Minimizar Aceleración / Jerk)
    aceleracion = paso_diffs[:, 1:, :] - paso_diffs[:, :-1, :]  # [batch, 8, 3]
    loss_smooth = self.w_smooth * torch.mean(aceleracion**2)

    # D) Normalización de Cuaterniones (Si existen 7 salidas)
    loss_quat = 0.0
    if y_pred.shape[-1] == 7:
      q = y_pred[:, :, 3:]  # [batch, 10, 4]
      norm_q = torch.norm(q, dim=-1)
      loss_quat = self.w_quat * torch.mean((norm_q - 1.0) ** 2)

    # PÉRDIDA TOTAL PINN
    total_loss = (
        self.w_mse * loss_mse + loss_geom + loss_speed + loss_smooth + loss_quat
    )
    return total_loss, loss_mse


# =====================================================================
# 3. DATASET Y MODELO (IGUAL QUE ANTES)
# =====================================================================
class MPCDataset(Dataset):

  def __init__(self, npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    self.x_hist = torch.tensor(data['X_hist'], dtype=torch.float32)
    self.u_cand = torch.tensor(data['U_cand'], dtype=torch.float32)
    self.y_fut = torch.tensor(data['Y_fut'], dtype=torch.float32)

  def __len__(self):
    return len(self.x_hist)

  def __getitem__(self, idx):
    return self.x_hist[idx], self.u_cand[idx], self.y_fut[idx]


class MPCDirectPredictor(nn.Module):

  def __init__(
      self,
      input_hist_dim,
      u_cand_dim=4,
      hidden_size=128,
      num_layers=2,
      t_out=10,
      output_dim=3,
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


# =====================================================================
# 4. PREPARACIÓN DE DATOS -- SPLIT TRAIN/VAL REPARTIDO Y SIN FUGA (sobre 'largo')
# =====================================================================
# Igual estrategia que train_v8_time_100_2.py: 'largo' se divide en
# N_BLOQUES contiguos a lo largo de TODA la grabación (no solo el tramo
# final); una fracción FRAC_VAL de esos bloques, repartidos uniformemente,
# se reserva para validación. La diferencia con v8 es que acá el ventaneo
# YA ocurrió en dataset_pred_filt.py (cada muestra k de X_hist/U_cand/Y_fut
# cubre un rango de T_IN+T_OUT filas originales) -- por eso, en vez de
# generar ventanas nuevas, se descarta cualquier muestra k cuyo margen de
# solapamiento (k hasta k+T_IN+T_OUT) cruce un borde entre un bloque de
# train y uno de val, para que ningún par comparta filas originales.
FRAC_VAL = 0.10
N_BLOQUES = 20


def dividir_indices_por_bloques(n_total, margen_solapamiento, frac_val=FRAC_VAL, n_bloques=N_BLOQUES):
  tam_bloque = n_total // n_bloques
  n_bloques_val = max(1, round(n_bloques * frac_val))
  paso = n_bloques / n_bloques_val
  bloques_val = {
      int(round(paso / 2 + i * paso)) % n_bloques for i in range(n_bloques_val)
  }

  def bloque_de(idx):
    return min(idx // tam_bloque, n_bloques - 1)

  idx_train, idx_val = [], []
  descartadas = 0

  for k in range(n_total):
    fin = min(k + margen_solapamiento, n_total - 1)
    bloque_inicio = bloque_de(k)
    bloque_fin = bloque_de(fin)
    if bloque_inicio != bloque_fin:
      descartadas += 1
      continue
    (idx_val if bloque_inicio in bloques_val else idx_train).append(k)

  print(f'   (bloques de validación repartidos: {sorted(bloques_val)}/{n_bloques} '
        f'| {descartadas} muestras descartadas por cruzar un borde de bloque)')

  return idx_train, idx_val


dataset_completo = MPCDataset(PATH_DATASET)
sample_x, sample_u, sample_y = dataset_completo[0]
t_in, input_hist_dim = sample_x.shape
t_out, output_dim = sample_y.shape

idx_train, idx_val = dividir_indices_por_bloques(
    len(dataset_completo), margen_solapamiento=t_in + t_out
)
train_dataset = Subset(dataset_completo, idx_train)
val_dataset = Subset(dataset_completo, idx_val)
train_size = len(train_dataset)
val_size = len(val_dataset)

print(f'📦 Muestras de entrenamiento: {train_size} | Muestras de validación: '
      f'{val_size} (ambas de "largo" -- "corto" queda reservado como test ciego final)')

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)

# =====================================================================
# 5. INSTANCIACIÓN Y OPTIMIZADOR
# =====================================================================
model = MPCDirectPredictor(
    input_hist_dim=input_hist_dim,
    u_cand_dim=len(sample_u),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    t_out=t_out,
    output_dim=output_dim,
    dropout=DROPOUT,
).to(DEVICE)

# Criterio PINN
criterion = PINNLossMPC(params_y=params_y).to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# =====================================================================
# 6. ENTRENAMIENTO
# =====================================================================
best_val_mse = float('inf')
history_train_loss, history_val_mse = [], []

print('\n🔥 Iniciando entrenamiento con PINN Loss...\n')

for epoch in range(1, EPOCHS + 1):
  model.train()
  train_loss = 0.0

  for batch_x, batch_u, batch_y in train_loader:
    batch_x, batch_u, batch_y = (
        batch_x.to(DEVICE),
        batch_u.to(DEVICE),
        batch_y.to(DEVICE),
    )

    optimizer.zero_grad()
    predictions = model(batch_x, batch_u)

    # Calcular PINN Loss
    total_loss, pure_mse = criterion(predictions, batch_y)
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    train_loss += total_loss.item() * len(batch_x)

  train_loss /= train_size

  # Validación
  model.eval()
  val_mse_sum = 0.0

  with torch.no_grad():
    for batch_x, batch_u, batch_y in val_loader:
      batch_x, batch_u, batch_y = (
          batch_x.to(DEVICE),
          batch_u.to(DEVICE),
          batch_y.to(DEVICE),
      )

      predictions = model(batch_x, batch_u)
      _, pure_mse = criterion(predictions, batch_y)
      val_mse_sum += pure_mse.item() * len(batch_x)

  val_mse = val_mse_sum / val_size
  scheduler.step(val_mse)

  history_train_loss.append(train_loss)
  history_val_mse.append(val_mse)

  if val_mse < best_val_mse:
    best_val_mse = val_mse
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'input_hist_dim': input_hist_dim,
            'u_cand_dim': len(sample_u),
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            't_out': t_out,
            'output_dim': output_dim,
            'best_val_mse': best_val_mse,
        },
        MODEL_SAVE_PATH,
    )
    saved_flag = '⭐ [Guardado]'
  else:
    saved_flag = ''

  if epoch % 5 == 0 or epoch == 1:
    print(
        f'Epoch [{epoch:03d}/{EPOCHS:03d}] | PINN Train Loss: {train_loss:.6f}'
        f' | Val MSE: {val_mse:.6f} {saved_flag}'
    )

print(f'\n✅ Entrenamiento completado. Mejor Val MSE: {best_val_mse:.6f}')
