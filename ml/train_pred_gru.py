import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

# =====================================================================
# 1. CONFIGURACIÓN E HIPERPARÁMETROS
# =====================================================================
# Comparación directa contra train_pred.py: MISMO dataset, MISMA PINNLossMPC
# (con la ponderación por eje + dirección X-Z), MISMO split -- la única
# diferencia es el encoder recurrente (GRU en vez de LSTM, ver sección 3).
PATH_DATASET = 'dataset_pred_v08_completo_15_conRot_directo_filt.npy'
PATH_METADATA = 'dataset_pred_v08_completo_15_conRot_directo_filt_params.json'
MODEL_SAVE_PATH = 'best_mpc_pinn_predictor_gru_15.pth'

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🚀 Entrenando (GRU) en dispositivo: {DEVICE}')

# Cargar parámetros de escalado desde el JSON
with open(PATH_METADATA, 'r') as f:
  metadata = json.load(f)
params_y = metadata['Y_transformer']


# =====================================================================
# 2. FUNCIÓN DE PÉRDIDA INFORMADA POR LA FÍSICA (PINN LOSS) -- idéntica a
# train_pred.py, sin cambios.
# =====================================================================
class PINNLossMPC(nn.Module):

  def __init__(
      self,
      params_y,
      w_mse=100.0,
      w_direction_3d=30.0,  # Alineación angular del rollout en 3D completo (no solo X-Z)
      w_sph_max=10.0,
      w_sph_min=10.0,
      w_cyl_max=10.0,
      w_speed=0.0,
      w_smooth=0.0,
      w_quat=5.0,
      max_step_dist=0.01,  # Máximo desplazamiento permitido por paso (1.5 cm)
      pesos_ejes_mse=(1.5, 1.0, 2.2),  # [X, Y, Z] -- Z y X (horizontal) pesan
      # más que Y (altura) porque el error observado ahí es mayor (~4.9mm/
      # ~3.5mm vs. Y) y porque el problema de las "agujas" perpendiculares a
      # la curva real ocurre en el plano X-Z, no en la altura.
  ):
    super(PINNLossMPC, self).__init__()
    self.params_y = params_y
    self.w_mse = w_mse
    self.w_direction_3d = w_direction_3d
    self.w_sph_max = w_sph_max
    self.w_sph_min = w_sph_min
    self.w_cyl_max = w_cyl_max
    self.w_speed = w_speed
    self.w_smooth = w_smooth
    self.w_quat = w_quat
    self.max_step_dist = max_step_dist
    self.register_buffer('pesos_ejes_mse', torch.tensor(pesos_ejes_mse, dtype=torch.float32))

    self.mse = nn.MSELoss()

    # Extraer límites para desescalar de [-1, 1] a Metros Reales
    self.min_x = params_y['rel_x']['min_t']
    self.max_x = params_y['rel_x']['max_t']
    self.min_y = params_y['rel_y']['min_t']
    self.max_y = params_y['rel_y']['max_t']
    self.min_z = params_y['rel_z']['min_t']
    self.max_z = params_y['rel_z']['max_t']

  def desescalar_xyz(self, y_scaled):
    # y_scaled shape: [batch, T_OUT, num_vars]
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
    # 1. Pérdida Principal MSE (Puntos objetivo, espacio escalado [-1,1]) --
    # se sigue devolviendo tal cual para no romper la comparabilidad de
    # best_val_mse entre corridas.
    loss_mse = self.mse(y_pred, y_true)

    # 1a. MSE ponderado por eje, en el MISMO espacio escalado [-1,1] que usa
    # loss_mse (NO en metros reales) -- w_mse ya está calibrado para esta
    # escala. rel_y tiene un rango de solo 150mm contra 440mm de X/Z, así
    # que ponderar en metros reales encoge el término de Y ~178x y el de
    # X/Z ~21x respecto al escalado -- sin subir w_mse en esa misma
    # proporción, esta pérdida queda opacada por loss_direction_3d y las
    # geométricas, y el modelo deja de priorizar acertar la posición.
    error_xyz_escalado = (y_pred[:, :, :3] - y_true[:, :, :3]) ** 2  # [batch, T_OUT, 3]
    loss_mse_ponderado = torch.mean(error_xyz_escalado * self.pesos_ejes_mse)

    # 2. Desescalar a Metros Reales -- de AMBOS, predicción y verdad. Esto sí
    # hace falta en metros: las penalizaciones geométricas comparan contra
    # umbrales físicos fijos (0.335m, etc.), y loss_direction_3d es
    # invariante a escala (cosine similarity), así que da igual.
    x_pred, y_pred_real, z_pred = self.desescalar_xyz(y_pred)
    x_true, y_true_real, z_true = self.desescalar_xyz(y_true)

    pos_pred = torch.stack([x_pred, y_pred_real, z_pred], dim=-1)  # [batch, T_OUT, 3]
    pos_true = torch.stack([x_true, y_true_real, z_true], dim=-1)  # [batch, T_OUT, 3]

    # 2b. Pérdida de DIRECCIÓN en 3D completo (X, Y, Z), con similitud coseno
    # entre desplazamientos consecutivos real vs. predicho. Se excluyen las
    # transiciones donde el desplazamiento real en 3D es menor a 0.5mm
    # (dirección real poco significativa/ruidosa ahí).
    dir_pred_3d = pos_pred[:, 1:, :] - pos_pred[:, :-1, :]  # [batch, T_OUT-1, 3]
    dir_true_3d = pos_true[:, 1:, :] - pos_true[:, :-1, :]  # [batch, T_OUT-1, 3]

    norm_true_3d = torch.norm(dir_true_3d, dim=-1)  # Distancia euclidiana 3D por paso
    mask_movimiento = norm_true_3d > 0.0005  # Solo penalizar si se movió más de 0.5 mm en 3D

    cos_sim_3d = torch.nn.functional.cosine_similarity(
        dir_pred_3d, dir_true_3d, dim=-1, eps=1e-6
    )
    loss_dir_raw = 1.0 - cos_sim_3d

    if mask_movimiento.sum() > 0:
      loss_direction_3d = torch.mean(loss_dir_raw[mask_movimiento])
    else:
      loss_direction_3d = torch.tensor(0.0, device=y_pred.device)

    # Distancias Geométricas (sobre la posición PREDICHA)
    dist_esferica_cuadrada = x_pred**2 + y_pred_real**2 + z_pred**2
    dist_cilindrica_cuadrada = x_pred**2 + z_pred**2

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
    paso_diffs = pos_pred[:, 1:, :] - pos_pred[:, :-1, :]  # [batch, T_OUT-1, 3]
    dist_por_paso = torch.norm(paso_diffs, dim=-1)  # [batch, T_OUT-1]

    pen_velocidad = torch.relu(dist_por_paso - self.max_step_dist)
    loss_speed = self.w_speed * torch.mean(pen_velocidad)

    # C) Restricción de Suavizado Temporal (Minimizar Aceleración / Jerk)
    aceleracion = paso_diffs[:, 1:, :] - paso_diffs[:, :-1, :]  # [batch, T_OUT-2, 3]
    loss_smooth = self.w_smooth * torch.mean(aceleracion**2)

    # D) Normalización de Cuaterniones (Si existen 7 salidas)
    loss_quat = 0.0
    if y_pred.shape[-1] == 7:
      q = y_pred[:, :, 3:]  # [batch, T_OUT, 4]
      norm_q = torch.norm(q, dim=-1)
      loss_quat = self.w_quat * torch.mean((norm_q - 1.0) ** 2)

    # PÉRDIDA TOTAL PINN
    total_loss = (
        self.w_mse * loss_mse_ponderado
        + self.w_direction_3d * loss_direction_3d
        + loss_geom
        + loss_speed
        + loss_smooth
        + loss_quat
    )
    return total_loss, loss_mse


# =====================================================================
# 3. DATASET Y MODELO -- ÚNICA DIFERENCIA REAL CON train_pred.py: el
# encoder recurrente es GRU en vez de LSTM (menos parámetros -- funde las
# puertas de olvido/entrada en una sola puerta de actualización y no tiene
# estado de celda separado -- comparar val_mse contra la versión LSTM).
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


class MPCDirectPredictorGRU(nn.Module):

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
    super(MPCDirectPredictorGRU, self).__init__()
    self.encoder_gru = nn.GRU(
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
    out, _ = self.encoder_gru(x_hist)  # GRU: solo hidden state, sin cell state
    last_hidden = out[:, -1, :]
    combined = torch.cat([last_hidden, u_cand], dim=1)
    y_pred_flat = self.mlp(combined)
    return y_pred_flat.view(-1, self.t_out, self.output_dim)


# =====================================================================
# 4. PREPARACIÓN DE DATOS -- SPLIT TRAIN/VAL REPARTIDO Y SIN FUGA (sobre
# 'largo'), idéntico a train_pred.py. 'corto' queda reservado como test
# ciego final.
# =====================================================================
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
model = MPCDirectPredictorGRU(
    input_hist_dim=input_hist_dim,
    u_cand_dim=len(sample_u),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    t_out=t_out,
    output_dim=output_dim,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f'🧮 Parámetros totales del modelo (GRU): {n_params:,}')

# Criterio PINN (idéntico a train_pred.py)
criterion = PINNLossMPC(params_y=params_y).to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
)

# =====================================================================
# 6. ENTRENAMIENTO
# =====================================================================
best_val_mse = float('inf')
history_train_loss, history_val_mse = [], []

print('\n🔥 Iniciando entrenamiento (GRU) con PINN Loss...\n')

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
            'arquitectura': 'GRU',
        },
        MODEL_SAVE_PATH,
    )
    saved_flag = '⭐ [Guardado]'
  else:
    saved_flag = ''

  current_lr = optimizer.param_groups[0]['lr']
  if epoch % 5 == 0 or epoch == 1 or saved_flag:
    print(
        f'Epoch [{epoch:03d}/{EPOCHS:03d}] | PINN Train Loss: {train_loss:.6f}'
        f' | Val MSE: {val_mse:.6f} | LR: {current_lr:.2e} {saved_flag}'
    )

print(f'\n✅ Entrenamiento (GRU) completado. Mejor Val MSE: {best_val_mse:.6f}')
