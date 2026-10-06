import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

# =====================================================================
# 1. CONFIGURATION AND HYPERPARAMETERS
# =====================================================================
# "Lightweight" variant of train_pred.py: instead of 'completo' (17 inputs:
# real+meta+torque+tension) + conRot (7 outputs with quaternion), it uses
# 'real_meta' (9 inputs: time + real angles + target angles) +
# sinRot (3 outputs, position only) -- dataset_pred_v01_real_meta_sinRot
# (v01 in the numbering of dataset_pred_filt.py: x_tag='real_meta' is the
# first of the loop, y_tag='sinRot' too, hence v01; it is NOT a naming
# error -- dataset_pred_v08 = 'completo'+conRot, the one used by train_pred.py).
#
# NOTE: 'corto' (independent recording) is NOT used here for validation -- it is
# reserved 100% as the final blind test, without ever being touched during
# training. Validation comes from 'largo' itself, with the same split
# by distributed blocks as train_v8_time_100_2.py (see section 4).
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

# Load scaling parameters from the JSON
with open(PATH_METADATA, 'r') as f:
  metadata = json.load(f)
params_y = metadata['Y_transformer']


# =====================================================================
# 2. PHYSICS-INFORMED LOSS FUNCTION (PINN LOSS)
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
      max_step_dist=0.015,  # Maximum displacement allowed per step (1.5 cm)
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

    # Extract limits to unscale from [-1, 1] to Real Meters
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
    # 1. Main MSE Loss (Target points)
    loss_mse = self.mse(y_pred, y_true)

    # 2. Unscale predictions to Real Meters
    x_real, y_real, z_real = self.desescalar_xyz(y_pred)

    # Geometric Distances
    dist_esferica_cuadrada = x_real**2 + y_real**2 + z_real**2
    dist_cilindrica_cuadrada = x_real**2 + z_real**2

    # A) Spatial Geometric Penalties
    pen_sph_max = torch.relu(dist_esferica_cuadrada - 0.335**2)
    pen_sph_min = torch.relu(0.22**2 - dist_esferica_cuadrada)
    pen_cyl_max = torch.relu(dist_cilindrica_cuadrada - 0.23**2)

    loss_geom = (
        self.w_sph_max * torch.mean(pen_sph_max)
        + self.w_sph_min * torch.mean(pen_sph_min)
        + self.w_cyl_max * torch.mean(pen_cyl_max)
    )

    # B) Maximum Per-Step Velocity Constraint (Physical Continuity)
    # Compute difference between consecutive steps (t+1 - t, t+2 - t+1, ...)
    pos_real = torch.stack([x_real, y_real, z_real], dim=-1)  # [batch, 10, 3]
    paso_diffs = pos_real[:, 1:, :] - pos_real[:, :-1, :]  # [batch, 9, 3]
    dist_por_paso = torch.norm(paso_diffs, dim=-1)  # [batch, 9]

    pen_velocidad = torch.relu(dist_por_paso - self.max_step_dist)
    loss_speed = self.w_speed * torch.mean(pen_velocidad)

    # C) Temporal Smoothing Constraint (Minimize Acceleration / Jerk)
    aceleracion = paso_diffs[:, 1:, :] - paso_diffs[:, :-1, :]  # [batch, 8, 3]
    loss_smooth = self.w_smooth * torch.mean(aceleracion**2)

    # D) Quaternion Normalization (If there are 7 outputs)
    loss_quat = 0.0
    if y_pred.shape[-1] == 7:
      q = y_pred[:, :, 3:]  # [batch, 10, 4]
      norm_q = torch.norm(q, dim=-1)
      loss_quat = self.w_quat * torch.mean((norm_q - 1.0) ** 2)

    # TOTAL PINN LOSS
    total_loss = (
        self.w_mse * loss_mse + loss_geom + loss_speed + loss_smooth + loss_quat
    )
    return total_loss, loss_mse


# =====================================================================
# 3. DATASET AND MODEL (SAME AS BEFORE)
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
# 4. DATA PREPARATION -- DISTRIBUTED, LEAK-FREE TRAIN/VAL SPLIT (over 'largo')
# =====================================================================
# Same strategy as train_v8_time_100_2.py: 'largo' is divided into
# N_BLOQUES contiguous blocks across the WHOLE recording (not just the final
# stretch); a fraction FRAC_VAL of those blocks, uniformly distributed,
# is reserved for validation. The difference with v8 is that here the windowing
# ALREADY happened in dataset_pred_filt.py (each sample k of X_hist/U_cand/Y_fut
# covers a range of T_IN+T_OUT original rows) -- so, instead of
# generating new windows, any sample k whose overlap margin
# (k up to k+T_IN+T_OUT) crosses a border between a
# train block and a val block is discarded, so that no pair shares original rows.
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
# 5. INSTANTIATION AND OPTIMIZER
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

# PINN criterion
criterion = PINNLossMPC(params_y=params_y).to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# =====================================================================
# 6. TRAINING
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

    # Compute PINN Loss
    total_loss, pure_mse = criterion(predictions, batch_y)
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    train_loss += total_loss.item() * len(batch_x)

  train_loss /= train_size

  # Validation
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
