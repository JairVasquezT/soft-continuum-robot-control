import argparse
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
# NOTE: 'corto' (independent recording) is NO LONGER used here for validation --
# it is reserved 100% as the final blind test (dataset_valid_pred_v8.py), without
# ever being touched during training. It used to be used as val_loader
# AND as the "test" reported at the end, which biases that metric (the checkpoint
# was chosen precisely for performing well there). Validation now comes from
# 'largo' itself, with the same distributed-block split as
# train_v8_time_100_2.py (see section 4).
PATH_DATASET = 'dataset_pred_v08_completo_10_conRot_directo_filt.npy'
PATH_METADATA = 'dataset_pred_v08_completo_10_conRot_directo_filt_params.json'
# NOTE: this script predicts INCREMENTAL DISPLACEMENTS relative to t0 (see
# PINNLossMPC below), so the .npy needs the key 'Y_t0' --
# regenerate with dataset_pred_filt.py (already updated to save it) if the
# existing file predates this change.

# PINNLossMPC gains configurable via CLI (see run_experimentos_pesos_loss.py
# for the sweep of 16 combinations) -- the defaults reproduce the
# current behavior if the script is run without arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='best_mpc_pinn_predictor_incr_2.pth')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--w-mse', type=float, default=100.0)
parser.add_argument('--w-direction-3d', type=float, default=0.0)
parser.add_argument('--w-speed', type=float, default=0.0)
parser.add_argument('--w-smooth', type=float, default=0.0)
parser.add_argument('--max-step-dist', type=float, default=0.01,
                     help='Umbral de velocidad máxima por paso (m) que penaliza w_speed '
                          '(por defecto 0.01; train_pred_v2.py usaba 0.015).')
parser.add_argument('--pesos-ejes-mse', type=float, nargs=3, default=[1.5, 1.0, 2.2],
                     metavar=('PESO_X', 'PESO_Y', 'PESO_Z'))
args = parser.parse_args()

MODEL_SAVE_PATH = args.output

BATCH_SIZE = 128
EPOCHS = args.epochs
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
      w_mse=1000.0,
      w_direction_3d=20.0,
      w_sph_max=00.0,
      w_sph_min=00.0,
      w_cyl_max=00.0,
      w_speed=0.0,
      w_smooth=0.2,
      w_quat=0.0,
      max_step_dist=0.003,  # 1.0 cm per step (16.7 ms)
      pesos_ejes_mse=(2.0, 1.0, 2.0),  # [X, Y, Z]
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
    self.register_buffer(
        'pesos_ejes_mse', torch.tensor(pesos_ejes_mse, dtype=torch.float32)
    )

    self.mse = nn.MSELoss()

    self.min_x = params_y['rel_x']['min_t']
    self.max_x = params_y['rel_x']['max_t']
    self.min_y = params_y['rel_y']['min_t']
    self.max_y = params_y['rel_y']['max_t']
    self.min_z = params_y['rel_z']['min_t']
    self.max_z = params_y['rel_z']['max_t']

  def desescalar_xyz(self, y_scaled):
    # Unscales an ABSOLUTE POSITION in [-1,1] to real meters (it carries the
    # "+1"+min offset/bias specific to that scale).
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

  def desescalar_delta_xyz(self, delta_scaled_xyz):
    # Unscales a DISPLACEMENT (difference between two positions already in
    # [-1,1]) to real meters -- WITHOUT the offset/bias of desescalar_xyz: a
    # delta has no "origin" of its own. If scaled = 2*real/(max-min),
    # then real = scaled*(max-min)/2. delta_scaled_xyz: [..., 3].
    dx = delta_scaled_xyz[..., 0] * (self.max_x - self.min_x) / 2.0
    dy = delta_scaled_xyz[..., 1] * (self.max_y - self.min_y) / 2.0
    dz = delta_scaled_xyz[..., 2] * (self.max_z - self.min_z) / 2.0
    return torch.stack([dx, dy, dz], dim=-1)

  def forward(self, y_pred, y_true, y_t0):
    # y_pred: raw network output, [batch, T_OUT, output_dim] -- the
    # channels [:3] are the predicted INCREMENTAL DISPLACEMENT (ΔX, ΔY, ΔZ)
    # relative to t0, in the same scaled space [-1,1] as the absolute
    # position (see desescalar_delta_xyz). y_true: Y_fut exactly as it comes from the
    # dataset (scaled ABSOLUTE position). y_t0: scaled ABSOLUTE
    # position/orientation at t0 (last step of the history), [batch,
    # output_dim] -- it is known data (ground truth), never predicted, so
    # the "error" at t0 is 0 by construction: the network no longer has to
    # rediscover the current position from the history, only the
    # incremental motion from there, a target of much smaller scale
    # than the absolute position.
    y_true_delta_xyz = y_true[:, :, :3] - y_t0[:, :3].unsqueeze(1)

    # 1. Axis-Weighted MSE over the scaled DELTA
    error_xyz_escalado = (y_pred[:, :, :3] - y_true_delta_xyz) ** 2
    loss_mse_ponderado = torch.mean(error_xyz_escalado * self.pesos_ejes_mse)

    # 2. Unscale the predicted and real deltas to real meters
    delta_pred = self.desescalar_delta_xyz(y_pred[:, :, :3])
    delta_true = self.desescalar_delta_xyz(y_true_delta_xyz)

    # 3. Reconstruct the real ABSOLUTE position by adding t0 -- needed for
    # the geometric penalties, which are fixed physical limits of the
    # robot in the base frame (not relative to t0).
    x_t0, y_t0_real, z_t0 = self.desescalar_xyz(y_t0.unsqueeze(1))
    pos_t0_real = torch.stack([x_t0, y_t0_real, z_t0], dim=-1)  # [batch, 1, 3]

    pos_pred = pos_t0_real + delta_pred
    pos_true = pos_t0_real + delta_true

    x_pred, y_pred_real, z_pred = pos_pred[..., 0], pos_pred[..., 1], pos_pred[..., 2]
    x_true, y_true_real, z_true = pos_true[..., 0], pos_true[..., 1], pos_true[..., 2]

    # 4. Direction in 3D -- the t0 offset cancels between
    # consecutive steps, so it makes no difference whether pos_pred/pos_true or delta_pred/
    # delta_true are used directly; pos_* is used so as not to break the rest of the
    # formula as it was.
    dir_pred_3d = pos_pred[:, 1:, :] - pos_pred[:, :-1, :]
    dir_true_3d = pos_true[:, 1:, :] - pos_true[:, :-1, :]

    norm_true_3d = torch.norm(dir_true_3d, dim=-1)
    mask_movimiento = norm_true_3d > 0.0005  # Movements > 0.5 mm

    cos_sim_3d = torch.nn.functional.cosine_similarity(
        dir_pred_3d, dir_true_3d, dim=-1, eps=1e-6
    )
    loss_dir_raw = 1.0 - cos_sim_3d

    if mask_movimiento.sum() > 0:
      loss_direction_3d = torch.mean(loss_dir_raw[mask_movimiento])
    else:
      loss_direction_3d = torch.tensor(0.0, device=y_pred.device)

    # 4. Spatial Geometry
    dist_esferica_cuadrada = x_pred**2 + y_pred_real**2 + z_pred**2
    dist_cilindrica_cuadrada = x_pred**2 + z_pred**2

    pen_sph_max = torch.relu(dist_esferica_cuadrada - 0.335**2)
    pen_sph_min = torch.relu(0.22**2 - dist_esferica_cuadrada)
    pen_cyl_max = torch.relu(dist_cilindrica_cuadrada - 0.23**2)

    loss_geom = (
        self.w_sph_max * torch.mean(pen_sph_max)
        + self.w_sph_min * torch.mean(pen_sph_min)
        + self.w_cyl_max * torch.mean(pen_cyl_max)
    )

    # 5. Velocity and Smoothing
    paso_diffs = pos_pred[:, 1:, :] - pos_pred[:, :-1, :]
    dist_por_paso = torch.norm(paso_diffs, dim=-1)

    pen_velocidad = torch.relu(dist_por_paso - self.max_step_dist)
    loss_speed = self.w_speed * torch.mean(pen_velocidad)

    aceleracion = paso_diffs[:, 1:, :] - paso_diffs[:, :-1, :]
    loss_smooth = self.w_smooth * torch.mean(aceleracion**2)

    loss_quat = 0.0
    if y_pred.shape[-1] == 7:
      q = y_pred[:, :, 3:]
      norm_q = torch.norm(q, dim=-1)
      loss_quat = self.w_quat * torch.mean((norm_q - 1.0) ** 2)

    # Total PINN Loss
    total_loss = (
        self.w_mse * loss_mse_ponderado
        + self.w_direction_3d * loss_direction_3d
        + loss_geom
        + loss_speed
        + loss_smooth
        + loss_quat
    )
    return total_loss, loss_mse_ponderado


# =====================================================================
# 3. DATASET AND MODEL (SAME AS BEFORE)
# =====================================================================
class MPCDataset(Dataset):

  def __init__(self, npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    self.x_hist = torch.tensor(data['X_hist'], dtype=torch.float32)
    self.u_cand = torch.tensor(data['U_cand'], dtype=torch.float32)
    self.y_fut = torch.tensor(data['Y_fut'], dtype=torch.float32)
    self.y_t0 = torch.tensor(data['Y_t0'], dtype=torch.float32)

  def __len__(self):
    return len(self.x_hist)

  def __getitem__(self, idx):
    return self.x_hist[idx], self.u_cand[idx], self.y_fut[idx], self.y_t0[idx]


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
sample_x, sample_u, sample_y, sample_y_t0 = dataset_completo[0]
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
criterion = PINNLossMPC(
    params_y=params_y,
    w_mse=args.w_mse,
    w_direction_3d=args.w_direction_3d,
    w_speed=args.w_speed,
    w_smooth=args.w_smooth,
    max_step_dist=args.max_step_dist,
    pesos_ejes_mse=tuple(args.pesos_ejes_mse),
).to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
)

# =====================================================================
# 6. ALIGNED TRAINING AND VALIDATION
# =====================================================================
best_val_loss = float('inf')
history_train_loss, history_val_loss = [], []

print('\n🔥 Iniciando entrenamiento con PINN Loss...\n')

for epoch in range(1, EPOCHS + 1):
  model.train()
  train_loss = 0.0

  for batch_x, batch_u, batch_y, batch_y_t0 in train_loader:
    batch_x, batch_u, batch_y, batch_y_t0 = (
        batch_x.to(DEVICE),
        batch_u.to(DEVICE),
        batch_y.to(DEVICE),
        batch_y_t0.to(DEVICE),
    )

    optimizer.zero_grad()
    predictions = model(batch_x, batch_u)

    total_loss, _ = criterion(predictions, batch_y, batch_y_t0)
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    train_loss += total_loss.item() * len(batch_x)

  train_loss /= train_size

  # Validation aligned with PINN Loss
  model.eval()
  val_loss_sum = 0.0
  val_mse_sum = 0.0

  with torch.no_grad():
    for batch_x, batch_u, batch_y, batch_y_t0 in val_loader:
      batch_x, batch_u, batch_y, batch_y_t0 = (
          batch_x.to(DEVICE),
          batch_u.to(DEVICE),
          batch_y.to(DEVICE),
          batch_y_t0.to(DEVICE),
      )

      predictions = model(batch_x, batch_u)
      total_loss, pure_mse_pond = criterion(predictions, batch_y, batch_y_t0)

      val_loss_sum += total_loss.item() * len(batch_x)
      val_mse_sum += pure_mse_pond.item() * len(batch_x)

  val_loss = val_loss_sum / val_size
  val_mse = val_mse_sum / val_size

  # The Scheduler and Saving respond to the Total PINN Loss
  scheduler.step(val_loss)

  history_train_loss.append(train_loss)
  history_val_loss.append(val_loss)

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'input_hist_dim': input_hist_dim,
            'u_cand_dim': len(sample_u),
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            't_out': t_out,
            'output_dim': output_dim,
            'best_val_loss': best_val_loss,
            'best_val_mse': val_mse,
            # The first 3 output channels are an incremental
            # DISPLACEMENT (ΔX,ΔY,ΔZ) relative to t0, not an
            # absolute position -- any evaluation script must add the
            # real position at t0 before comparing against the
            # absolute trajectory (see desescalar_delta_xyz/pos_t0_real in this file).
            'predice_incrementos': True,
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
        f' | Val PINN Loss: {val_loss:.6f} | Val MSE: {val_mse:.6f} | LR:'
        f' {current_lr:.2e} {saved_flag}'
    )

print(f'\n✅ Entrenamiento completado. Mejor Val PINN Loss: {best_val_loss:.6f}')