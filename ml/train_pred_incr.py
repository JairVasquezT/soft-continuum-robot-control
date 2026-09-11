import argparse
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
# NOTA: 'corto' (grabación independiente) YA NO se usa acá para validar --
# se reserva 100% como test ciego final (dataset_valid_pred_v8.py), sin
# tocarse nunca durante el entrenamiento. Antes se usaba como val_loader
# Y como "test" reportado al final, lo cual sesga esa métrica (el checkpoint
# se elegía justamente por rendir bien ahí). La validación ahora sale de
# 'largo' mismo, con el mismo split por bloques distribuidos que
# train_v8_time_100_2.py (ver sección 4).
PATH_DATASET = 'dataset_pred_v08_completo_10_conRot_directo_filt.npy'
PATH_METADATA = 'dataset_pred_v08_completo_10_conRot_directo_filt_params.json'
# NOTA: este script predice DESPLAZAMIENTOS INCREMENTALES relativos a t0 (ver
# PINNLossMPC más abajo), por lo que el .npy necesita la clave 'Y_t0' --
# regenerar con dataset_pred_filt.py (ya actualizado para guardarla) si el
# archivo existente es de antes de este cambio.

# Ganancias de PINNLossMPC parametrizables por CLI (ver run_experimentos_pesos_loss.py
# para el barrido de 16 combinaciones) -- los defaults reproducen el
# comportamiento actual si se corre el script sin argumentos.
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
      w_mse=1000.0,
      w_direction_3d=20.0,
      w_sph_max=00.0,
      w_sph_min=00.0,
      w_cyl_max=00.0,
      w_speed=0.0,
      w_smooth=0.2,
      w_quat=0.0,
      max_step_dist=0.003,  # 1.0 cm por paso (16.7 ms)
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
    # Desescala una POSICIÓN ABSOLUTA en [-1,1] a metros reales (lleva el
    # offset/bias "+1"+min propio de esa escala).
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
    # Desescala un DESPLAZAMIENTO (diferencia entre dos posiciones ya en
    # [-1,1]) a metros reales -- SIN el offset/bias de desescalar_xyz: un
    # delta no tiene "origen" propio. Si escalado = 2*real/(max-min),
    # entonces real = escalado*(max-min)/2. delta_scaled_xyz: [..., 3].
    dx = delta_scaled_xyz[..., 0] * (self.max_x - self.min_x) / 2.0
    dy = delta_scaled_xyz[..., 1] * (self.max_y - self.min_y) / 2.0
    dz = delta_scaled_xyz[..., 2] * (self.max_z - self.min_z) / 2.0
    return torch.stack([dx, dy, dz], dim=-1)

  def forward(self, y_pred, y_true, y_t0):
    # y_pred: salida cruda de la red, [batch, T_OUT, output_dim] -- los
    # canales [:3] son el DESPLAZAMIENTO INCREMENTAL (ΔX, ΔY, ΔZ) predicho
    # respecto a t0, en el mismo espacio escalado [-1,1] que la posición
    # absoluta (ver desescalar_delta_xyz). y_true: Y_fut tal cual sale del
    # dataset (posición ABSOLUTA escalada). y_t0: posición/orientación
    # ABSOLUTA escalada en t0 (último paso del historial), [batch,
    # output_dim] -- es dato conocido (ground truth), nunca predicho, así
    # que el "error" en t0 es 0 por construcción: la red ya no tiene que
    # redescubrir la posición actual a partir del historial, solo el
    # movimiento incremental desde ahí, un objetivo de escala mucho menor
    # que la posición absoluta.
    y_true_delta_xyz = y_true[:, :, :3] - y_t0[:, :3].unsqueeze(1)

    # 1. MSE Ponderado por Eje sobre el DELTA escalado
    error_xyz_escalado = (y_pred[:, :, :3] - y_true_delta_xyz) ** 2
    loss_mse_ponderado = torch.mean(error_xyz_escalado * self.pesos_ejes_mse)

    # 2. Desescalar los deltas predicho y real a metros reales
    delta_pred = self.desescalar_delta_xyz(y_pred[:, :, :3])
    delta_true = self.desescalar_delta_xyz(y_true_delta_xyz)

    # 3. Reconstruir la posición ABSOLUTA real sumando t0 -- necesaria para
    # las penalizaciones geométricas, que son límites físicos fijos del
    # robot en el marco de la base (no relativos a t0).
    x_t0, y_t0_real, z_t0 = self.desescalar_xyz(y_t0.unsqueeze(1))
    pos_t0_real = torch.stack([x_t0, y_t0_real, z_t0], dim=-1)  # [batch, 1, 3]

    pos_pred = pos_t0_real + delta_pred
    pos_true = pos_t0_real + delta_true

    x_pred, y_pred_real, z_pred = pos_pred[..., 0], pos_pred[..., 1], pos_pred[..., 2]
    x_true, y_true_real, z_true = pos_true[..., 0], pos_true[..., 1], pos_true[..., 2]

    # 4. Dirección en 3D -- el offset de t0 se cancela entre pasos
    # consecutivos, así que da igual usar pos_pred/pos_true o delta_pred/
    # delta_true directamente; se usa pos_* para no romper el resto de la
    # fórmula tal cual estaba.
    dir_pred_3d = pos_pred[:, 1:, :] - pos_pred[:, :-1, :]
    dir_true_3d = pos_true[:, 1:, :] - pos_true[:, :-1, :]

    norm_true_3d = torch.norm(dir_true_3d, dim=-1)
    mask_movimiento = norm_true_3d > 0.0005  # Movimientos > 0.5 mm

    cos_sim_3d = torch.nn.functional.cosine_similarity(
        dir_pred_3d, dir_true_3d, dim=-1, eps=1e-6
    )
    loss_dir_raw = 1.0 - cos_sim_3d

    if mask_movimiento.sum() > 0:
      loss_direction_3d = torch.mean(loss_dir_raw[mask_movimiento])
    else:
      loss_direction_3d = torch.tensor(0.0, device=y_pred.device)

    # 4. Geometría Espacial
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

    # 5. Velocidad y Suavizado
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

    # Pérdida Total PINN
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
# 3. DATASET Y MODELO (IGUAL QUE ANTES)
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
# 6. ENTRENAMIENTO Y VALIDACIÓN ALINEADOS
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

  # Validación alineada con PINN Loss
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

  # El Scheduler y el Guardado responden a la Pérdida Total PINN
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
            # Los 3 primeros canales de la salida son un DESPLAZAMIENTO
            # incremental (ΔX,ΔY,ΔZ) relativo a t0, no una posición
            # absoluta -- cualquier script de evaluación debe sumarle la
            # posición real en t0 antes de comparar contra la trayectoria
            # absoluta (ver desescalar_delta_xyz/pos_t0_real en este archivo).
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