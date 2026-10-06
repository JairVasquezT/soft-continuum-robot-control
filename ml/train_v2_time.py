import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=90)
parser.add_argument('--output', type=str, default='soft_robot_lstm_v2_13_time_best.pth')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.00025)
args = parser.parse_args()

# ==========================================
# 1. CONFIGURATION AND DATA LOADING (VERSION V2)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔥 Entrenando en el dispositivo: {device}')

# Load the v2 dataset (9 input columns)
data_v2 = np.load(
    'dataset_v02_real_meta_sinRot_filt.npy', allow_pickle=True
).item()
X_raw = data_v2['X']  # shape: (N, 9)
Y_raw = data_v2['Y']  # shape: (N, 3) -> rel_x, rel_y, rel_z

# Load the JSON normalization parameters
with open('dataset_v02_real_meta_sinRot_filt_params.json', 'r') as f:
  norm_params = json.load(f)

X_TRANS = norm_params['X_transformer']
Y_TRANS = norm_params['Y_transformer']


# ==========================================
# 2. WINDOWING WITH DISTRIBUTED, LEAK-FREE TRAIN/VAL SPLIT
# ==========================================
# The recording is divided into N_BLOQUES contiguous blocks across the WHOLE
# sequence (not just the final stretch); a fraction FRAC_VAL of those
# blocks, uniformly distributed, is reserved for validation. Any
# window whose row range crosses the border between a train block and a
# val block is discarded entirely, so no pair shares a single step
# between the two sets (same strategy as train_v8_time_100_2.py).
FRAC_VAL = 0.10
N_BLOQUES = 20


def crear_secuencias_split(X, Y, window_size=90, frac_val=FRAC_VAL, n_bloques=N_BLOQUES):
  n_total = len(X)
  tam_bloque = n_total // n_bloques
  n_bloques_val = max(1, round(n_bloques * frac_val))
  paso = n_bloques / n_bloques_val
  bloques_val = {
      int(round(paso / 2 + i * paso)) % n_bloques for i in range(n_bloques_val)
  }

  def bloque_de(idx_fila):
    return min(idx_fila // tam_bloque, n_bloques - 1)

  datos = {'train': ([], []), 'val': ([], [])}
  descartadas = 0

  for i in range(n_total - window_size):
    bloque_inicio = bloque_de(i)
    bloque_fin = bloque_de(i + window_size)
    if bloque_inicio != bloque_fin:
      descartadas += 1
      continue

    destino = 'val' if bloque_inicio in bloques_val else 'train'
    X_l, Y_l = datos[destino]
    X_l.append(X[i : i + window_size])
    Y_l.append(Y[i + window_size])

  print(f'   (bloques de validación repartidos: {sorted(bloques_val)}/{n_bloques} '
        f'| {descartadas} ventanas descartadas por cruzar un borde de bloque)')

  return {
      'train': (np.array(datos['train'][0]), np.array(datos['train'][1])),
      'val': (np.array(datos['val'][0]), np.array(datos['val'][1])),
  }


WINDOW_SIZE = args.window_size
split = crear_secuencias_split(X_raw, Y_raw, WINDOW_SIZE)
X_train, y_train = split['train']
X_val, y_val = split['val']

print(f'📦 Muestras de entrenamiento: {len(X_train)} | Muestras de validación: {len(X_val)}')

# Separate DataLoaders
train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    ),
    batch_size=64, shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    ),
    batch_size=64, shuffle=False,
)


# ==========================================
# 3. PHYSICALLY INSPIRED LOSS FUNCTION (PINN CUSTOM LOSS)
# ==========================================
class RobotBlandoLoss(nn.Module):

  def __init__(
      self,
      y_params,
      w_pinn=0.1,
      w_sph_max=1.0,
      w_sph_min=1.0,
      w_cyl_max=1.0,
  ):
    super(RobotBlandoLoss, self).__init__()
    self.base_loss = nn.HuberLoss()  # Robust base loss
    self.w_pinn = w_pinn
    self.w_sph_max = w_sph_max
    self.w_sph_min = w_sph_min
    self.w_cyl_max = w_cyl_max

    # Save unscaling constants for rel_x, rel_y, rel_z
    self.min_x, self.max_x = (
        y_params['rel_x']['min_t'],
        y_params['rel_x']['max_t'],
    )
    self.min_y, self.max_y = (
        y_params['rel_y']['min_t'],
        y_params['rel_y']['max_t'],
    )
    self.min_z, self.max_z = (
        y_params['rel_z']['min_t'],
        y_params['rel_z']['max_t'],
    )

  def forward(self, y_pred, y_true):
    # 1. Standard loss in the normalized space [-1, 1]
    loss_base = self.base_loss(y_pred, y_true)

    # 2. Unscale predictions to real physical dimensions
    x_real = self.min_x + (y_pred[:, 0] + 1.0) * (self.max_x - self.min_x) / 2.0
    y_real = self.min_y + (y_pred[:, 1] + 1.0) * (self.max_y - self.min_y) / 2.0
    z_real = self.min_z + (y_pred[:, 2] + 1.0) * (self.max_z - self.min_z) / 2.0

    # 3. Real geometric constraints
    dist_esferica_cuadrada = x_real**2 + y_real**2 + z_real**2
    dist_cilindrica_cuadrada = x_real**2 + z_real**2

    # A) Maximum Sphere: r <= 0.335 m
    pen_sph_max = torch.relu(dist_esferica_cuadrada - 0.335**2)

    # B) Minimum Sphere: r >= 0.22 m
    pen_sph_min = torch.relu(0.22**2 - dist_esferica_cuadrada)

    # C) Maximum Cylinder: r_cyl <= 0.23 m
    pen_cyl_max = torch.relu(dist_cilindrica_cuadrada - 0.23**2)

    # Average PINN penalty
    loss_pinn = (
        self.w_sph_max * torch.mean(pen_sph_max)
        + self.w_sph_min * torch.mean(pen_sph_min)
        + self.w_cyl_max * torch.mean(pen_cyl_max)
    )

    return loss_base + self.w_pinn * loss_pinn


criterion = RobotBlandoLoss(y_params=Y_TRANS, w_pinn=0.05)


# ==========================================
# 4. LSTM ARCHITECTURE
# ==========================================
class SoftRobotLSTM(nn.Module):

  def __init__(
      self,
      input_size=9,
      hidden_size=128,
      num_layers=2,
      output_size=3,
      dropout=0.2,
  ):
    super(SoftRobotLSTM, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # PyTorch handles the h0 and c0 states automatically if omitted
    out, _ = self.lstm(x)
    last_step = out[:, -1, :]
    out_regularized = self.dropout(last_step)
    return self.fc(out_regularized)


# Instantiate Model, Optimizer and SCHEDULER
model = SoftRobotLSTM(
    input_size=9, hidden_size=args.hidden_size, num_layers=args.num_layers,
    output_size=3, dropout=args.dropout,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# SCHEDULER: Reduces the LR if val_loss does not improve after 8 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
)

# ==========================================
# 5. TRAINING AND VALIDATION LOOP
# ==========================================
PATH_PESOS_SALIDA = args.output
EPOCHS = 100
best_val_loss = float('inf')

print(
    '🚀 Iniciando entrenamiento del modelo V2 con Física, Scheduler y'
    ' Validación...'
)

for epoch in range(EPOCHS):
  # --- TRAINING PHASE ---
  model.train()
  train_loss = 0.0
  for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    optimizer.zero_grad()
    predictions = model(batch_X)
    loss = criterion(predictions, batch_y)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    train_loss += loss.item()

  avg_train_loss = train_loss / len(train_loader)

  # --- VALIDATION PHASE ---
  model.eval()
  val_loss = 0.0
  with torch.no_grad():
    for batch_X_v, batch_y_v in val_loader:
      batch_X_v, batch_y_v = batch_X_v.to(device), batch_y_v.to(device)
      preds_v = model(batch_X_v)
      loss_v = criterion(preds_v, batch_y_v)
      val_loss += loss_v.item()

  avg_val_loss = val_loss / len(val_loader)

  # Update the Scheduler using the VALIDATION LOSS
  scheduler.step(avg_val_loss)

  # Save only the BEST model according to the validation loss
  if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    torch.save(model.state_dict(), PATH_PESOS_SALIDA)

  # Print progress every 5 epochs and show the current LR
  if (epoch + 1) % 5 == 0 or epoch == 0:
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f'Época [{epoch+1:03d}/{EPOCHS}] -> Loss Train: {avg_train_loss:.5f} |'
        f' Loss Val: {avg_val_loss:.5f} | LR: {current_lr:.6f}'
    )

print(
    f'🧠 Entrenamiento finalizado. El mejor modelo se guardó con Val Loss:'
    f' {best_val_loss:.5f}'
)