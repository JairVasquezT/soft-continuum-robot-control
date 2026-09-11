import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=90)
parser.add_argument('--output', type=str, default='soft_robot_lstm_v7_11_90win.pth')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.0005)
args = parser.parse_args()

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS (MODELO V7 SINGLE-STREAM)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(
    '🔥 Entrenando Modelo V7 Single-Stream (13 Entradas -> 3 Salidas) en:'
    f' {device}'
)

PATH_DATASET_NPY = 'dataset_v07_real_meta_tension_sinRot_filt.npy'
PATH_DATASET_JSON = 'dataset_v07_real_meta_tension_sinRot_filt_params.json'
PATH_PESOS_SALIDA = args.output

# Cargar matrices escaladas [-1, 1]
data_v7 = np.load(PATH_DATASET_NPY, allow_pickle=True).item()
X_raw = data_v7['X']  # shape: (N, 13) -> 9 cinemáticos + 4 tensiones
Y_raw = data_v7['Y']  # shape: (N, 3)  -> rel_x, rel_y, rel_z

with open(PATH_DATASET_JSON, 'r') as f:
  norm_params = json.load(f)

Y_TRANS = norm_params['Y_transformer']

print(
    f'📊 Características detectadas -> Entradas (X): {X_raw.shape[1]} |'
    f' Salidas (Y): {Y_raw.shape[1]}'
)

# ==========================================
# 2. VENTANADO CON SPLIT TRAIN/VAL REPARTIDO Y SIN FUGA
# ==========================================
# La grabación se divide en N_BLOQUES contiguos a lo largo de TODA la
# secuencia (no solo el tramo final); una fracción FRAC_VAL de esos
# bloques, repartidos uniformemente, se reserva para validación. Cualquier
# ventana cuyo rango de filas cruce el borde entre un bloque de train y uno
# de val se descarta por completo, así ningún par comparte un solo paso
# entre ambos conjuntos (misma estrategia que train_v8_time_100_2.py).
WINDOW_SIZE = args.window_size
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


split = crear_secuencias_split(X_raw, Y_raw, WINDOW_SIZE)
X_train, Y_train = split['train']
X_val, Y_val = split['val']

print(f'📦 Muestras de entrenamiento: {len(X_train)} | Muestras de validación: {len(X_val)}')

# DataLoaders independientes
train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    ),
    batch_size=64,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32),
    ),
    batch_size=64,
    shuffle=False,  # En validación se mantiene el orden sin shuffle
)


# ==========================================
# 3. PÉRDIDA FÍSICA FUSIONADA (PINN LOSS)
# ==========================================
class RobotBlandoLoss(nn.Module):

  def __init__(
      self,
      y_params,
      w_pinn=0.05,
      w_sph_max=1.0,
      w_sph_min=1.0,
      w_cyl_max=1.0,
  ):
    super(RobotBlandoLoss, self).__init__()
    self.base_loss = nn.HuberLoss(delta=0.01)
    self.w_pinn = w_pinn
    self.w_sph_max = w_sph_max
    self.w_sph_min = w_sph_min
    self.w_cyl_max = w_cyl_max

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
    loss_base = self.base_loss(y_pred, y_true)

    # Desescalar predicciones a metros
    x_real = self.min_x + (y_pred[:, 0] + 1.0) * (self.max_x - self.min_x) / 2.0
    y_real = self.min_y + (y_pred[:, 1] + 1.0) * (self.max_y - self.min_y) / 2.0
    z_real = self.min_z + (y_pred[:, 2] + 1.0) * (self.max_z - self.min_z) / 2.0

    dist_esferica_cuadrada = x_real**2 + y_real**2 + z_real**2
    dist_cilindrica_cuadrada = x_real**2 + z_real**2

    pen_sph_max = torch.relu(dist_esferica_cuadrada - 0.335**2)
    pen_sph_min = torch.relu(0.22**2 - dist_esferica_cuadrada)
    pen_cyl_max = torch.relu(dist_cilindrica_cuadrada - 0.23**2)

    loss_pinn = (
        self.w_sph_max * torch.mean(pen_sph_max)
        + self.w_sph_min * torch.mean(pen_sph_min)
        + self.w_cyl_max * torch.mean(pen_cyl_max)
    )

    return loss_base + self.w_pinn * loss_pinn


criterion = RobotBlandoLoss(y_params=Y_TRANS, w_pinn=0.05)


# ==========================================
# 4. ARQUITECTURA SINGLE-STREAM LSTM (V7)
# ==========================================
class V7SingleStreamLSTM(nn.Module):

  def __init__(
      self,
      input_size=13,
      hidden_size=128,
      num_layers=2,
      output_size=3,
      dropout=0.2,
  ):
    super(V7SingleStreamLSTM, self).__init__()

    # LSTM unificada para las 13 características de entrada
    self.lstm = nn.LSTM(
        input_size,
        hidden_size,
        num_layers=num_layers,
        batch_first=True,
    )

    self.dropout = nn.Dropout(dropout)

    # Capas totalmente conectadas de salida
    self.fc = nn.Sequential(
        nn.Linear(hidden_size, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, output_size),
    )

  def forward(self, x):
    # x shape: (batch_size, window_size, 13)
    out, _ = self.lstm(x)

    # Tomar el último paso temporal de la secuencia
    last_step = out[:, -1, :]

    return self.fc(last_step)


model = V7SingleStreamLSTM(
    input_size=13,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    output_size=3,
    dropout=args.dropout,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

# SCHEDULER AJUSTADO
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, threshold=1e-5, min_lr=1e-5
)

# ==========================================
# 5. BUCLE DE ENTRENAMIENTO Y VALIDACIÓN
# ==========================================
EPOCHS = 100
best_val_loss = float('inf')

print(
    f'🚀 Iniciando entrenamiento V7 Single-Stream (Ventana={WINDOW_SIZE} | 13'
    ' Entradas)...'
)

for epoch in range(EPOCHS):
  # --- FASE 1: ENTRENAMIENTO ---
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

  # --- FASE 2: VALIDACIÓN ---
  model.eval()
  val_loss = 0.0

  with torch.no_grad():
    for batch_X, batch_y in val_loader:
      batch_X, batch_y = batch_X.to(device), batch_y.to(device)

      predictions = model(batch_X)
      loss = criterion(predictions, batch_y)

      val_loss += loss.item()

  avg_val_loss = val_loss / len(val_loader)

  # SCHEDULER EVALÚA PÉRDIDA DE VALIDACIÓN
  scheduler.step(avg_val_loss)

  # CHECKPOINT MEJOR MODELO
  if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    torch.save(model.state_dict(), PATH_PESOS_SALIDA)

  if (epoch + 1) % 10 == 0:
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f'Época [{epoch+1}/{EPOCHS:03d}] -> Train Loss: {avg_train_loss:.6f} |'
        f' Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}'
    )

print(
    '\n🧠 Entrenamiento V7 Single-Stream completado. Mejor modelo guardado con'
    f' Val Loss: {best_val_loss:.6f} en {PATH_PESOS_SALIDA}'
)