import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS (MODELO V7)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔥 Entrenando Modelo V7 (13 Entradas -> 3 Salidas) en: {device}')

PATH_DATASET_NPY = 'dataset_v07_real_meta_tension_sinRot_filt.npy'
PATH_DATASET_JSON = 'dataset_v07_real_meta_tension_sinRot_filt_params.json'
PATH_PESOS_SALIDA = 'soft_robot_lstm_v7_5_45win.pth'

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
# 2. CREACIÓN DE VENTANAS Y DIVISIÓN 85% TRAIN / 15% VAL
# ==========================================
WINDOW_SIZE = 45


def crear_secuencias(X, Y, window_size=45):
  X_seq, Y_seq = [], []
  for i in range(len(X) - window_size):
    X_seq.append(X[i : i + window_size])
    Y_seq.append(Y[i + window_size])
  return np.array(X_seq), np.array(Y_seq)


X_windows, Y_windows = crear_secuencias(X_raw, Y_raw, window_size=WINDOW_SIZE)

# 🆕 DIVISIÓN SECUENCIAL (85% Train / 15% Validation)
val_size = int(len(X_windows) * 0.15)
train_size = len(X_windows) - val_size

X_train, Y_train = X_windows[:train_size], Y_windows[:train_size]
X_val, Y_val = X_windows[train_size:], Y_windows[train_size:]

print(f'📊 Ventanas creadas: Total = {len(X_windows)}')
print(f' ▫ Entrenando con: {len(X_train)} muestras (85%)')
print(f' ▫ Validando con:  {len(X_val)} muestras (15%)')

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
# 4. ARQUITECTURA MULTIRRAMA (DUAL-STREAM V7)
# ==========================================
class V7DualStreamLSTM(nn.Module):

  def __init__(
      self,
      kin_input_size=9,
      tens_input_size=4,
      kin_hidden=96,
      tens_hidden=48,
      output_size=3,
      dropout=0.2,
  ):
    super(V7DualStreamLSTM, self).__init__()

    # Rama Cinemática: 9 entradas (delta_real_m1..4, delta_t, delta_meta_m1..4)
    self.lstm_kin = nn.LSTM(
        kin_input_size, kin_hidden, num_layers=2, batch_first=True
    )

    # Rama Dinámica: 4 entradas (tension_m1..4)
    self.lstm_tens = nn.LSTM(
        tens_input_size, tens_hidden, num_layers=1, batch_first=True
    )

    self.dropout = nn.Dropout(dropout)

    # Capa de Fusión
    self.fc = nn.Sequential(
        nn.Linear(kin_hidden + tens_hidden, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, output_size),
    )

  def forward(self, x_kin, x_tens):
    out_kin, _ = self.lstm_kin(x_kin)
    out_tens, _ = self.lstm_tens(x_tens)

    last_kin = out_kin[:, -1, :]
    last_tens = out_tens[:, -1, :]

    fusion = torch.cat((last_kin, last_tens), dim=1)
    return self.fc(fusion)


model = V7DualStreamLSTM(
    kin_input_size=9,
    tens_input_size=4,
    kin_hidden=96,
    tens_hidden=48,
    output_size=3,
    dropout=0.2,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# 🆕 SCHEDULER AJUSTADO (Patience=12, min_lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, threshold=1e-5, min_lr=1e-5
)

# ==========================================
# 5. BUCLE DE ENTRENAMIENTO CON VALIDACIÓN Y BEST MODEL CHECKPOINT
# ==========================================
EPOCHS = 160
best_val_loss = float('inf')

print(
    f'🚀 Iniciando entrenamiento V7 (Ventana={WINDOW_SIZE} | 9 Cinemática + 4'
    ' Tensiones)...'
)

for epoch in range(EPOCHS):
  # --- FASE 1: ENTRENAMIENTO ---
  model.train()
  train_loss = 0.0

  for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    # Separación de entradas V7: 0..8 (Cinemática), 9..12 (Tensiones)
    batch_X_kin = batch_X[:, :, :9]
    batch_X_tens = batch_X[:, :, 9:]

    optimizer.zero_grad()
    predictions = model(batch_X_kin, batch_X_tens)
    loss = criterion(predictions, batch_y)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    train_loss += loss.item()

  avg_train_loss = train_loss / len(train_loader)

  # --- FASE 2: VALIDACIÓN (Sin cálculo de gradientes) ---
  model.eval()
  val_loss = 0.0

  with torch.no_grad():
    for batch_X, batch_y in val_loader:
      batch_X, batch_y = batch_X.to(device), batch_y.to(device)

      batch_X_kin = batch_X[:, :, :9]
      batch_X_tens = batch_X[:, :, 9:]

      predictions = model(batch_X_kin, batch_X_tens)
      loss = criterion(predictions, batch_y)

      val_loss += loss.item()

  avg_val_loss = val_loss / len(val_loader)

  # 🆕 EL SCHEDULER EVALÚA LA PÉRDIDA DE VALIDACIÓN
  scheduler.step(avg_val_loss)

  # 🆕 CHECKPOINT: GUARDAR MEJOR MODELO SEGÚN VAL_LOSS
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
    f"\n🧠 Entrenamiento V7 completado. Mejor modelo guardado con Val Loss:"
    f' {best_val_loss:.6f} en {PATH_PESOS_SALIDA}'
)