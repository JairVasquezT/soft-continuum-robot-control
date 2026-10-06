import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. CONFIGURATION AND DATA LOADING (VERSION V8 COMPLETO)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(
    f"🔥 Entrenando Modelo V8 Completo (17 Entradas -> 3 Salidas) al 100% en:"
    f" {device}"
)

PATH_DATASET_NPY = 'dataset_v01_completo_sinRot_filt.npy'
PATH_DATASET_JSON = 'dataset_v01_completo_sinRot_filt_params.json'
PATH_DATASET_VAL_NPY = 'dataset_corto_v01_completo_sinRot_filt.npy'
PATH_PESOS_SALIDA = 'soft_robot_lstm_v8_17_time.pth'

# Load scaled matrices
data_v8 = np.load(PATH_DATASET_NPY, allow_pickle=True).item()
X_raw = data_v8['X']  # shape: (N, 17) -> 1 delta_t + 4 real_m + 4 meta_m + 4 couple + 4 tension
Y_raw = data_v8['Y']  # shape: (N, 3)  -> rel_x, rel_y, rel_z

data_v8_val = np.load(PATH_DATASET_VAL_NPY, allow_pickle=True).item()
X_raw_val = data_v8_val['X']
Y_raw_val = data_v8_val['Y']

# Load the JSON normalization parameters
with open(PATH_DATASET_JSON, 'r') as f:
  norm_params = json.load(f)

X_TRANS = norm_params['X_transformer']
Y_TRANS = norm_params['Y_transformer']

# Confirm the dynamic input dimension
num_features_input = X_raw.shape[1]
print(
    f"📊 Características detectadas en entrada (X): {num_features_input} |"
    f" Salidas (Y): {Y_raw.shape[1]}"
)


# ==========================================
# 2. CREATION OF TEMPORAL WINDOWS (100% OF THE DATA)
# ==========================================
def crear_secuencias(X, Y, window_size=90):
  X_seq, Y_seq = [], []
  for i in range(len(X) - window_size):
    X_seq.append(X[i : i + window_size])
    Y_seq.append(Y[i + window_size])
  return np.array(X_seq), np.array(Y_seq)


WINDOW_SIZE = 90
X_windows, Y_windows = crear_secuencias(X_raw, Y_raw, window_size=WINDOW_SIZE)
X_windows_val, Y_windows_val = crear_secuencias(
    X_raw_val, Y_raw_val, window_size=WINDOW_SIZE
)

print(f"📦 Total de Muestras para Entrenamiento (100% de 'largo'): {len(X_windows)}")
print(f"🧪 Muestras de Validación ('corto', independiente): {len(X_windows_val)}")

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_windows, dtype=torch.float32)
Y_tensor = torch.tensor(Y_windows, dtype=torch.float32)

X_val_tensor = torch.tensor(X_windows_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_windows_val, dtype=torch.float32)

# DataLoader with 100% of 'largo' for training
train_loader = DataLoader(
    TensorDataset(X_tensor, Y_tensor), batch_size=64, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_tensor, Y_val_tensor), batch_size=64, shuffle=False
)


# ==========================================
# 3. FUSED PHYSICS LOSS (PINN LOSS)
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
    # Huber Loss for high precision in sub-millimeter errors
    self.base_loss = nn.HuberLoss(delta=0.01)
    self.w_pinn = w_pinn
    self.w_sph_max = w_sph_max
    self.w_sph_min = w_sph_min
    self.w_cyl_max = w_cyl_max

    # Extract real limits from the JSON for the inverse transformation
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
    # 1. Base mathematical loss in scaled space [-1, 1]
    loss_base = self.base_loss(y_pred, y_true)

    # 2. Unscale predictions to physical coordinates (Meters)
    x_real = self.min_x + (y_pred[:, 0] + 1.0) * (self.max_x - self.min_x) / 2.0
    y_real = self.min_y + (y_pred[:, 1] + 1.0) * (self.max_y - self.min_y) / 2.0
    z_real = self.min_z + (y_pred[:, 2] + 1.0) * (self.max_z - self.min_z) / 2.0

    # 3. 3D-space geometric barriers
    dist_esferica_cuadrada = x_real**2 + y_real**2 + z_real**2
    dist_cilindrica_cuadrada = x_real**2 + z_real**2

    # Constraints
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
# 4. ADAPTABLE LSTM ARCHITECTURE
# ==========================================
class SoftRobotLSTM(nn.Module):

  def __init__(
      self,
      input_size=17,
      hidden_size=128,
      num_layers=2,
      output_size=3,
      dropout=0.2,
  ):
    super(SoftRobotLSTM, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(
        input_size,
        hidden_size,
        num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    h0 = torch.zeros(
        self.num_layers, x.size(0), self.hidden_size, device=x.device
    )
    c0 = torch.zeros(
        self.num_layers, x.size(0), self.hidden_size, device=x.device
    )

    out, _ = self.lstm(x, (h0, c0))
    last_step = out[:, -1, :]
    out_regularized = self.dropout(last_step)
    return self.fc(out_regularized)


# Instantiate the model with the 17 inputs of the V8 Dataset
model = SoftRobotLSTM(
    input_size=num_features_input,
    hidden_size=128,
    num_layers=2,
    output_size=3,
    dropout=0.2,
).to(device)

# AdamW optimizer and Scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
)

# ==========================================
# 5. TRAINING LOOP (100% of 'largo') + VALIDATION ON 'corto'
# ==========================================
EPOCHS = 160
history_loss = []
history_val_loss = []
best_val_loss = float('inf')

print(
    f"🚀 Iniciando entrenamiento de V8 Completo ({num_features_input}"
    f" Entradas) con Ventana = {WINDOW_SIZE}..."
)

for epoch in range(EPOCHS):
  model.train()
  epoch_loss = 0.0
  for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    optimizer.zero_grad()
    predictions = model(batch_X)
    loss = criterion(predictions, batch_y)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    epoch_loss += loss.item()

  avg_loss = epoch_loss / len(train_loader)
  history_loss.append(avg_loss)

  # --- VALIDATION on 'corto' (recording independent from 'largo') ---
  model.eval()
  val_epoch_loss = 0.0
  with torch.no_grad():
    for batch_X, batch_y in val_loader:
      batch_X, batch_y = batch_X.to(device), batch_y.to(device)
      predictions = model(batch_X)
      loss = criterion(predictions, batch_y)
      val_epoch_loss += loss.item()

  avg_val_loss = val_epoch_loss / len(val_loader)
  history_val_loss.append(avg_val_loss)

  # Update the learning rate and save the best checkpoint according to VAL
  scheduler.step(avg_val_loss)

  if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'input_size': num_features_input,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': Y_raw.shape[1],
            'dropout': 0.2,
            'window_size': WINDOW_SIZE,
            'best_val_loss': best_val_loss,
        },
        PATH_PESOS_SALIDA,
    )
    saved_flag = '⭐ [Guardado]'
  else:
    saved_flag = ''

  if (epoch + 1) % 10 == 0 or saved_flag:
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f'Época [{epoch+1}/{EPOCHS}] -> Train Loss: {avg_loss:.6f} | Val'
        f' Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f} {saved_flag}'
    )

print(f"🧠 Mejor checkpoint (Val Loss: {best_val_loss:.6f}) guardado en '{PATH_PESOS_SALIDA}'")