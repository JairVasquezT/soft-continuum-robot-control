import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=90)
parser.add_argument('--output', type=str, default='soft_robot_lstm_v8_18.pth')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.0005)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Entrenando Modelo V8 (estándar, sin velocidad) en: {device}")

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================
PATH_DATASET_NPY = 'dataset_v08_completo_sinRot_filt.npy'
PATH_DATASET_JSON = 'dataset_v08_completo_sinRot_filt_params.json'
PATH_PESOS_SALIDA = args.output

data_v8 = np.load(PATH_DATASET_NPY, allow_pickle=True).item()
X_raw, Y_raw = data_v8['X'], data_v8['Y']

with open(PATH_DATASET_JSON, 'r') as f:
    norm_params = json.load(f)

Y_TRANS = norm_params['Y_transformer']

num_features_input = X_raw.shape[1]
print(f"📊 Entradas: {num_features_input} | Salidas: {Y_raw.shape[1]}")


# ==========================================
# 2. VENTANADO ESTÁNDAR Y SPLIT TRAIN/VAL
# ==========================================
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

        X_l.append(X[i: i + window_size])
        Y_l.append(Y[i + window_size])

    print(f"   (bloques de validación repartidos: {sorted(bloques_val)}/{n_bloques} "
          f"| {descartadas} ventanas descartadas por cruzar un borde de bloque)")

    return {
        'train': (np.array(datos['train'][0]), np.array(datos['train'][1])),
        'val': (np.array(datos['val'][0]), np.array(datos['val'][1]))
    }


WINDOW_SIZE = args.window_size
split = crear_secuencias_split(X_raw, Y_raw, WINDOW_SIZE)
X_train, Y_train = split['train']
X_val, Y_val = split['val']

print(f"📦 Muestras de entrenamiento: {len(X_train)} | Muestras de validación: {len(X_val)}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                  torch.tensor(Y_train, dtype=torch.float32)),
    batch_size=64, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                  torch.tensor(Y_val, dtype=torch.float32)),
    batch_size=64, shuffle=False
)


# ==========================================
# 3. PÉRDIDA FÍSICA (GEOMETRÍA BASE)
# ==========================================
class RobotBlandoLoss(nn.Module):
    def __init__(self, y_params, w_pinn=0.05, w_sph_max=1.0, w_sph_min=1.0, w_cyl_max=1.0):
        super().__init__()
        self.base_loss = nn.HuberLoss(delta=0.01)
        self.w_pinn = w_pinn
        self.w_sph_max = w_sph_max
        self.w_sph_min = w_sph_min
        self.w_cyl_max = w_cyl_max

        self.min_x, self.max_x = y_params['rel_x']['min_t'], y_params['rel_x']['max_t']
        self.min_y, self.max_y = y_params['rel_y']['min_t'], y_params['rel_y']['max_t']
        self.min_z, self.max_z = y_params['rel_z']['min_t'], y_params['rel_z']['max_t']

    def _desescalar(self, y):
        x = self.min_x + (y[:, 0] + 1.0) * (self.max_x - self.min_x) / 2.0
        y_ = self.min_y + (y[:, 1] + 1.0) * (self.max_y - self.min_y) / 2.0
        z = self.min_z + (y[:, 2] + 1.0) * (self.max_z - self.min_z) / 2.0
        return x, y_, z

    def forward(self, y_pred, y_true):
        loss_base = self.base_loss(y_pred, y_true)

        x_real, y_real, z_real = self._desescalar(y_pred)
        dist_esf2 = x_real ** 2 + y_real ** 2 + z_real ** 2
        dist_cil2 = x_real ** 2 + z_real ** 2

        # Geometría del espacio de trabajo
        pen_sph_max = torch.relu(dist_esf2 - 0.335 ** 2)
        pen_sph_min = torch.relu(0.22 ** 2 - dist_esf2)
        pen_cyl_max = torch.relu(dist_cil2 - 0.23 ** 2)

        loss_pinn = (self.w_sph_max * torch.mean(pen_sph_max)
                     + self.w_sph_min * torch.mean(pen_sph_min)
                     + self.w_cyl_max * torch.mean(pen_cyl_max))

        return loss_base + self.w_pinn * loss_pinn


criterion = RobotBlandoLoss(y_params=Y_TRANS, w_pinn=0.05)


# ==========================================
# 4. ARQUITECTURA RED
# ==========================================
class SoftRobotLSTM(nn.Module):
    def __init__(self, input_size=17, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(out[:, -1, :]))


model = SoftRobotLSTM(input_size=num_features_input, hidden_size=args.hidden_size,
                       num_layers=args.num_layers, output_size=3, dropout=args.dropout).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                   patience=8, min_lr=1e-6)

# ==========================================
# 5. ENTRENAMIENTO Y VALIDACIÓN
# ==========================================
EPOCHS = 100
best_val_loss = float('inf')

print(f"🚀 Iniciando entrenamiento estándar (ventana={WINDOW_SIZE})...")

for epoch in range(EPOCHS):
    # FASE DE ENTRENAMIENTO
    model.train()
    epoch_loss = 0.0
    for Xb, Yb in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)

        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, Yb)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # FASE DE VALIDACIÓN
    model.eval()
    val_epoch_loss = 0.0
    with torch.no_grad():
        for Xb, Yb in val_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, Yb)
            val_epoch_loss += loss.item()

    avg_val_loss = val_epoch_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    # CHECKPOINTING
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': num_features_input,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'output_size': 3,
            'dropout': args.dropout,
            'window_size': WINDOW_SIZE,
            'best_val_loss': best_val_loss
        }, PATH_PESOS_SALIDA)
        saved_flag = '⭐ [Guardado]'
    else:
        saved_flag = ''

    if (epoch + 1) % 10 == 0 or saved_flag:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Época [{epoch+1}/{EPOCHS}] -> Train: {avg_loss:.6f} | '
              f'Val: {avg_val_loss:.6f} | LR: {current_lr:.6f} {saved_flag}')

print(f"🧠 Mejor checkpoint (Val Loss: {best_val_loss:.6f}) guardado en '{PATH_PESOS_SALIDA}'")