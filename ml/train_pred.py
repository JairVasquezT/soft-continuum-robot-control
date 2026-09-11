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
PATH_DATASET = 'dataset_pred_v07_completo_10_sinRot_directo_filt.npy'
PATH_METADATA = 'dataset_pred_v07_completo_10_sinRot_directo_filt_params.json'

# Ganancias de PINNLossMPC parametrizables por CLI (ver run_experimentos_pesos_loss.py
# para el barrido de 16 combinaciones) -- los defaults reproducen el
# comportamiento actual si se corre el script sin argumentos.
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='best_mpc_pinn_predictor_8_10_dir.pth')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--w-mse', type=float, default=1000.0)
parser.add_argument('--w-direction-3d', type=float, default=0.2)
parser.add_argument('--salto-direccion', type=int, default=4,
                     help='Separación en pasos entre los dos puntos que se comparan '
                          'para loss_direction_3d (p.ej. 4 compara t+1 con t+5, no '
                          't+1 con t+2 -- consecutivos están muy cerca y son ruidosos).')
parser.add_argument('--w-speed', type=float, default=0.0)
parser.add_argument('--w-smooth', type=float, default=0.0)
parser.add_argument('--max-step-dist', type=float, default=0.003,
                     help='Umbral de velocidad máxima por paso (m) que penaliza w_speed '
                          '(por defecto 0.01; train_pred_v2.py usaba 0.015).')
parser.add_argument('--pesos-ejes-mse', type=float, nargs=3, default=[2.0, 1.0, 2.0],
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
      w_mse=1.0,
      w_direction_3d=0.0,#el primero fue 0.0
      w_sph_max=5.0,
      w_sph_min=5.0,
      w_cyl_max=5.0,
      w_speed=0.0,
      w_smooth=0.0,
      w_quat=0.0,
      max_step_dist=0.003,  # 1.0 cm por paso (16.7 ms)
      pesos_ejes_mse=(2.0, 1.0, 2.0),  # [X, Y, Z]
      salto_direccion=4,  # t+1 vs t+5 en vez de t+1 vs t+2 (más ruidoso)
  ):
    super(PINNLossMPC, self).__init__()
    self.params_y = params_y
    self.w_mse = w_mse
    self.w_direction_3d = w_direction_3d
    self.salto_direccion = salto_direccion
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

  def forward(self, y_pred, y_true, horizontes_validos):
    # horizontes_validos: [batch] -- cantidad de pasos futuros consecutivos
    # (desde t+1) que son realmente válidos para esa muestra (ver
    # calcular_horizonte_valido en dataset_pred_filt.py: corta en el primer
    # paso donde cambia el comando, hay dropout de OptiTrack o la
    # frecuencia cae). mask_paso[b, j] = True si el paso j (0-indexado) es
    # válido para la muestra b -- se usa para enmascarar TODOS los términos
    # de la loss que leen pasos del horizonte, no solo el MSE, para que
    # ninguno promedie basura más allá del horizonte válido de cada
    # muestra. Todos los promedios se normalizan por la cantidad TOTAL de
    # elementos válidos en el lote (no por muestra), para que cada paso de
    # tiempo válido pese igual sin importar de qué muestra vino (una
    # muestra con horizonte corto no debe pesar como una completa).
    batch_size, t_out_actual, _ = y_pred.shape
    idx_paso = torch.arange(t_out_actual, device=y_pred.device).unsqueeze(0)  # [1, T]
    mask_paso = idx_paso < horizontes_validos.unsqueeze(1)  # [batch, T] bool
    mask_paso_f = mask_paso.unsqueeze(-1).float()  # [batch, T, 1]

    # 1. MSE Ponderado por Eje en Espacio Escalado [-1, 1], enmascarado
    error_xyz_escalado = (y_pred[:, :, :3] - y_true[:, :, :3]) ** 2 * self.pesos_ejes_mse
    error_enmascarado = error_xyz_escalado * mask_paso_f
    loss_mse_ponderado = error_enmascarado.sum() / (mask_paso_f.sum() * 3 + 1e-8)

    # 2. Desescalar a Metros Reales
    x_pred, y_pred_real, z_pred = self.desescalar_xyz(y_pred)
    x_true, y_true_real, z_true = self.desescalar_xyz(y_true)

    pos_pred = torch.stack([x_pred, y_pred_real, z_pred], dim=-1)
    pos_true = torch.stack([x_true, y_true_real, z_true], dim=-1)

    # 3. Dirección en 3D -- comparando puntos separados por
    # self.salto_direccion pasos (p.ej. t+1 vs t+5 con salto=4) en vez de
    # pasos consecutivos (t+1 vs t+2): a un paso de distancia el
    # desplazamiento real es tan chico que la dirección es puro ruido: con
    # más separación el vector de desplazamiento es más largo y su
    # dirección más estable/significativa. Una transición (k, k+s) es
    # válida solo si el paso k+s (el más lejano) es válido -- si lo es, k
    # también lo es porque horizonte_valido cuenta pasos consecutivos desde
    # el inicio.
    s = self.salto_direccion
    dir_pred_3d = pos_pred[:, s:, :] - pos_pred[:, :-s, :]
    dir_true_3d = pos_true[:, s:, :] - pos_true[:, :-s, :]

    norm_true_3d = torch.norm(dir_true_3d, dim=-1)
    mask_movimiento = norm_true_3d > 0.0005  # Movimientos > 0.5 mm
    mask_horizonte_dir = mask_paso[:, s:]
    mask_dir_total = mask_movimiento & mask_horizonte_dir

    cos_sim_3d = torch.nn.functional.cosine_similarity(
        dir_pred_3d, dir_true_3d, dim=-1, eps=1e-6
    )
    loss_dir_raw = 1.0 - cos_sim_3d

    if mask_dir_total.sum() > 0:
      loss_direction_3d = torch.mean(loss_dir_raw[mask_dir_total])
    else:
      loss_direction_3d = torch.tensor(0.0, device=y_pred.device)

    # 4. Geometría Espacial -- penalización por paso, enmascarada por paso
    mask_paso_2d = mask_paso.float()  # [batch, T]
    n_pasos_validos = mask_paso_2d.sum() + 1e-8

    dist_esferica_cuadrada = x_pred**2 + y_pred_real**2 + z_pred**2
    dist_cilindrica_cuadrada = x_pred**2 + z_pred**2

    pen_sph_max = torch.relu(dist_esferica_cuadrada - 0.335**2)
    pen_sph_min = torch.relu(0.22**2 - dist_esferica_cuadrada)
    pen_cyl_max = torch.relu(dist_cilindrica_cuadrada - 0.23**2)

    loss_geom = (
        self.w_sph_max * (pen_sph_max * mask_paso_2d).sum() / n_pasos_validos
        + self.w_sph_min * (pen_sph_min * mask_paso_2d).sum() / n_pasos_validos
        + self.w_cyl_max * (pen_cyl_max * mask_paso_2d).sum() / n_pasos_validos
    )

    # 5. Velocidad y Suavizado -- una transición/aceleración es válida si el
    # paso MÁS LEJANO que involucra (k+1 para velocidad, k+2 para
    # aceleración) es válido, mismo criterio que la dirección.
    paso_diffs = pos_pred[:, 1:, :] - pos_pred[:, :-1, :]
    dist_por_paso = torch.norm(paso_diffs, dim=-1)
    mask_vel = mask_paso[:, 1:].float()
    n_validos_vel = mask_vel.sum() + 1e-8

    pen_velocidad = torch.relu(dist_por_paso - self.max_step_dist)
    loss_speed = self.w_speed * (pen_velocidad * mask_vel).sum() / n_validos_vel

    aceleracion = paso_diffs[:, 1:, :] - paso_diffs[:, :-1, :]
    mask_acc = mask_paso[:, 2:].unsqueeze(-1).float()  # [batch, T-2, 1]
    loss_smooth = self.w_smooth * (aceleracion**2 * mask_acc).sum() / (mask_acc.sum() * 3 + 1e-8)

    loss_quat = 0.0
    if y_pred.shape[-1] == 7:
      q = y_pred[:, :, 3:]
      norm_q = torch.norm(q, dim=-1)
      pen_quat = (norm_q - 1.0) ** 2
      loss_quat = self.w_quat * (pen_quat * mask_paso_2d).sum() / n_pasos_validos

    # Pérdida Total PINN
    total_loss = (
        self.w_mse * loss_mse_ponderado
        + self.w_direction_3d * loss_direction_3d
        + loss_geom
        + loss_speed
        + loss_smooth
        + loss_quat
    )
    componentes = {
        'mse': loss_mse_ponderado.detach(),
        'dir': loss_direction_3d.detach(),
        'geom': loss_geom.detach(),
        'speed': loss_speed.detach(),
        'smooth': loss_smooth.detach(),
    }
    return total_loss, loss_mse_ponderado, componentes


# =====================================================================
# 3. DATASET Y MODELO (IGUAL QUE ANTES)
# =====================================================================
class MPCDataset(Dataset):

  def __init__(self, npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    self.x_hist = torch.tensor(data['X_hist'], dtype=torch.float32)
    self.u_cand = torch.tensor(data['U_cand'], dtype=torch.float32)
    self.y_fut = torch.tensor(data['Y_fut'], dtype=torch.float32)
    self.horizonte_valido = torch.tensor(data['horizonte_valido'], dtype=torch.long)

  def __len__(self):
    return len(self.x_hist)

  def __getitem__(self, idx):
    return self.x_hist[idx], self.u_cand[idx], self.y_fut[idx], self.horizonte_valido[idx]


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
sample_x, sample_u, sample_y, sample_horizonte = dataset_completo[0]
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
    salto_direccion=args.salto_direccion,
    w_speed=args.w_speed,
    w_smooth=args.w_smooth,
    max_step_dist=args.max_step_dist,
    pesos_ejes_mse=tuple(args.pesos_ejes_mse),
).to(DEVICE)

print('\n⚖️  Pesos de PINNLossMPC:')
print(f'   w_mse={criterion.w_mse} | pesos_ejes_mse={criterion.pesos_ejes_mse.tolist()}')
print(f'   w_direction_3d={criterion.w_direction_3d} | salto_direccion={criterion.salto_direccion}')
print(f'   w_sph_max={criterion.w_sph_max} | w_sph_min={criterion.w_sph_min} | '
      f'w_cyl_max={criterion.w_cyl_max}')
print(f'   w_speed={criterion.w_speed} | max_step_dist={criterion.max_step_dist} | '
      f'w_smooth={criterion.w_smooth} | w_quat={criterion.w_quat}')

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

  for batch_x, batch_u, batch_y, batch_horizonte in train_loader:
    batch_x, batch_u, batch_y, batch_horizonte = (
        batch_x.to(DEVICE),
        batch_u.to(DEVICE),
        batch_y.to(DEVICE),
        batch_horizonte.to(DEVICE),
    )

    optimizer.zero_grad()
    predictions = model(batch_x, batch_u)

    total_loss, _, _ = criterion(predictions, batch_y, batch_horizonte)
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    train_loss += total_loss.item() * len(batch_x)

  train_loss /= train_size

  # Validación alineada con PINN Loss
  model.eval()
  val_loss_sum = 0.0
  val_mse_sum = 0.0
  val_componentes_sum = {'mse': 0.0, 'dir': 0.0, 'geom': 0.0, 'speed': 0.0, 'smooth': 0.0}

  with torch.no_grad():
    for batch_x, batch_u, batch_y, batch_horizonte in val_loader:
      batch_x, batch_u, batch_y, batch_horizonte = (
          batch_x.to(DEVICE),
          batch_u.to(DEVICE),
          batch_y.to(DEVICE),
          batch_horizonte.to(DEVICE),
      )

      predictions = model(batch_x, batch_u)
      total_loss, pure_mse_pond, componentes = criterion(predictions, batch_y, batch_horizonte)

      val_loss_sum += total_loss.item() * len(batch_x)
      val_mse_sum += pure_mse_pond.item() * len(batch_x)
      for nombre, valor in componentes.items():
        val_componentes_sum[nombre] += valor.item() * len(batch_x)

  val_loss = val_loss_sum / val_size
  val_mse = val_mse_sum / val_size
  val_componentes = {nombre: v / val_size for nombre, v in val_componentes_sum.items()}

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
    print(f"mse={val_componentes['mse']:.6f} | dir={val_componentes['dir']:.6f} | "
          f"geom={val_componentes['geom']:.6f} | speed={val_componentes['speed']:.6f} | "
          f"smooth={val_componentes['smooth']:.6f}")


print(f'\n✅ Entrenamiento completado. Mejor Val PINN Loss: {best_val_loss:.6f}')