import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS (VERSION V1)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Entrenando en el dispositivo: {device}")

# Cargar el dataset v2 (vienen X e Y empaquetados en un diccionario)
data_v2 = np.load('dataset_v2_conMeta_sinRot.npy', allow_pickle=True).item()
X_raw = data_v2['X']  # Contiene: delta_real_m1 a m4 (4 columnas) y delta_meta_m1 a m4 (4 columnas)
Y_raw = data_v2['Y']  # Contiene: rel_x, rel_y, rel_z (3 columnas)

# Cargar los parámetros de normalización JSON
with open('dataset_v2_conMeta_sinRot_params.json', 'r') as f:
    norm_params = json.load(f)

# ==========================================
# 2. CREACIÓN DE VENTANAS TEMPORALES
# ==========================================
def crear_secuencias(X, Y, window_size=90):
    X_seq, Y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i : i + window_size])  
        Y_seq.append(Y[i + window_size])      
    return np.array(X_seq), np.array(Y_seq)

WINDOW_SIZE = 20
X_windows, Y_windows = crear_secuencias(X_raw, Y_raw, window_size=WINDOW_SIZE)

# Convertir a Tensores de PyTorch
X_tensor = torch.tensor(X_windows, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_windows, dtype=torch.float32).to(device)

# División SECUENCIAL CRONOLÓGICA (Evita que el test conozca el pasado inmediato)
#train_size = int(0.8 * len(X_tensor))
#X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
#y_train, y_test = Y_tensor[:train_size], Y_tensor[train_size:]

# DataLoader con Shuffle habilitado para romper correlación entre épocas
#train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
train_loader = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=64, shuffle=True)

# ==========================================
# 3. ARQUITECTURA LSTM MODIFICADA (INPUT_SIZE = 4)
# ==========================================
class SoftRobotLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=3, dropout=0.0):
        super(SoftRobotLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        last_step = out[:, -1, :]
        out_regularized = self.dropout(last_step)
        return self.fc(out_regularized)

# Instanciar el modelo para V2 (8 entradas -> 3 salidas)
model = SoftRobotLSTM(input_size=8, hidden_size=64, num_layers=2, output_size=3, dropout=0.2).to(device)

# ==========================================
# 4. CONFIGURACIÓN DEL OPTIMIZADOR Y COSTO
# ==========================================
criterion = nn.HuberLoss() # MAE
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# ==========================================
# 5. BUCLE DE ENTRENAMIENTO
# ==========================================
EPOCHS = 150  
history_loss = []

print("🚀 Iniciando entrenamiento del modelo V2...")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        
        # Clip de gradiente para proteger estabilidad numérica
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    history_loss.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Época [{epoch+1}/{EPOCHS}] -> Error Promedio (MAE Norm): {avg_loss:.5f}")

# ==========================================
# 6. EVALUACIÓN EN DATOS DE TEST COREADOS
# ==========================================
'''
model.eval()
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
test_preds_list = []

print("\n🔍 Evaluando en el 20% de datos de prueba...")
with torch.no_grad():
    for batch_X_test, _ in test_loader:
        preds = model(batch_X_test)
        test_preds_list.append(preds.cpu().numpy())

test_predictions_scaled = np.vstack(test_preds_list)
y_test_real_scaled = y_test.cpu().numpy()
'''

# Guardar los pesos entrenados
torch.save(model.state_dict(), 'soft_robot_lstm_v2_1.pth')
print("🧠 Pesos guardados como 'soft_robot_lstm_v2_1.pth'")

# ==========================================
# 7. TRADUCTOR REVERSO MATEMÁTICO (DESNORMALIZACIÓN DESDE JSON)
# ==========================================
def desnormalizar_coordenadas(data_scaled, params_json):
    data_mm = np.zeros_like(data_scaled)
    ejes = ['rel_x', 'rel_y', 'rel_z']
    
    for idx, eje in enumerate(ejes):
        min_t = params_json['Y_transformer'][eje]['min_t']
        max_t = params_json['Y_transformer'][eje]['max_t']
        
        # Despeje de la normalización [-1, 1]:
        # raw = min_t + ((scaled + 1) / 2) * (max_t - min_t)
        data_mm[:, idx] = min_t + ((data_scaled[:, idx] + 1) / 2) * (max_t - min_t)
        
    return data_mm * 1000.0  # Multiplicamos por 1000 si deseas visualizarlo en milímetros puros

'''
# Convertir predicciones y valores reales del test a unidades físicas
y_test_real_mm = desnormalizar_coordenadas(y_test_real_scaled, norm_params)
y_test_pred_mm = desnormalizar_coordenadas(test_predictions_scaled, norm_params)

# Cálculo de Métricas Físicas Reales sobre el conjunto de TEST
mae_por_eje = np.mean(np.abs(y_test_real_mm - y_test_pred_mm), axis=0)
distancias_3d = np.sqrt(np.sum((y_test_real_mm - y_test_pred_mm)**2, axis=1))
error_3d_promedio = np.mean(distancias_3d)


print("\n=======================================================")
print("📊 RESULTADOS FINALES DE VALIDACIÓN CRUZADA EN TEST (V1)")
print("=======================================================")
print(f"Error MAE Eje X: {mae_por_eje[0]:.3f} mm")
print(f"Error MAE Eje Y: {mae_por_eje[1]:.3f} mm")
print(f"Error MAE Eje Z: {mae_por_eje[2]:.3f} mm")
print(f"📐 ERROR DE DISTANCIA EUCLÍDEA 3D PROMEDIO: {error_3d_promedio:.3f} mm")
print("=======================================================\n")

'''
# ==========================================
# 8. GRAFICAR RESULTADOS DEL TEST SEGUIDO
# ==========================================
'''
plt.figure(figsize=(12, 5))

# Grafica Izquierda: Pérdida
plt.subplot(1, 2, 1)
plt.plot(history_loss, label='Error de Entrenamiento (MAE)', color='blue')
plt.title('Convergencia del Modelo V1')
plt.xlabel('Épocas')
plt.ylabel('Loss (Norm)')
plt.grid(True)
plt.legend()

# Grafica Derecha: Comportamiento en el Eje X real del test predictivo
plt.subplot(1, 2, 2)
plt.plot(y_test_real_mm[:, 0], label='Real Oculto (OptiTrack)', color='black')
plt.plot(y_test_pred_mm[:, 0], label='Predicho (LSTM V1)', linestyle='--', color='crimson')
plt.title('Predicción en Datos de Test (Futuro): Eje X')
plt.xlabel('Frames de Test')
plt.ylabel('Posición Real (mm)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
'''