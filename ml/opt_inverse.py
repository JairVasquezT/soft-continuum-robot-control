import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torch import nn


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "soft_robot_lstm_v2_3_time.pth"
PARAMS_PATH = BASE_DIR / "dataset_v2_conMeta_sinRot_params.json"

columnas_validas = [
    'timestamp', 't_unix', 't_relativo', 'combo_id',
    'meta_m1', 'meta_m2', 'meta_m3', 'meta_m4',
    'real_m1', 'real_m2', 'real_m3', 'real_m4',
    'base_x', 'base_y', 'base_z', 'base_qx', 'base_qy', 'base_qz', 'base_qw',
    'efector_x', 'efector_y', 'efector_z', 'efector_qx', 'efector_qy', 'efector_qz', 'efector_qw'
]
columnas_reales = [c for c in columnas_validas if c.startswith('real_m')]
columnas_meta = [c for c in columnas_validas if c.startswith('meta_m')]
columna_tiempo = 't_relativo'


class SoftRobotLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, device=None):
        if device is None:
            device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        out, _ = self.lstm(x, (h0, c0))
        last_step = out[:, -1, :]
        out_regularized = self.dropout(last_step)
        return self.fc(out_regularized)


def load_model_and_scalers(model_path=MODEL_PATH, params_path=PARAMS_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoftRobotLSTM(input_size=9, hidden_size=128, num_layers=2, output_size=3).to(device)

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo en {model_path}")

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = {}

    return model, device, params


MODEL, DEVICE, SCALERS = load_model_and_scalers()


def desnormalizar_vector(vector_normalizado, column_name):
    if not SCALERS:
        return np.asarray(vector_normalizado, dtype=float)

    if "Y_transformer" not in SCALERS:
        return np.asarray(vector_normalizado, dtype=float)

    transformer = SCALERS["Y_transformer"].get(column_name, {})
    if not transformer:
        return np.asarray(vector_normalizado, dtype=float)

    min_t = float(transformer.get("min_t", -1.0))
    max_t = float(transformer.get("max_t", 1.0))
    if np.isclose(max_t, min_t):
        return np.full_like(vector_normalizado, min_t, dtype=float)
    return min_t + (vector_normalizado + 1.0) * (max_t - min_t) / 2.0


def forward_kinematics_prediction(motores_objetivo, secuencia_historica, delta_t=0.05):
    """
    Construye un paso nuevo con la propuesta de motores, lo normaliza,
    y devuelve la posición 3D estimada desnormalizada.
    """
    motores_objetivo = np.asarray(motores_objetivo, dtype=np.float32).reshape(-1)
    secuencia_historica = np.asarray(secuencia_historica, dtype=np.float32)

    if secuencia_historica.ndim != 2 or secuencia_historica.shape[1] != 9:
        raise ValueError("secuencia_historica debe ser una matriz de forma (N, 9)")

    # 1. Construir el paso actual crudo (en unidades reales)
    paso_actual = np.hstack([motores_objetivo, motores_objetivo, delta_t]).astype(np.float32)
    nueva_secuencia_cruda = np.vstack([secuencia_historica, paso_actual])

    # 2. 🔥 NORMALIZAR LA ENTRADA AL RANGO [-1, 1] USANDO EL JSON 🔥
    nueva_secuencia_scaled = np.zeros_like(nueva_secuencia_cruda, dtype=np.float32)
    
    if "X_transformer" in SCALERS:
        # El JSON contiene las llaves en orden: real_m1..4, meta_m1..4, delta_t
        for idx, key in enumerate(SCALERS["X_transformer"].keys()):
            min_t = float(SCALERS["X_transformer"][key]["min_t"])
            max_t = float(SCALERS["X_transformer"][key]["max_t"])
            
            # Evitar división por cero
            if np.isclose(max_t, min_t):
                nueva_secuencia_scaled[:, idx] = 0.0
            else:
                # Fórmula de mapeo al rango [-1, 1]
                nueva_secuencia_scaled[:, idx] = 2.0 * (nueva_secuencia_cruda[:, idx] - min_t) / (max_t - min_t) - 1.0
    else:
        # Si por alguna razón no hay scalers, dejamos el crudo (no recomendado)
        nueva_secuencia_scaled = nueva_secuencia_cruda

    # 3. Convertir a Tensor el bloque ya normalizado
    secuencia_tensor = torch.tensor(nueva_secuencia_scaled, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # 4. Inferencia
    with torch.no_grad():
        pred_norm = MODEL(secuencia_tensor, device=DEVICE).cpu().numpy()[0]

    # 5. Desnormalizar la salida (Tu función actual está perfecta aquí)
    pred_real = np.array([
        desnormalizar_vector(pred_norm[0], "rel_x"),
        desnormalizar_vector(pred_norm[1], "rel_y"),
        desnormalizar_vector(pred_norm[2], "rel_z"),
    ], dtype=np.float32)

    return pred_real


def funcion_objetivo(motores_propuestos, x_deseado, secuencia_historica, lambda_suavizado=0.4):
    # 1. Calcular el error de posicionamiento en el espacio (en mm)
    x_predicho = forward_kinematics_prediction(motores_propuestos, secuencia_historica)
    error_posicion = np.linalg.norm(x_predicho - x_deseado) * 1000.0
    
    # 2. 🔥 NUEVO: Penalizar el "salto" desde el último paso conocido (en ticks) 🔥
    motores_anteriores = secuencia_historica[-1, :4] # El paso 59 real
    esfuerzo_motores = np.linalg.norm(motores_propuestos - motores_anteriores)
    
    # Costo total combinado
    costo_total = error_posicion + (lambda_suavizado * esfuerzo_motores)
    return float(costo_total)


def obtener_buffer_historico(path=None, n_pasos=59):
    if path is None:
        path = BASE_DIR / "largo_20260715_180227_622.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo de historial: {path}")

    df = pd.read_csv(path, names=columnas_validas, header=0)
    df.columns = df.columns.str.strip()

    faltantes = [c for c in columnas_validas if c not in df.columns]
    if faltantes:
        raise ValueError(f"El CSV no contiene las columnas esperadas. Faltan: {faltantes}")

    df = df.loc[:, columnas_validas].copy()
    df['delta_t'] = df[columna_tiempo].astype(float).diff()
    df['delta_t'] = df['delta_t'].bfill()
    df = df.iloc[1:].reset_index(drop=True)

    secuencia = []
    for _, row in df.head(n_pasos).iterrows():
        valores_reales = [float(row[c]) for c in columnas_reales]
        valores_meta = [float(row[c]) for c in columnas_meta]
        delta_t = float(row['delta_t'])
        if np.isnan(delta_t):
            delta_t = 0.05

        secuencia.append(valores_reales + valores_meta + [delta_t])

    arr = np.asarray(secuencia, dtype=np.float32)
    arr[:, :4] = arr[:, :4] - 2040.0
    arr[:, 4:8] = arr[:, 4:8] - 2040.0
    return arr


def resolver_cinematica_inversa(posicion_deseada, historial=None, x0=None, bounds=None):
    if historial is None:
        historial = obtener_buffer_historico()

    posicion_deseada = np.asarray(posicion_deseada, dtype=np.float32)
    if x0 is None:
        x0 = np.zeros(4, dtype=np.float32)
    if bounds is None:
        bounds = [(-150, 150)] * 4

    resultado = minimize(
        funcion_objetivo,
        x0=x0,
        args=(posicion_deseada, historial),
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-3,
    )
    return resultado


if __name__ == "__main__":
    # =====================================================================
    # EVALUACIÓN MASIVA SOBRE EL DATASET COMPLETÓ (Offline Validation)
    # =====================================================================
    
    path_csv = BASE_DIR / "largo_20260715_180227_622.csv"
    
    df_completo = pd.read_csv(path_csv, names=columnas_validas, header=0)
    df_completo.columns = df_completo.columns.str.strip()
    df_completo['delta_t'] = df_completo[columna_tiempo].astype(float).diff().bfill()
    
    N_PASOS_HISTORIAL = 59
    
    errores_mm = []
    convergencias = []
    motores_calculados = []
    motores_reales = []
    
    indices_evaluacion = np.linspace(N_PASOS_HISTORIAL + 1, len(df_completo) - 1, 50, dtype=int)
    
    print(f"🔄 Iniciando evaluación masiva en {len(indices_evaluacion)} puntos...")
    
    for idx in indices_evaluacion:
        # 1. Construir el historial dinámico
        df_historia = df_completo.iloc[idx - N_PASOS_HISTORIAL : idx].copy()
        
        secuencia = []
        for _, row in df_historia.iterrows():
            v_reales = [float(row[c]) for c in columnas_reales]
            v_meta = [float(row[c]) for c in columnas_meta]
            d_t = float(row['delta_t']) if not np.isnan(row['delta_t']) else 0.05
            secuencia.append(v_reales + v_meta + [d_t])
            
        historial_dinamico = np.asarray(secuencia, dtype=np.float32)
        # Quitar el offset de 2040 a los motores
        historial_dinamico[:, :4] -= 2040.0
        historial_dinamico[:, 4:8] -= 2040.0
        
        # 2. 🔥 FIX 2: OBJETIVO EN COORDENADAS RELATIVAS 🔥
        fila_objetivo = df_completo.iloc[idx]
        posicion_deseada = np.array([
            float(fila_objetivo['efector_x']) - float(fila_objetivo['base_x']), 
            float(fila_objetivo['efector_y']) - float(fila_objetivo['base_y']), 
            float(fila_objetivo['efector_z']) - float(fila_objetivo['base_z'])
        ], dtype=np.float32)
        
        motores_reales_test = np.array([float(fila_objetivo[c]) for c in columnas_reales]) - 2040.0
        
        # 3. 🔥 FIX 3: INICIO CALIENTE (WARM START) 🔥
        # En vez de empezar en [0,0,0,0], empezamos en la posición real del paso anterior (t-1)
        x0_warm_start = historial_dinamico[-1, :4].copy()

        # 4. 🔥 FIX 1: LÍMITES REALISTAS 🔥
        # Ampliamos los límites para que cubran el rango real de movimiento del Dynamixel
        limites_amplios = [(-1500, 1500)] * 4

        # Resolver Cinemática Inversa
        resultado = minimize(
            funcion_objetivo,
            x0=x0_warm_start, 
            args=(posicion_deseada, historial_dinamico),
            method="L-BFGS-B",
            bounds=limites_amplios,
            tol=1e-3,
            options={'eps': 0.7, 'maxiter': 50} 
        )
        
        errores_mm.append(resultado.fun)
        convergencias.append(resultado.success)
        motores_calculados.append(resultado.x)
        motores_reales.append(motores_reales_test)

    # =====================================================================
    # REPORTE
    # =====================================================================
    print("\n=======================================================")
    print("📊 REPORTE DE EVALUACIÓN GLOBAL DEL OPTIMIZADOR")
    print("=======================================================")
    print(f"Puntos evaluados: {len(indices_evaluacion)}")
    print(f"Tasa de convergencia exitosa: {np.mean(convergencias) * 100:.1f}%")
    print(f"Error medio de posicionamiento: {np.mean(errores_mm):.3f} mm")
    print(f"Error máximo registrado: {np.max(errores_mm):.3f} mm")
    print("=======================================================\n")
    
    print("📋 COMPARATIVA REAL vs SUGERIDO (Primeros 5 puntos):")
    for i in range(5):
        print(f"Punto {i+1}:")
        print(f"  -> Motores Reales (Dataset): {np.round(motores_reales[i], 1)}")
        print(f"  -> Motores Sugeridos (IK):   {np.round(motores_calculados[i], 1)}")
        print(f"  -> Error Residual (Dist):    {errores_mm[i]:.2f} mm\n")