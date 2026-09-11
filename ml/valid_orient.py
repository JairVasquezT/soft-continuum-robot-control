import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import matplotlib.pyplot as plt

columnas = [
    'timestamp', 't_unix', 't_relativo', 'combo_id',
    'meta_m1', 'meta_m2', 'meta_m3','meta_m4',
    'real_m1', 'real_m2', 'real_m3','real_m4',
    'base_x', 'base_y', 'base_z', 'base_qx', 'base_qy', 'base_qz', 'base_qw',
    'efector_x', 'efector_y', 'efector_z', 'efector_qx', 'efector_qy', 'efector_qz', 'efector_qw'
]

df = pd.read_csv('logs_trayectoria_final_largo.csv', names=columnas)
df = df.iloc[1:].reset_index(drop=True)

plt.figure(figsize=(12, 4))
plt.plot(df['base_qw'], color='royalblue', linewidth=1.5)
plt.title("Diagnóstico de Estabilidad de la Base (OptiTrack)", fontsize=12)
plt.xlabel("Número de Fila (Tiempo)")
plt.ylabel("Valor del Cuaternión W")
plt.grid(True, linestyle='--')
plt.show()