# OptiTrack + Dynamixel Sync - Documentación

## Descripción General

Este módulo integra OptiTrack (Motive 2.2.0) con los motores Dynamixel para capturar datos sincronizados de motion capture y posiciones de actuadores.

El sistema genera **3 archivos CSV**:
1. **dynamixel_*.csv** — Posiciones de los 4 motores (1 línea por movimiento)
2. **optitrack_*.csv** — Datos de tracking OptiTrack (1 línea por frame @ 180Hz)
3. **sync_*.csv** — Datos sincronizados (1 línea cada ciclo @ 60Hz por defecto)

---

## Estructura de Archivos CSV

### 1. dynamixel_*.csv
**Frecuencia**: 1 línea por combinación de posición (no determinada por Hz)

**Encabezado**:
```
timestamp, combo_idx, combo_total, motor1_goal, motor1_actual, motor2_goal, motor2_actual, motor3_goal, motor3_actual, motor4_goal, motor4_actual
```

**Descripción**:
- `timestamp`: Timestamp UNIX del PC cuando se capturaron los datos
- `combo_idx`: Índice de la combinación actual (1 a 625)
- `combo_total`: Total de combinaciones (625 = 5^4)
- `motorX_goal`: Posición objetivo enviada al motor X
- `motorX_actual`: Posición actual leída del motor X

**Ejemplo**:
```
1688765432.123, 1, 625, 1550, 1548, 1550, 1549, 1400, 1401, 1400, 1399
1688765432.234, 2, 625, 1550, 1550, 1550, 1551, 1400, 1402, 1400, 1400
```

---

### 2. optitrack_*.csv
**Frecuencia**: 180 Hz (1 frame cada 5.56 ms)

**Encabezado**:
```
frame_number, timestamp, RB0_X, RB0_Y, RB0_Z, RB0_Qx, RB0_Qy, RB0_Qz, RB0_Qw, RB1_X, RB1_Y, RB1_Z, RB1_Qx, RB1_Qy, RB1_Qz, RB1_Qw, ...
```

**Descripción**:
- `frame_number`: Número de frame de Motive
- `timestamp`: Timestamp del frame según NatNet
- `RBX_*`: Datos del cuerpo rígido X
  - `RBX_X, RBX_Y, RBX_Z`: Posición en metros
  - `RBX_Qx, RBX_Qy, RBX_Qz, RBX_Qw`: Rotación como quaternión

**Ejemplo**:
```
12345, 1688765432.456, -0.123, 0.456, 1.234, 0.1, 0.2, 0.3, 0.999, 0.050, 0.150, 1.100, 0.15, 0.25, 0.35, 0.990
12346, 1688765432.461, -0.124, 0.457, 1.235, 0.1, 0.2, 0.3, 0.999, 0.051, 0.151, 1.101, 0.15, 0.25, 0.35, 0.990
```

---

### 3. sync_*.csv
**Frecuencia**: 60 Hz por defecto (configurable con `--sync-rate`)

**Encabezado**:
```
timestamp_sync, motor1_goal, motor1_actual, motor2_goal, motor2_actual, motor3_goal, motor3_actual, motor4_goal, motor4_actual, RB0_X, RB0_Y, RB0_Z, RB0_Qx, RB0_Qy, RB0_Qz, RB0_Qw, ...
```

**Descripción**:
- Combina datos de Dynamixel + OptiTrack
- Alineados a una frecuencia común (60 Hz por defecto)
- Cada fila contiene el último dato disponible de ambos sistemas

**Ejemplo**:
```
1688765432.123, 1550, 1548, 1550, 1549, 1400, 1401, 1400, 1399, -0.123, 0.456, 1.234, 0.1, 0.2, 0.3, 0.999, ...
1688765432.183, 1550, 1548, 1550, 1549, 1400, 1401, 1400, 1399, -0.124, 0.457, 1.235, 0.1, 0.2, 0.3, 0.999, ...
```

---

## Configuración

### Variables en `robot_config.py`

```python
# OptiTrack / Motive
OPTITRACK_HOST = '127.0.0.1'        # IP del servidor Motive (localhost en este PC)
OPTITRACK_PORT = 1511               # Puerto NatNet (fijo)
OPTITRACK_SAMPLING_RATE = 180       # Hz - frecuencia de Motive

# Sincronización
SYNC_SAMPLING_RATE = 60             # Hz - frecuencia para CSV combinado
```

### Parámetros CLI

```bash
# Parámetro                      Descripción                     Default
--sampling-rate INT              Frecuencia OptiTrack en Hz      180
--sync-rate INT                  Frecuencia CSV sincronizado     cfg.SYNC_SAMPLING_RATE
--no-simulate                    Usar hardware real              False (simulación)
--output-dir PATH                Directorio para logs            continuum_robot/data/
```

---

## Ejemplos de Uso

### Ejemplo 1: Simulación con CSV sincronizado a 60Hz (default)
```bash
python -m continuum_robot.examples.optitrack_dynamixel_sync
```

Genera:
- `dynamixel_20240706_143000.csv`
- `optitrack_20240706_143000.csv`
- `sync_20240706_143000.csv` (60 Hz)

---

### Ejemplo 2: Hardware real, CSV sincronizado a 30Hz
```bash
python -m continuum_robot.examples.optitrack_dynamixel_sync --no-simulate --sync-rate 30
```

---

### Ejemplo 3: OptiTrack a 240Hz, CSV sincronizado a 120Hz
```bash
python -m continuum_robot.examples.optitrack_dynamixel_sync \
  --sampling-rate 240 --sync-rate 120 --no-simulate
```

---

### Ejemplo 4: Guardar logs en carpeta personalizada
```bash
python -m continuum_robot.examples.optitrack_dynamixel_sync \
  --no-simulate --output-dir /path/to/custom/folder
```

---

## Configurar Motive 2.2.0 para NatNet

1. Abrir **Motive 2.2.0**
2. Ir a **View > Data Streaming**
3. Habilitar **Unicast Data Streaming**
4. En **Streaming IP Address**, especificar la IP del cliente (o `127.0.0.1` si es en el mismo PC)
5. Verificar que el puerto sea **1511** (por defecto)
6. Ejecutar el script Python

---

## Análisis de Datos (Python)

### Leer los CSVs generados

```python
import pandas as pd

# Leer Dynamixel
df_dxl = pd.read_csv('dynamixel_20240706_143000.csv')
print(df_dxl.head())

# Leer OptiTrack
df_opt = pd.read_csv('optitrack_20240706_143000.csv')
print(df_opt.head())

# Leer sincronizado
df_sync = pd.read_csv('sync_20240706_143000.csv')
print(df_sync.head())
```

### Calcular error de seguimiento

```python
import numpy as np

# Calcular error entre posición objetivo y actual
df_dxl['motor1_error'] = np.abs(df_dxl['motor1_goal'] - df_dxl['motor1_actual'])
print(df_dxl[['motor1_goal', 'motor1_actual', 'motor1_error']].head())
```

### Correlacionar Dynamixel con OptiTrack

```python
# El CSV sincronizado facilita la correlación:
# - Cada fila tiene timestamp_sync, datos Dynamixel + datos OptiTrack alineados
df_sync['motor1_error'] = np.abs(df_sync['motor1_goal'] - df_sync['motor1_actual'])

# Graficar posición vs posición OptiTrack (si hay RB asociado)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df_sync['timestamp_sync'], df_sync['motor1_actual'], label='Motor 1 actual')
plt.plot(df_sync['timestamp_sync'], df_sync['RB0_X'], label='RB0 X')
plt.legend()
plt.show()
```

---

## Notas Técnicas

- **180 Hz OptiTrack**: Motive captura a 180 Hz por defecto. Cambiar en Motive si es necesario.
- **CSV sincronizado**: Los datos se alinean cada `1/SYNC_SAMPLING_RATE` segundos. Utiliza el último dato disponible de cada sistema.
- **Timestamping**: Cada dato tiene su propio timestamp, lo que permite análisis offline.
- **Rotaciones**: Se almacenan como quaterniones (Qx, Qy, Qz, Qw) para evitar singularidades.

---

## Troubleshooting

**Error: "No se pudo conectar a NatNet"**
- Verificar que Motive esté corriendo
- Verificar que la IP en `OPTITRACK_HOST` sea correcta
- Verificar que el puerto 1511 no esté bloqueado por firewall

**CSV vacío o con pocas líneas**
- Aumentar número de combinaciones o duración del experimento
- Verificar que `SYNC_SAMPLING_RATE` no sea demasiado alto

**Errores de sincronización**
- Asegurar que la frecuencia de OptiTrack sea > frecuencia sincronizada
- Ejemplo: si `SYNC_SAMPLING_RATE = 240`, la frecuencia de OptiTrack debe ser >= 240 Hz
