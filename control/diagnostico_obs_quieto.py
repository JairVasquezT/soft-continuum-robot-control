r"""Looks for stretches where the 4 motors stayed EXACTLY still (same

real ticks in consecutive rows) and measures how much the reported
position varied (Obs and, if available, Opti) during those stretches.

If Obs varies considerably with the motors still, it is the LSTM
observer's own estimate that is unstable/drifting -- it is not the robot
actually moving. If Opti (real camera) also varies by the same magnitude, then it is
real physical movement (cable settling, etc).

Usage: python continuum_robot/control/diagnostico_obs_quieto.py ruta\al\mpc_experiment_YYYYMMDD_HHMMSS.csv
"""
import csv
import math
import sys

path = sys.argv[1]
MIN_FILAS_QUIETO = 10  # minimum run of rows with identical ticks to count

rows = []
header = None
with open(path, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    for r in reader:
        if header is None:
            header = r
            continue
        if len(r) == len(header):
            rows.append(r)

# The real col. 0 of the CSV is the write timestamp (it is not in `header`,
# which describes from col. 1 onward -- see run_control_loop).
idx = {name: i + 1 for i, name in enumerate(header)}


def col(row, name):
    return float(row[idx[name]])


REAL_COLS = ['real_m1_ticks', 'real_m2_ticks', 'real_m3_ticks', 'real_m4_ticks']
OBS_COLS = ['pos_obs_x_mm', 'pos_obs_y_mm', 'pos_obs_z_mm']
OPTI_COLS = ['pos_opti_x_mm', 'pos_opti_y_mm', 'pos_opti_z_mm']

if not all(c in idx for c in REAL_COLS):
    print('Este CSV no tiene las columnas real_mX_ticks -- corré una corrida nueva '
          'con la versión actualizada de MPC.py primero.')
    sys.exit(1)

# 1. Detect runs of 100% identical real ticks
rachas = []
inicio = 0
ticks_prev = None
for i, row in enumerate(rows):
    ticks_cur = tuple(col(row, c) for c in REAL_COLS)
    if ticks_prev is not None and ticks_cur != ticks_prev:
        if i - inicio >= MIN_FILAS_QUIETO:
            rachas.append((inicio, i - 1))
        inicio = i
    ticks_prev = ticks_cur
if len(rows) - inicio >= MIN_FILAS_QUIETO:
    rachas.append((inicio, len(rows) - 1))

print(f'Filas totales: {len(rows)}')
print(f'Rachas con los 4 motores 100% quietos (>= {MIN_FILAS_QUIETO} filas seguidas): {len(rachas)}')
print()

if not rachas:
    print('No se encontró ningún tramo con motores totalmente quietos en este CSV.')
    sys.exit(0)

for (a, b) in rachas:
    n = b - a + 1
    obs_vals = [[col(rows[i], c) for c in OBS_COLS] for i in range(a, b + 1)]
    obs_range = [
        max(v[k] for v in obs_vals) - min(v[k] for v in obs_vals) for k in range(3)
    ]
    obs_range_norm = math.sqrt(sum(d * d for d in obs_range))
    print(f'Filas {a}-{b} ({n} filas quietas): variación Obs (X,Y,Z) mm = '
          f'{obs_range[0]:.2f}, {obs_range[1]:.2f}, {obs_range[2]:.2f}  '
          f'| magnitud total ~{obs_range_norm:.2f} mm')

    opti_vals = [[col(rows[i], c) for c in OPTI_COLS] for i in range(a, b + 1)]
    tiene_opti = not any(v != v for fila in opti_vals for v in fila)  # detects NaN
    if tiene_opti:
        opti_range = [
            max(v[k] for v in opti_vals) - min(v[k] for v in opti_vals) for k in range(3)
        ]
        opti_range_norm = math.sqrt(sum(d * d for d in opti_range))
        print(f'  variación Opti (X,Y,Z) mm = '
              f'{opti_range[0]:.2f}, {opti_range[1]:.2f}, {opti_range[2]:.2f}  '
              f'| magnitud total ~{opti_range_norm:.2f} mm')
    else:
        print('  (sin datos de OptiTrack en este tramo -- N/D)')
