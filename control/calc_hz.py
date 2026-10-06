r"""Computes the real average Hz of an MPC.py run from the CSV,

comparing TWO timestamp sources to isolate whether a large gap comes from the
control loop or from the thread that writes to disk (e.g. OneDrive stalls):

  - Column 0 (WRITE): datetime.utcnow() taken inside CSVLogger.log(),
    in the consumer thread that writes to disk. A gap here may be just the
    stalled logger (slow disk / OneDrive), not necessarily the real loop.
  - Column 1 (t_epoch, GENERATION): time.time() taken in the main control
    loop, when building the row. This is the one that really matters for measuring
    whether the physical control loop is hanging.

Usage: python continuum_robot/control/calc_hz.py ruta\al\mpc_experiment_YYYYMMDD_HHMMSS.csv
"""
import sys
from datetime import datetime

path = sys.argv[1]

write_ts = []
epoch_ts = []

with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        try:
            wts = datetime.fromisoformat(parts[0])
            ets = float(parts[1])
        except ValueError:
            continue  # header row or other non-numeric row
        write_ts.append(wts)
        epoch_ts.append(ets)

if len(epoch_ts) < 2:
    print('No hay suficientes filas con timestamp para calcular Hz.')
    sys.exit(1)


def resumen(nombre, deltas_seg, n_filas):
    duracion = sum(deltas_seg)
    hz = len(deltas_seg) / duracion if duracion > 0 else 0.0
    peor_i = max(range(len(deltas_seg)), key=lambda i: deltas_seg[i])
    print(f'--- {nombre} ---')
    print(f'  Hz promedio: {hz:.2f} Hz')
    print(f'  dt promedio: {(duracion / len(deltas_seg)) * 1000:.2f} ms')
    print(f'  dt máximo: {deltas_seg[peor_i] * 1000:.2f} ms  '
          f'(entre fila {peor_i + 1} y fila {peor_i + 2} de {n_filas})')


deltas_write = [(write_ts[i + 1] - write_ts[i]).total_seconds() for i in range(len(write_ts) - 1)]
deltas_epoch = [epoch_ts[i + 1] - epoch_ts[i] for i in range(len(epoch_ts) - 1)]

print(f'Filas totales: {len(epoch_ts)}')
print()
resumen('Timestamp de ESCRITURA a disco (columna 0, hilo logger)', deltas_write, len(write_ts))
print()
resumen('Timestamp de GENERACIÓN en el lazo de control (t_epoch, columna 1)', deltas_epoch, len(epoch_ts))
print()
if deltas_epoch and max(deltas_epoch) < 0.5 <= max(deltas_write):
    print('=> El lazo de control (t_epoch) está OK. El hueco grande es del hilo que')
    print('   escribe a disco (logger), no del control físico. Sospechar de I/O/OneDrive.')
elif deltas_epoch and max(deltas_epoch) >= 0.5:
    print('=> El lazo de control (t_epoch) TAMBIÉN tiene el hueco grande -> es un')
    print('   colgado real del lazo, no solo del logger.')
