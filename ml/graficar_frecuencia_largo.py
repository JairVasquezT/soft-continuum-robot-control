"""Histogram of the instantaneous sampling frequency (1/delta_t) of the
'largo' recording, for the report -- shows how it is distributed between ~30 and
~60 Hz (lost packets/frames make some delta_t be double or
more of the nominal 60Hz period).

Usage:
  python graficar_frecuencia_largo.py
  python graficar_frecuencia_largo.py --csv otra_grabacion.csv
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNAS = [
    'timestamp', 't_unix', 't_relativo', 'combo_id', 'valid_lecture',
    'meta_m1', 'meta_m2', 'meta_m3', 'meta_m4',
    'real_m1', 'real_m2', 'real_m3', 'real_m4',
    'couple_m1', 'couple_m2', 'couple_m3', 'couple_m4',
    'tension_m1', 'tension_m2', 'tension_m3', 'tension_m4',
    'base_x', 'base_y', 'base_z', 'base_qx', 'base_qy', 'base_qz', 'base_qw',
    'efector_x', 'efector_y', 'efector_z',
    'efector_qx', 'efector_qy', 'efector_qz', 'efector_qw',
]

FS_NOMINAL_HZ = 60.0
FS_FILTRO_HZ = 55.0  # same threshold as MAX_DELTA_T in dataset_filtre.py/dataset_pred_filt.py


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', type=str, default='largo_20260730_191948_808.csv')
  parser.add_argument('--salida', type=str, default='frecuencia_largo.png')
  args = parser.parse_args()

  df = pd.read_csv(args.csv, names=COLUMNAS)
  delta_t = df['t_relativo'].astype(float).diff().dropna()
  delta_t = delta_t[delta_t > 0]  # discards spurious zeros/negatives (clock resets, etc.)
  freq_hz = 1.0 / delta_t

  print(f'✓ {len(df)} filas leídas de {args.csv}')
  print(f'  {len(freq_hz)} intervalos válidos calculados')
  print(f'  Frecuencia: media={freq_hz.mean():.2f}Hz | mediana={freq_hz.median():.2f}Hz '
        f'| min={freq_hz.min():.2f}Hz | max={freq_hz.max():.2f}Hz')
  bajo_filtro = (freq_hz < FS_FILTRO_HZ).sum()
  print(f'  Muestras bajo el umbral de filtrado ({FS_FILTRO_HZ}Hz): '
        f'{bajo_filtro}/{len(freq_hz)} ({100 * bajo_filtro / len(freq_hz):.2f}%)')

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  # Full histogram (free range, to see the tail of low frequencies)
  ax1.hist(freq_hz, bins=100, color='steelblue', edgecolor='black', linewidth=0.3)
  ax1.axvline(FS_NOMINAL_HZ, color='green', linestyle='--', label=f'{FS_NOMINAL_HZ:.0f} Hz nominal')
  ax1.axvline(FS_FILTRO_HZ, color='red', linestyle='--', label=f'{FS_FILTRO_HZ:.0f} Hz umbral de filtro')
  ax1.set_xlabel('Frecuencia instantánea (Hz)')
  ax1.set_ylabel('Cantidad de muestras')
  ax1.set_title('Distribución completa de frecuencia instantánea (1/Δt)')
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  # Zoom on the 30-60 Hz range (where the majority is concentrated)
  en_rango = freq_hz[(freq_hz >= 25) & (freq_hz <= 65)]
  ax2.hist(en_rango, bins=80, color='steelblue', edgecolor='black', linewidth=0.3)
  ax2.axvline(FS_NOMINAL_HZ, color='green', linestyle='--', label=f'{FS_NOMINAL_HZ:.0f} Hz nominal')
  ax2.axvline(FS_FILTRO_HZ, color='red', linestyle='--', label=f'{FS_FILTRO_HZ:.0f} Hz umbral de filtro')
  ax2.axvline(30.0, color='orange', linestyle=':', label='30 Hz (mitad de la nominal)')
  ax2.set_xlabel('Frecuencia instantánea (Hz)')
  ax2.set_ylabel('Cantidad de muestras')
  ax2.set_title('Zoom 25-65 Hz')
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  plt.tight_layout()
  fig.savefig(args.salida, dpi=150)
  print(f'✓ Guardado: {args.salida}')


if __name__ == '__main__':
  main()
