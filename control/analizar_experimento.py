"""Analiza un CSV de mpc_experimento_repeticiones_*.csv para el informe:

  1. Gráficos por waypoint (trayectoria 3D real vs. target + error vs.
     tiempo, un intento por color, verde=éxito/rojo=fallo), guardados como
     PNG.
  2. Estadísticas de t_calc_ms (tiempo de cómputo del CEM por ciclo de
     Action Hold) -- para respaldar numéricamente la afirmación de "tiempo
     real" de la sección de formulación del MPC.
  3. Resumen por intento (resultado, error final, duración) para inspeccionar
     patrones puntuales (p.ej. por qué un intento falla y el siguiente al
     mismo punto converge rápido).

Uso:
  python analizar_experimento.py mpc_experimento_repeticiones_XXXXXXXX.csv
  python analizar_experimento.py mpc_experimento_repeticiones_XXXXXXXX.csv --wp 2 4 7 8
  python analizar_experimento.py mpc_experimento_repeticiones_XXXXXXXX.csv --listar
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLS = [
    'log_ts', 't_epoch', 'wp_idx', 'intento_idx',
    'target_x_mm', 'target_y_mm', 'target_z_mm',
    'pos_ctrl_x_mm', 'pos_ctrl_y_mm', 'pos_ctrl_z_mm',
    'pos_obs_x_mm', 'pos_obs_y_mm', 'pos_obs_z_mm',
    'pos_opti_x_mm', 'pos_opti_y_mm', 'pos_opti_z_mm',
    'err_mm', 'err_x_mm', 'err_y_mm', 'err_z_mm',
    'err_opti_mm', 'err_opti_x_mm', 'err_opti_y_mm', 'err_opti_z_mm',
    'u1', 'u2', 'u3', 'u4',
    'real_m1_ticks', 'real_m2_ticks', 'real_m3_ticks', 'real_m4_ticks',
    'cmd_m1_ticks', 'cmd_m2_ticks', 'cmd_m3_ticks', 'cmd_m4_ticks',
    'bloqueado_hist_m1', 'bloqueado_hist_m2', 'bloqueado_hist_m3', 'bloqueado_hist_m4',
    'en_limite_m1', 'en_limite_m2', 'en_limite_m3', 'en_limite_m4',
    'es_action_hold', 't_calc_ms', 'cost', 'resultado',
]

TOLERANCIA_MM_DEFAULT = 10.0


def cargar(path):
  df = pd.read_csv(path, header=0, names=COLS)
  return df


def listar_waypoints(df):
  print('Waypoints disponibles en este CSV:')
  for wp in sorted(df.wp_idx.unique()):
    sub = df[df.wp_idx == wp]
    target = sub.iloc[0][['target_x_mm', 'target_y_mm', 'target_z_mm']].to_numpy(dtype=float)
    intentos = sorted(sub.intento_idx.unique())
    resultados = [
        sub[sub.intento_idx == it].iloc[-1].resultado for it in intentos
    ]
    print(f'  WP{wp + 1} (wp_idx={wp}): target={np.round(target, 2).tolist()} mm '
          f'| resultados={resultados}')


def graficar_waypoint(df, wp_idx, carpeta_salida, tolerancia_mm):
  sub = df[df.wp_idx == wp_idx]
  if sub.empty:
    print(f'⚠️ No hay datos para wp_idx={wp_idx} (WP{wp_idx + 1}) en este CSV.')
    return None

  target = sub.iloc[0][['target_x_mm', 'target_y_mm', 'target_z_mm']].to_numpy(dtype=float)
  colores = {'exito': 'green', 'fallo': 'red'}

  fig = plt.figure(figsize=(14, 6))

  ax1 = fig.add_subplot(1, 2, 1, projection='3d')
  for intento in sorted(sub.intento_idx.unique()):
    s = sub[sub.intento_idx == intento]
    resultado = s.iloc[-1].resultado
    color = colores.get(resultado, 'gray')
    ax1.plot(
        s.pos_opti_x_mm, s.pos_opti_y_mm, s.pos_opti_z_mm,
        color=color, alpha=0.75, linewidth=1.5,
        label=f'Intento {intento + 1} ({resultado})',
    )
  ax1.scatter(*target, color='black', marker='*', s=220, label='Target', zorder=5)
  ax1.set_xlabel('X (mm)')
  ax1.set_ylabel('Y (mm)')
  ax1.set_zlabel('Z (mm)')
  ax1.set_title(f'WP{wp_idx + 1} target={np.round(target, 1).tolist()} mm\nTrayectoria 3D (OptiTrack)')
  ax1.legend(fontsize=8)
  ax1.grid(True)

  ax2 = fig.add_subplot(1, 2, 2)
  for intento in sorted(sub.intento_idx.unique()):
    s = sub[sub.intento_idx == intento]
    resultado = s.iloc[-1].resultado
    color = colores.get(resultado, 'gray')
    t = s.t_epoch - s.t_epoch.iloc[0]
    ax2.plot(t, s.err_opti_mm, color=color, alpha=0.85, linewidth=1.5,
              label=f'Intento {intento + 1} ({resultado})')
  ax2.axhline(tolerancia_mm, color='gray', linestyle='--', linewidth=1,
               label=f'Tolerancia ({tolerancia_mm:.0f} mm)')
  ax2.set_xlabel('Tiempo (s)')
  ax2.set_ylabel('Error OptiTrack (mm)')
  ax2.set_title(f'WP{wp_idx + 1} -- Error vs. tiempo')
  ax2.legend(fontsize=8)
  ax2.grid(True)

  plt.tight_layout()
  carpeta_salida.mkdir(parents=True, exist_ok=True)
  salida = carpeta_salida / f'wp{wp_idx + 1}_trayectoria_error.png'
  fig.savefig(salida, dpi=150)
  plt.close(fig)
  print(f'✓ Guardado: {salida}')
  return salida


def estadisticas_t_calc(df, action_hold_cycles=4, freq_hz=60.0):
  ah = df[df.es_action_hold == 1]
  print(f"\n{'=' * 60}\nTIEMPO DE CÓMPUTO POR CICLO (t_calc_ms)\n{'=' * 60}")
  print(f'n = {len(ah)} recálculos de Action Hold en todo el CSV')
  print(f'  media   = {ah.t_calc_ms.mean():.2f} ms')
  print(f'  mediana = {ah.t_calc_ms.median():.2f} ms')
  print(f'  p95     = {ah.t_calc_ms.quantile(0.95):.2f} ms')
  print(f'  máximo  = {ah.t_calc_ms.max():.2f} ms')
  presupuesto_ms = action_hold_cycles * 1000.0 / freq_hz
  print(f'\nPresupuesto disponible por recálculo (action_hold_cycles='
        f'{action_hold_cycles} a {freq_hz:.0f}Hz): {presupuesto_ms:.1f} ms')
  margen_pct = 100.0 * (1 - ah.t_calc_ms.mean() / presupuesto_ms)
  print(f'Margen promedio respecto al presupuesto: {margen_pct:.1f} %')
  excedidos = (ah.t_calc_ms > presupuesto_ms).sum()
  print(f'Ciclos que excedieron el presupuesto: {excedidos}/{len(ah)} '
        f'({100 * excedidos / len(ah):.2f} %)')


def resumen_intentos(df):
  print(f"\n{'=' * 60}\nRESUMEN POR INTENTO\n{'=' * 60}")
  for wp in sorted(df.wp_idx.unique()):
    for it in sorted(df[df.wp_idx == wp].intento_idx.unique()):
      s = df[(df.wp_idx == wp) & (df.intento_idx == it)]
      dur = s.t_epoch.iloc[-1] - s.t_epoch.iloc[0]
      print(f'WP{wp + 1} intento{it + 1}: resultado={s.iloc[-1].resultado:<6} '
            f'err_final={s.iloc[-1].err_opti_mm:6.2f}mm  duracion={dur:5.1f}s  '
            f'n_filas={len(s)}')


def investigar_patron(df, wp_idx):
  """Timeline detallado de un waypoint puntual, para inspeccionar patrones
  de intento a intento (p.ej. "falla y despues converge rapido dos veces")."""
  sub = df[df.wp_idx == wp_idx]
  if sub.empty:
    print(f'⚠️ No hay datos para wp_idx={wp_idx} (WP{wp_idx + 1}).')
    return
  print(f"\n{'=' * 60}\nPATRÓN DETALLADO -- WP{wp_idx + 1}\n{'=' * 60}")
  print('NOTA: _resetear_estado_mpc_para_nuevo_intento() limpia '
        'cem_std_ticks/error_history/stagnation_counter/last_delta_ticks '
        'ANTES de cada intento (ver run_experimento_repeticiones en MPC.py) '
        '-- no hay memoria persistida del CEM entre intentos. Si un intento '
        'falla y el siguiente converge rápido, NO puede deberse a que el '
        'CEM "recuerde" la std/dirección del intento anterior -- esa '
        'hipótesis queda descartada por el propio código. Explicaciones '
        'más probables: (a) las anclas aleatorias del multi-arranque se '
        'redibujan en cada intento (torch.rand nuevo), así que el punto de '
        'arranque cambia por azar; (b) histéresis física del cable (el '
        'estado mecánico tras el retorno a Home puede no ser idéntico '
        'bit a bit aunque el comando sea el mismo).')
  for it in sorted(sub.intento_idx.unique()):
    s = sub[sub.intento_idx == it]
    dur = s.t_epoch.iloc[-1] - s.t_epoch.iloc[0]
    err_inicial = s.iloc[0].err_opti_mm
    err_final = s.iloc[-1].err_opti_mm
    err_min = s.err_opti_mm.min()
    print(f'  Intento {it + 1}: resultado={s.iloc[-1].resultado} | '
          f'duración={dur:.1f}s | err inicial={err_inicial:.2f}mm | '
          f'err mínimo={err_min:.2f}mm | err final={err_final:.2f}mm | '
          f'{len(s)} filas')


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('csv', type=str, help='Ruta al CSV mpc_experimento_repeticiones_*.csv')
  parser.add_argument('--wp', type=int, nargs='+', default=None,
                       help='Waypoints a graficar/investigar, en numeración 1-based '
                            '(ej. --wp 2 4 7 8). Por defecto: todos los del CSV.')
  parser.add_argument('--listar', action='store_true',
                       help='Solo lista los waypoints disponibles en el CSV y sale.')
  parser.add_argument('--salida', type=str, default='graficos_experimento',
                       help='Carpeta donde guardar los PNG (por defecto ./graficos_experimento).')
  parser.add_argument('--tolerancia-mm', type=float, default=TOLERANCIA_MM_DEFAULT,
                       help='Tolerancia (mm) a marcar en el gráfico de error vs. tiempo.')
  parser.add_argument('--action-hold-cycles', type=int, default=4)
  parser.add_argument('--freq-hz', type=float, default=60.0)
  args = parser.parse_args()

  df = cargar(args.csv)
  print(f'✓ Cargado {args.csv}: {len(df)} filas, '
        f'{df.wp_idx.nunique()} waypoints, {df.intento_idx.nunique()} intentos por waypoint (máx).')

  if args.listar:
    listar_waypoints(df)
    return

  wps_1based = args.wp if args.wp is not None else sorted(w + 1 for w in df.wp_idx.unique())
  wps_0based = [w - 1 for w in wps_1based]

  carpeta_salida = Path(args.salida)
  for wp_idx in wps_0based:
    graficar_waypoint(df, wp_idx, carpeta_salida, args.tolerancia_mm)

  estadisticas_t_calc(df, action_hold_cycles=args.action_hold_cycles, freq_hz=args.freq_hz)
  resumen_intentos(df)

  for wp_idx in wps_0based:
    investigar_patron(df, wp_idx)


if __name__ == '__main__':
  main()
