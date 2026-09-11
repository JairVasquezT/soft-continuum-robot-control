r"""Grafica los resultados de una corrida de MPC.py a partir de su CSV de log.

Genera:
  1. Trayectoria 3D (target, control, OptiTrack).
  2. Error 3D y por eje vs. tiempo, con líneas verticales en cada cambio
     de waypoint.
  3. Por motor: ticks reales vs. comando enviado vs. sugerencia cruda del
     optimizador, marcando en rojo los ciclos donde la histéresis
     (--delta-min-ticks) bloqueó la corrección, y en naranja los ciclos
     donde el target quedó cerca del límite físico configurado.
  4. Costo del CEM vs. tiempo.
  5. Tiempo de cálculo (t_calc_ms) y frecuencia real del lazo de control.

  6. Predicción del modelo (t_out pasos adelante) vs. lo que realmente midió
     el sensor esa misma cantidad de pasos después -- distingue un problema
     de calibración/sesgo del modelo (predice bien pero el error real
     persiste igual) de un problema real de alcance físico.

Uso: python continuum_robot/control/plot_resultados.py ruta\al\mpc_experiment_YYYYMMDD_HHMMSS.csv
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
  print('Uso: python plot_resultados.py ruta\\al\\mpc_experiment_*.csv')
  sys.exit(1)

# 🎯 Filtro por waypoint: si no es None, se queda solo con las filas de ESE
# wp_idx (útil para logs de run_experimento_repeticiones, con varios
# waypoints y repeticiones en un mismo CSV) -- cambiar acá para ver otro
# punto. None = ver el CSV completo, sin filtrar (todos los waypoints).
WP_IDX_A_VER = 10

path = sys.argv[1]
df = pd.read_csv(path, header=0)

if WP_IDX_A_VER is not None and 'wp_idx' in df.columns:
  total_antes = len(df)
  df = df[df['wp_idx'] == WP_IDX_A_VER].reset_index(drop=True)
  if df.empty:
    print(f'❌ No hay filas con wp_idx={WP_IDX_A_VER} en este CSV.')
    sys.exit(1)
  print(f'🔎 Filtrado a wp_idx={WP_IDX_A_VER}: {len(df)}/{total_antes} filas')

# La col. 0 real del CSV es el timestamp de escritura del logger (antepuesto
# automáticamente por CSVLogger.log()); pandas la toma como nombre de columna
# "basura" al leer la fila de encabezado -- se ignora, no hace falta.
t = df['t_epoch'].values
t_rel = t - t[0]

# Índices donde cambia el "target lógico" -- wp_idx (waypoint) y, si existe
# (logs de run_experimento_repeticiones), también intento_idx (repetición:
# cada intento vuelve a Home y reintenta el mismo waypoint, así que también
# es un reinicio real del target desde el punto de vista de optimize()).
if 'intento_idx' in df.columns:
  target_logico = df['wp_idx'].astype(str) + '_' + df['intento_idx'].astype(str)
else:
  target_logico = df['wp_idx']
cambia_target = target_logico.values[1:] != target_logico.values[:-1]
wp_changes = np.where(cambia_target)[0] + 1
wp_change_times = t_rel[wp_changes] if len(wp_changes) else []

MOTOR_IDS = [1, 2, 3, 4]

print(f'✓ {len(df)} filas cargadas | {df["wp_idx"].nunique()} waypoint(s) | '
      f'duración total: {t_rel[-1]:.1f}s')

# =====================================================================
# 1. TRAYECTORIA 3D
# =====================================================================
fig1 = plt.figure(figsize=(8, 7))
ax = fig1.add_subplot(111, projection='3d')
ax.plot(df['pos_ctrl_x_mm'], df['pos_ctrl_y_mm'], df['pos_ctrl_z_mm'],
        color='crimson', linewidth=1.5, label='Control (fuente usada)')
if df['pos_opti_x_mm'].notna().any():
  ax.plot(df['pos_opti_x_mm'], df['pos_opti_y_mm'], df['pos_opti_z_mm'],
          color='black', linewidth=1.0, alpha=0.6, label='OptiTrack')
targets_unicos = df.drop_duplicates('wp_idx')[['target_x_mm', 'target_y_mm', 'target_z_mm']]
ax.scatter(targets_unicos['target_x_mm'], targets_unicos['target_y_mm'],
           targets_unicos['target_z_mm'], color='blue', s=80, marker='^',
           label='Waypoints objetivo')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
titulo_wp = f' (wp_idx={WP_IDX_A_VER})' if WP_IDX_A_VER is not None else ''
ax.set_title(f'Trayectoria 3D{titulo_wp}')
ax.legend()

# =====================================================================
# 2. ERROR 3D Y POR EJE VS. TIEMPO
# =====================================================================
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax2a.plot(t_rel, df['err_mm'], color='crimson')
ax2a.set_ylabel('Error 3D (mm)')
ax2a.set_title('Error respecto al target vs. tiempo')
ax2a.grid(True, alpha=0.3)

ax2b.plot(t_rel, df['err_x_mm'], label='Err X', alpha=0.8)
ax2b.plot(t_rel, df['err_y_mm'], label='Err Y', alpha=0.8)
ax2b.plot(t_rel, df['err_z_mm'], label='Err Z', alpha=0.8)
ax2b.axhline(0, color='gray', linewidth=0.8)
ax2b.set_ylabel('Error por eje (mm)')
ax2b.set_xlabel('Tiempo (s)')
ax2b.legend()
ax2b.grid(True, alpha=0.3)

for wt in wp_change_times:
  ax2a.axvline(wt, color='blue', linestyle='--', alpha=0.4)
  ax2b.axvline(wt, color='blue', linestyle='--', alpha=0.4)

# =====================================================================
# 3. POR MOTOR: TICKS REALES VS. COMANDO VS. SUGERENCIA CRUDA
# =====================================================================
fig3, axes3 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
for i, mid in enumerate(MOTOR_IDS):
  ax = axes3[i]
  ax.plot(t_rel, df[f'real_m{mid}_ticks'], color='black', linewidth=1.2, label='Real (encoder)')
  ax.plot(t_rel, df[f'cmd_m{mid}_ticks'], color='crimson', linewidth=1.0, alpha=0.8, label='Comando enviado')
  if f'raw_opt_m{mid}_ticks' in df.columns:
    ax.plot(t_rel, df[f'raw_opt_m{mid}_ticks'], color='gray', linewidth=0.8, alpha=0.5,
            linestyle=':', label='Sugerencia cruda CEM (pre-clip/histéresis)')

  bloqueado = df[f'bloqueado_hist_m{mid}'] == 1
  if bloqueado.any():
    ax.scatter(t_rel[bloqueado], df.loc[bloqueado, f'real_m{mid}_ticks'],
               color='red', s=14, zorder=5, label='Bloqueado por histéresis')

  en_limite = df[f'en_limite_m{mid}'] == 1
  if en_limite.any():
    ax.scatter(t_rel[en_limite], df.loc[en_limite, f'real_m{mid}_ticks'],
               color='orange', s=22, marker='^', zorder=5, label='Cerca del límite físico')

  ax.set_ylabel(f'm{mid} (ticks)')
  ax.grid(True, alpha=0.3)
  if i == 0:
    ax.legend(loc='upper right', fontsize=8)
  for wt in wp_change_times:
    ax.axvline(wt, color='blue', linestyle='--', alpha=0.3)
axes3[-1].set_xlabel('Tiempo (s)')
fig3.suptitle('Por motor: real vs. comando vs. sugerencia cruda del CEM')

# =====================================================================
# 4. COSTO DEL CEM VS. TIEMPO
# =====================================================================
fig4, ax4 = plt.subplots(figsize=(11, 4))
solo_hold = df[df['es_action_hold'] == 1]
ax4.plot(t_rel[df['es_action_hold'] == 1], solo_hold['cost'], color='purple', marker='.', markersize=3, linewidth=0.8)
ax4.set_ylabel('Costo CEM')
ax4.set_xlabel('Tiempo (s)')
ax4.set_title('Costo del candidato elegido (solo ciclos de recálculo)')
ax4.grid(True, alpha=0.3)
for wt in wp_change_times:
  ax4.axvline(wt, color='blue', linestyle='--', alpha=0.4)

# =====================================================================
# 5. TIEMPO DE CÁLCULO Y FRECUENCIA REAL DEL LAZO
# =====================================================================
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 4))
ax5a.plot(t_rel, df['t_calc_ms'], color='teal', linewidth=0.8)
ax5a.set_ylabel('t_calc_ms')
ax5a.set_xlabel('Tiempo (s)')
ax5a.set_title('Tiempo de cálculo del optimizador por ciclo')
ax5a.grid(True, alpha=0.3)

dt_ms = np.diff(t) * 1000.0
ax5b.hist(dt_ms, bins=50, color='teal', alpha=0.7)
hz_medio = 1000.0 / np.mean(dt_ms)
ax5b.axvline(np.mean(dt_ms), color='red', linestyle='--', label=f'media={np.mean(dt_ms):.1f}ms (~{hz_medio:.1f}Hz)')
ax5b.set_xlabel('dt entre ciclos (ms)')
ax5b.set_ylabel('Frecuencia (conteo)')
ax5b.set_title('Distribución del período real del lazo')
ax5b.legend()

# =====================================================================
# 5b. FRECUENCIA POR TIPO DE CICLO: multi-arranque (incluye ancla Jacobiano
# en modo hibrido/jacobiano) vs. CEM en régimen estable vs. hold
# =====================================================================
# es_action_hold==1 marca ciclos de RECÁLCULO (ver fig4 -- nombre engañoso,
# confirmado por el resto del script); ==0 son ciclos de "hold" que repiten
# el último comando sin llamar a optimize() (t_calc_ms≈0 ahí).
#
# El multi-arranque (que evalúa la ancla del Jacobiano junto con las demás)
# se dispara SIEMPRE en el primer recálculo tras un cambio de target lógico
# (wp_idx/intento_idx) -- eso es determinable con certeza desde el CSV. Los
# ciclos de recálculo restantes son CEM en régimen estable, sin Jacobiano.
# NOTA: el multi-arranque también puede dispararse por estancamiento genuino
# (sin cambio de target) -- eso NO es detectable desde este CSV, así que
# "multi-arranque" abajo es un piso (subestima levemente cuántos ciclos
# tocaron el Jacobiano), no el total exacto.
es_recalculo = df['es_action_hold'].values == 1
bloque_id = np.concatenate([[0], np.cumsum(cambia_target)])
df_tmp = pd.DataFrame({'bloque_id': bloque_id, 'es_recalculo': es_recalculo})
primeros_recalculo_idx = df_tmp[df_tmp['es_recalculo']].groupby('bloque_id').head(1).index.values

mask_multiarranque = np.zeros(len(df), dtype=bool)
mask_multiarranque[primeros_recalculo_idx] = True
mask_cem_estable = es_recalculo & ~mask_multiarranque
mask_hold = ~es_recalculo

# Alinear con dt_ms (dt_ms[i] = tiempo entre la fila i y la fila i+1, así
# que el tipo relevante es el de la fila i+1, la que se acaba de calcular).
grupos_dt = {
    'Multi-arranque (target nuevo, incl. Jacobiano)': mask_multiarranque[1:],
    'CEM régimen estable (sin Jacobiano)': mask_cem_estable[1:],
    'Hold (sin recálculo)': mask_hold[1:],
}

print('\n--- FRECUENCIA / TIEMPO DE CÁLCULO POR TIPO DE CICLO ---')
resumen_grupos = {}
for nombre, mask_dt in grupos_dt.items():
  n = int(mask_dt.sum())
  if n == 0:
    print(f'  {nombre}: sin muestras')
    continue
  dt_sub = dt_ms[mask_dt]
  hz_sub = 1000.0 / np.mean(dt_sub)
  t_calc_sub = df['t_calc_ms'].values[1:][mask_dt]
  print(f'  {nombre}: n={n} | dt medio={np.mean(dt_sub):.2f}ms (~{hz_sub:.2f}Hz) | '
        f't_calc_ms medio={np.nanmean(t_calc_sub):.2f}ms')
  resumen_grupos[nombre] = hz_sub

fig5c, ax5c = plt.subplots(figsize=(6, 4))
if resumen_grupos:
  nombres_cortos = ['Multi-arranque\n(+ Jacobiano)', 'CEM estable', 'Hold']
  valores = [resumen_grupos.get(n, 0.0) for n in grupos_dt.keys()]
  colores_barras = ['#dc2626', '#2563eb', '#94a3b8']
  ax5c.bar(nombres_cortos, valores, color=colores_barras)
  for i, v in enumerate(valores):
    ax5c.text(i, v, f'{v:.1f}Hz', ha='center', va='bottom', fontsize=9)
ax5c.set_ylabel('Frecuencia real media (Hz)')
ax5c.set_title('Frecuencia por tipo de ciclo')
ax5c.grid(True, alpha=0.3, axis='y')

# =====================================================================
# 6. PREDICCIÓN DEL MODELO (t_out ADELANTE) VS. REALIDAD
# =====================================================================
# Ajustar si t_out cambia en el predictor (dataset_pred_*_params.json -> 't_out').
T_OUT_STEPS = 10

pred_valida = df['pred_tout_x_mm'].notna() if 'pred_tout_x_mm' in df.columns else pd.Series(False, index=df.index)
if pred_valida.any():
  n = len(df)
  idx_pred = np.where(pred_valida.values)[0]
  idx_futuro = idx_pred + T_OUT_STEPS
  valido = idx_futuro < n
  idx_pred, idx_futuro = idx_pred[valido], idx_futuro[valido]

  t_pred = t_rel[idx_pred]
  pred_xyz = df[['pred_tout_x_mm', 'pred_tout_y_mm', 'pred_tout_z_mm']].values[idx_pred]
  real_xyz = df[['pos_ctrl_x_mm', 'pos_ctrl_y_mm', 'pos_ctrl_z_mm']].values[idx_futuro]
  err_prediccion_mm = np.linalg.norm(pred_xyz - real_xyz, axis=1)

  fig6, axes6 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
  ejes = ['X', 'Y', 'Z']
  colores = ['tab:blue', 'tab:orange', 'tab:green']
  for i, (eje, color) in enumerate(zip(ejes, colores)):
    axes6[0].plot(t_pred, pred_xyz[:, i], color=color, linestyle='--', alpha=0.7, label=f'Predicho {eje}')
    axes6[0].plot(t_pred, real_xyz[:, i], color=color, linewidth=1.2, label=f'Real {eje} ({T_OUT_STEPS} pasos después)')
  axes6[0].set_ylabel('Posición (mm)')
  axes6[0].set_title(f'Predicción del modelo (t+{T_OUT_STEPS}) vs. medición real esos mismos pasos después')
  axes6[0].legend(loc='upper right', fontsize=7, ncol=2)
  axes6[0].grid(True, alpha=0.3)

  axes6[1].plot(t_pred, err_prediccion_mm, color='crimson', linewidth=1.0)
  axes6[1].set_ylabel('Error de predicción (mm)')
  axes6[1].set_xlabel('Tiempo (s)')
  axes6[1].set_title('|Predicho - Real| -- si esto es chico y el error al target sigue alto, '
                      'el modelo predice bien pero no hay margen físico para corregir más')
  axes6[1].grid(True, alpha=0.3)
  for wt in wp_change_times:
    axes6[0].axvline(wt, color='blue', linestyle='--', alpha=0.3)
    axes6[1].axvline(wt, color='blue', linestyle='--', alpha=0.3)
else:
  print('⚠️ No hay columnas pred_tout_*_mm válidas en este CSV (corrida vieja, antes de agregar '
        'esta métrica, o log de run_experimento_repeticiones -- ese esquema nunca las tuvo).')

# =====================================================================
# 7. OVERLAY: the 5 attempts of this waypoint, error vs. time-since-start
# =====================================================================
# Only makes sense once WP_IDX_A_VER already filtered df to a single
# waypoint with several 'intento_idx' repetitions. Each attempt starts
# fresh from a blocking return-to-Home that is NEVER logged (see
# _mover_a_home_bloqueante in MPC.py, called before filas_intento is
# reset) -- so the last logged row of an attempt is already the last
# moment of that attempt itself, nothing from the next Home return leaks
# in. Time is reset to 0 at the start of EACH attempt so they overlap.
if WP_IDX_A_VER is not None and 'intento_idx' in df.columns:
  col_error = (
      'err_opti_mm' if 'err_opti_mm' in df.columns and df['err_opti_mm'].notna().any()
      else 'err_mm'
  )
  TOLERANCIA_MM = 10

  fig7, ax7 = plt.subplots(figsize=(10, 6))
  colores_intentos = plt.get_cmap('tab10').colors
  for intento_idx, grupo in df.groupby('intento_idx', sort=True):
    t_intento = grupo['t_epoch'].values
    t_rel_intento = t_intento - t_intento[0]
    color = colores_intentos[int(intento_idx) % len(colores_intentos)]
    ax7.plot(
        t_rel_intento, grupo[col_error], color=color, linewidth=1.3,
        label=f'Attempt {int(intento_idx) + 1}',
    )

  ax7.axhline(
      TOLERANCIA_MM, color='black', linestyle='--', linewidth=1.3,
      label=f'Tolerance ({TOLERANCIA_MM}mm)',
  )
  ax7.set_xlabel('Time since attempt start (s)')
  ax7.set_ylabel('Error (mm)')
  ax7.set_title(f'Error vs. time -- 5 attempts overlaid (Waypoint {WP_IDX_A_VER+1})')
  ax7.grid(True, alpha=0.3)
  ax7.legend(loc='upper right')

plt.tight_layout()
plt.show()
