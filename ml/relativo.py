import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projector
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# 📁 Set your file names here
ARCHIVO_ENTRADA = "largo_20260730_191948_808.csv"
ARCHIVO_SALIDA = "waypoints_relativos_largo.csv"

# Column 3 (0-indexed) of the raw CSV is 'combo_id' -- same layout as
# dataset_pred_filt.py: 0=timestamp,1=t_unix,2=t_relativo,3=combo_id,
# 4=valid_lecture,5-8=meta_mX,9-12=real_mX,... -- is the "order" number
# (combo_actual_idx in main.py) that stays fixed while the robot
# moves/settles toward that command, and changes when moving to the next one.
COL_COMBO_ID = 3


def procesar_csv(input_path, output_path):
  try:
    # 1. Load the CSV (assuming data without a header or handling it automatically)
    df = pd.read_csv(input_path, header=None)
  except Exception as e:
    print(f"❌ Error al abrir el archivo: {e}")
    return

  # Discard non-numeric rows (e.g. a stray header) using the
  # combo_id column as reference.
  combo_ids_num = pd.to_numeric(df.iloc[:, COL_COMBO_ID], errors='coerce')
  df = df[combo_ids_num.notna()].copy()
  df['_combo_id'] = combo_ids_num[combo_ids_num.notna()].astype(int)

  resultados = []
  combos_omitidos = 0

  # 2. Group by ORDER (combo_id), preserving the order of appearance
  # (sort=False -- do not reorder by value, respect the actual sequence in which
  # the commands were sent), and take the THIRD-TO-LAST row of each
  # group (3rd from the end): neither the last one (may already be mixed with
  # the transition to the next combo) nor a very early one (still
  # moving toward the target) -- a settled point without risking the edge.
  for combo_id, grupo in df.groupby('_combo_id', sort=False):
    if len(grupo) < 3:
      combos_omitidos += 1
      continue
    try:
      row = grupo.iloc[-3]

      # ACTUAL position of the 4 motors (encoder ticks) at that instant
      real_m = row.iloc[9:13].values.astype(float)

      # Extract Base data (Position and Quaternion)
      p_base = row.iloc[21:24].values.astype(float)
      q_base = row.iloc[24:28].values.astype(float)  # [qx, qy, qz, qw]

      # Extract End Effector data (Position)
      p_efector = row.iloc[28:31].values.astype(float)

      # Relative Transformation (end-effector position with respect to the
      # BASE frame): same formula as in dataset_pred_filt.py/MPC.py --
      # rotate the position difference by the inverse of the base
      # rotation, to express it in ITS reference frame, not in the
      # OptiTrack world frame.
      r_b = R.from_quat(q_base)
      p_rel_m = r_b.inv().apply(p_efector - p_base)
      p_rel_mm = p_rel_m * 1000.0  # Convert to millimeters

      # Save processed row (order + motors on the left, target on the right)
      resultados.append({
          'orden': int(combo_id),
          'real_m1_ticks': int(round(real_m[0])),
          'real_m2_ticks': int(round(real_m[1])),
          'real_m3_ticks': int(round(real_m[2])),
          'real_m4_ticks': int(round(real_m[3])),
          'target_x_mm': round(p_rel_mm[0], 2),
          'target_y_mm': round(p_rel_mm[1], 2),
          'target_z_mm': round(p_rel_mm[2], 2),
      })
    except (ValueError, IndexError):
      # Skip the combo if its third-to-last row has text/invalid data
      combos_omitidos += 1
      continue

  if combos_omitidos:
    print(f"⚠️ {combos_omitidos} orden(es) omitida(s) (menos de 3 lecturas o datos inválidos).")

  # 3. Create the new DataFrame
  df_salida = pd.DataFrame(resultados)

  # 4. Save to a new CSV without unnecessary columns
  df_salida.to_csv(output_path, index=False)

  print(f"✓ ¡Procesamiento completado con éxito!")
  print(f"✓ Guardado en: {output_path}\n")
  print(f"📌 {len(df_salida)} puntos generados (uno por orden -- antepenúltima lectura de cada combo).")
  print("📌 Vista previa del resultado (Primeras 10 filas):")
  print("=" * 45)
  print(df_salida.head(10).to_string(index=False))

  # -------------------------------------------------------------------
  # 5. 3D plot: one point per order (third-to-last reading), colored
  # by order number so that one can identify which combo each one
  # belongs to without cluttering the plot with 625 text labels.
  # -------------------------------------------------------------------
  if not df_salida.empty:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        df_salida['target_x_mm'], df_salida['target_y_mm'], df_salida['target_z_mm'],
        c=df_salida['orden'], cmap='plasma', s=18,
        picker=True, pickradius=5,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label('Número de orden (combo_id)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Antepenúltimo punto de cada orden ({len(df_salida)} puntos) -- click en un '
                 f'punto para ver su info')
    plt.tight_layout()

    # ---------------------------------------------------------------
    # Interactive selection: click on a point -> prints to the console and
    # shows in the plot itself which order/tick combination it comes from
    # (only works in the interactive window, not on an exported
    # image). The chosen point is highlighted in red to confirm which one
    # was selected.
    # ---------------------------------------------------------------
    info_text = ax.text2D(
        0.02, 0.98, 'Click en un punto para ver su información',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )
    resaltado = {'artist': None}

    def on_pick(event):
      if event.artist != sc or not len(event.ind):
        return
      i = event.ind[0]  # the closest one to the click, if several overlap there
      fila = df_salida.iloc[i]
      msg = (
          f"Orden {int(fila['orden'])}\n"
          f"Ticks m1-m4: ({int(fila['real_m1_ticks'])}, {int(fila['real_m2_ticks'])}, "
          f"{int(fila['real_m3_ticks'])}, {int(fila['real_m4_ticks'])})\n"
          f"Posición: ({fila['target_x_mm']:.1f}, {fila['target_y_mm']:.1f}, "
          f"{fila['target_z_mm']:.1f}) mm"
      )
      print(f"🖱️ {msg.replace(chr(10), ' | ')}")
      info_text.set_text(msg)

      if resaltado['artist'] is not None:
        resaltado['artist'].remove()
      resaltado['artist'] = ax.scatter(
          [fila['target_x_mm']], [fila['target_y_mm']], [fila['target_z_mm']],
          color='red', s=90, edgecolors='black', linewidths=1.2, zorder=10,
      )
      fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()


if __name__ == "__main__":
  procesar_csv(ARCHIVO_ENTRADA, ARCHIVO_SALIDA)
