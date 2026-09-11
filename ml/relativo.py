import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registra el proyector 3D
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# 📁 Configura aquí los nombres de tus archivos
ARCHIVO_ENTRADA = "largo_20260730_191948_808.csv"
ARCHIVO_SALIDA = "waypoints_relativos_largo.csv"

# Columna 3 (0-indexada) del CSV crudo es 'combo_id' -- mismo layout que
# dataset_pred_filt.py: 0=timestamp,1=t_unix,2=t_relativo,3=combo_id,
# 4=valid_lecture,5-8=meta_mX,9-12=real_mX,... -- es el número de "orden"
# (combo_actual_idx en main.py) que se mantiene fijo mientras el robot se
# mueve/asienta hacia ese comando, y cambia al pasar al siguiente.
COL_COMBO_ID = 3


def procesar_csv(input_path, output_path):
  try:
    # 1. Cargar el CSV (asumiendo datos sin encabezado o manejándolo automáticamente)
    df = pd.read_csv(input_path, header=None)
  except Exception as e:
    print(f"❌ Error al abrir el archivo: {e}")
    return

  # Descartar filas no numéricas (p.ej. un encabezado colado) usando la
  # columna de combo_id como referencia.
  combo_ids_num = pd.to_numeric(df.iloc[:, COL_COMBO_ID], errors='coerce')
  df = df[combo_ids_num.notna()].copy()
  df['_combo_id'] = combo_ids_num[combo_ids_num.notna()].astype(int)

  resultados = []
  combos_omitidos = 0

  # 2. Agrupar por ORDEN (combo_id), preservando el orden de aparición
  # (sort=False -- no reordenar por valor, respetar la secuencia real en la
  # que se mandaron los comandos), y tomar el ANTEPENÚLTIMO renglón de cada
  # grupo (3ro desde el final): ni el último (puede ya estar mezclado con
  # la transición hacia el siguiente combo) ni uno muy temprano (todavía en
  # movimiento hacia el target) -- un punto asentado sin arriesgar el borde.
  for combo_id, grupo in df.groupby('_combo_id', sort=False):
    if len(grupo) < 3:
      combos_omitidos += 1
      continue
    try:
      row = grupo.iloc[-3]

      # Posición REAL de los 4 motores (ticks del encoder) en ese instante
      real_m = row.iloc[9:13].values.astype(float)

      # Extraer datos de la Base (Posición y Cuaternión)
      p_base = row.iloc[21:24].values.astype(float)
      q_base = row.iloc[24:28].values.astype(float)  # [qx, qy, qz, qw]

      # Extraer datos del Efector Final (Posición)
      p_efector = row.iloc[28:31].values.astype(float)

      # Transformación Relativa (posición del efector respecto al marco de
      # la BASE): igual fórmula que en dataset_pred_filt.py/MPC.py --
      # rotar la diferencia de posiciones por la inversa de la rotación de
      # la base, para expresarla en SU sistema de referencia, no en el
      # marco mundo de OptiTrack.
      r_b = R.from_quat(q_base)
      p_rel_m = r_b.inv().apply(p_efector - p_base)
      p_rel_mm = p_rel_m * 1000.0  # Convertir a milímetros

      # Guardar fila procesada (orden + motores a la izquierda, target a la derecha)
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
      # Omite el combo si su antepenúltima fila trae texto/datos inválidos
      combos_omitidos += 1
      continue

  if combos_omitidos:
    print(f"⚠️ {combos_omitidos} orden(es) omitida(s) (menos de 3 lecturas o datos inválidos).")

  # 3. Crear el nuevo DataFrame
  df_salida = pd.DataFrame(resultados)

  # 4. Guardar en un nuevo CSV sin columnas innecesarias
  df_salida.to_csv(output_path, index=False)

  print(f"✓ ¡Procesamiento completado con éxito!")
  print(f"✓ Guardado en: {output_path}\n")
  print(f"📌 {len(df_salida)} puntos generados (uno por orden -- antepenúltima lectura de cada combo).")
  print("📌 Vista previa del resultado (Primeras 10 filas):")
  print("=" * 45)
  print(df_salida.head(10).to_string(index=False))

  # -------------------------------------------------------------------
  # 5. Gráfico 3D: un punto por orden (antepenúltima lectura), coloreado
  # por número de orden para poder identificar a qué combo pertenece cada
  # uno sin saturar el gráfico con 625 etiquetas de texto.
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
    # Selección interactiva: click en un punto -> imprime en consola y
    # muestra en el propio gráfico de qué orden/combinación de ticks viene
    # (solo funciona en la ventana interactiva, no sobre una imagen
    # exportada). Se resalta el punto elegido en rojo para confirmar cuál
    # quedó seleccionado.
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
      i = event.ind[0]  # el más cercano al click, si hay varios superpuestos ahí
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
