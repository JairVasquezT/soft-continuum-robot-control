"""Goes through, ONE BY ONE, all the checkpoints produced by
run_experimentos_ventana.py and opens their validation plot (dataset_valid_
v1/v2/v7/v8_time_filt.py) against the 'corto' dataset (independent recording,
never seen during training).

Each matplotlib window opens in the foreground and BLOCKS execution
(plt.show() is blocking) -- you can rotate the 3D view, adjust the view and save
the image manually with matplotlib's save button. When the window is
CLOSED, this script automatically moves on to the next checkpoint.

The plan (which checkpoints exist, with which architecture they were trained) is
reused directly from run_experimentos_ventana.py -- same source of
truth, so that file names and hyperparameters (hidden_size,
num_layers) never get out of sync between training and this review.

Usage:
  python ver_resultados_experimentos.py
  python ver_resultados_experimentos.py --resume_from "v7:w75"
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ENV_SUBPROCESO = os.environ.copy()
ENV_SUBPROCESO['PYTHONIOENCODING'] = 'utf-8'
ENV_SUBPROCESO['PYTHONUTF8'] = '1'

FAMILIA_POR_SCRIPT_ENTRENAMIENTO = {
    'train_v1_time.py': 'v1',
    'train_v2_time.py': 'v2',
    'train_v7_time.py': 'v7',
    'train_v8_time_100_2.py': 'v8',
}

SCRIPT_VALIDACION_POR_FAMILIA = {
    'v1': 'dataset_valid_v1_time_filt.py',
    'v2': 'dataset_valid_v2_time_filt.py',
    'v7': 'dataset_valid_v7_time_filt.py',
    'v8': 'dataset_valid_v8_time_filt.py',
}


def _cargar_plan_entrenamiento():
  """Imports run_experimentos_ventana.py as a module (without running its
  main()) to reuse EXPERIMENTOS/CONFIGS_EXTRA/WINDOW_SIZES as is."""
  spec = importlib.util.spec_from_file_location(
      'runner_entrenamiento', BASE_DIR / 'run_experimentos_ventana.py'
  )
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def construir_plan_visualizacion(runner):
  """Same phase1/phase2 structure as training, but pointing to the
  corresponding VALIDATION script and without --output/--lr (not applicable)."""
  plan = []

  for exp in runner.EXPERIMENTOS:
    familia = FAMILIA_POR_SCRIPT_ENTRENAMIENTO[exp['script']]
    validador = SCRIPT_VALIDACION_POR_FAMILIA[familia]

    # PHASE 1: window sweep, base architecture (hidden=128, layers=2)
    for w in runner.WINDOW_SIZES:
      plan.append({
          'familia': familia,
          'validador': validador,
          'checkpoint': exp['plantilla_f1'].format(w=w),
          'window_size': w,
          'hidden_size': 128,
          'num_layers': 2,
          'etiqueta': f'w{w}',
      })

    # PHASE 2: alternative configurations on the fixed window
    for cfg in runner.CONFIGS_EXTRA:
      plan.append({
          'familia': familia,
          'validador': validador,
          'checkpoint': exp['plantilla_f2'].format(
              w=runner.WINDOW_FIJO_FASE2, cfg=cfg['nombre']
          ),
          'window_size': runner.WINDOW_FIJO_FASE2,
          'hidden_size': cfg.get('hidden_size', 128),
          'num_layers': cfg.get('num_layers', 2),
          'etiqueta': f"w{runner.WINDOW_FIJO_FASE2}_{cfg['nombre']}",
      })

  return plan


def mostrar(tarea, indice, total):
  titulo = f"{tarea['familia']} | {tarea['etiqueta']}"
  print(f"\n{'=' * 70}\n▶ [{indice}/{total}] {titulo} "
        f"(checkpoint={tarea['checkpoint']})\n{'=' * 70}")

  cmd = [
      sys.executable, str(BASE_DIR / tarea['validador']),
      '--model', tarea['checkpoint'],
      '--window_size', str(tarea['window_size']),
      '--hidden_size', str(tarea['hidden_size']),
      '--num_layers', str(tarea['num_layers']),
      '--title', titulo,
  ]
  # Without capturing stdout/stderr: they inherit the real console (lets the
  # matplotlib graphical window display normally) and avoids the
  # UnicodeEncodeError from prints with emoji that does appear when redirecting
  # to a pipe (see run_experimentos_ventana.py).
  resultado = subprocess.run(cmd, cwd=str(BASE_DIR), env=ENV_SUBPROCESO)

  if resultado.returncode != 0:
    print(f"⚠️  {titulo}: el script de validación terminó con código "
          f"{resultado.returncode} (revisa el error arriba).")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--resume_from', type=str, default=None,
      help="Reanuda desde 'familia:etiqueta' (p.ej. 'v7:w75'), omitiendo "
           'todo lo anterior del plan.'
  )
  args = parser.parse_args()

  runner = _cargar_plan_entrenamiento()
  plan = construir_plan_visualizacion(runner)

  if args.resume_from:
    familia_resume, _, etiqueta_resume = args.resume_from.partition(':')
    idx = next(
        (i for i, t in enumerate(plan)
         if t['familia'] == familia_resume and t['etiqueta'] == etiqueta_resume),
        None,
    )
    if idx is None:
      print(f"⚠️  No se encontró '{args.resume_from}' en el plan. Nada que mostrar.")
      return
    omitidas = plan[:idx]
    plan = plan[idx:]
    print(f"⏭  Reanudando desde {args.resume_from}: se omiten {len(omitidas)} "
          'combinaciones ya revisadas.')

  # Filter out checkpoints that do not exist on disk yet (incomplete sweep or
  # failed runs) -- warn and continue, do not stop the entire review.
  disponibles, faltantes = [], []
  for tarea in plan:
    if (BASE_DIR / tarea['checkpoint']).exists():
      disponibles.append(tarea)
    else:
      faltantes.append(tarea)

  if faltantes:
    print(f"⚠️  {len(faltantes)} checkpoint(s) del plan no existen en disco todavía "
          '(se omiten):')
    for t in faltantes:
      print(f"    - {t['familia']} | {t['etiqueta']} -> {t['checkpoint']}")

  total = len(disponibles)
  print(f"\n📋 {total} gráfico(s) para revisar. Cerrá cada ventana para pasar al siguiente.\n")

  for i, tarea in enumerate(disponibles, start=1):
    mostrar(tarea, i, total)

  print(f"\n{'=' * 70}\n✅ Revisión completa: {total} gráficos mostrados.\n{'=' * 70}")


if __name__ == '__main__':
  main()
