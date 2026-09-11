"""Corre train_v1/v2/v7/v8 con varias configuraciones, una tras otra.

FASE 1 — barrido de WINDOW_SIZE con la configuración base (arquitectura y LR
actuales de cada script): 45, 60, 75, 90, 105, 120.

FASE 2 — con la ventana por defecto de cada script (90), prueba 3
configuraciones alternativas de arquitectura/LR, pensadas para no tocar los
pesos de la loss (w_pinn, w_sph_*, w_cyl_*) que ya causaron el colapso al
centro en una corrida anterior:
  - mayor_capacidad: hidden_size 128->192, dropout 0.2->0.25 (compensa la
    capacidad extra para no sobreajustar).
  - mas_profunda:    num_layers 2->3, dropout 0.2->0.3 (misma razón).
  - lr_bajo:         learning rate a la mitad del valor base del script,
    para ver si converge a un mínimo mejor con pasos más finos.

Cada combinación se lanza en un subproceso con --window_size/--output (y en
fase 2 también --hidden_size/--num_layers/--dropout/--lr), y toda la salida
(incluidas las líneas de "Época [...] -> ...") se escribe en tiempo real en
un .txt dentro de logs_experimentos/, además de mostrarse en consola.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / 'logs_experimentos'
LOG_DIR.mkdir(exist_ok=True)

# Al redirigir stdout a un pipe, Windows hace que el hijo use el codepage
# ANSI local (cp1252) en vez de UTF-8, lo que revienta los prints con
# emojis (🔥, 📦, ⭐, ...). Forzamos UTF-8 en el entorno del subproceso.
ENV_SUBPROCESO = os.environ.copy()
ENV_SUBPROCESO['PYTHONIOENCODING'] = 'utf-8'
ENV_SUBPROCESO['PYTHONUTF8'] = '1'

WINDOW_SIZES = [45, 60, 75, 90, 105, 120]
WINDOW_FIJO_FASE2 = 90

# script, plantilla de salida fase 1 ({w}=ventana), plantilla fase 2
# ({w}=ventana, {cfg}=nombre de config), lr base del script
EXPERIMENTOS = [
    {
        'script': 'train_v1_time.py',
        'plantilla_f1': 'soft_robot_lstm_v1_18_w{w}_time.pth',
        'plantilla_f2': 'soft_robot_lstm_v1_18_w{w}_{cfg}.pth',
        'lr_base': 0.0005,
    },
    {
        'script': 'train_v2_time.py',
        'plantilla_f1': 'soft_robot_lstm_v2_13_w{w}_time_best.pth',
        'plantilla_f2': 'soft_robot_lstm_v2_13_w{w}_{cfg}_best.pth',
        'lr_base': 0.00025,
    },
    {
        'script': 'train_v7_time.py',
        'plantilla_f1': 'soft_robot_lstm_v7_11_w{w}_time.pth',
        'plantilla_f2': 'soft_robot_lstm_v7_11_w{w}_{cfg}.pth',
        'lr_base': 0.0005,
    },
    {
        'script': 'train_v8_time_100_2.py',
        'plantilla_f1': 'soft_robot_lstm_v8_17_w{w}_vel.pth',
        'plantilla_f2': 'soft_robot_lstm_v8_17_w{w}_{cfg}.pth',
        'lr_base': 0.0005,
    },
]

# Configuraciones alternativas de arquitectura/LR para la fase 2.
# Cada override que falte se queda con el default del script (hidden=128,
# num_layers=2, dropout=0.2). 'lr_factor' se multiplica por el lr_base.
CONFIGS_EXTRA = [
    {'nombre': 'mayor_capacidad', 'hidden_size': 192, 'dropout': 0.25},
    {'nombre': 'mas_profunda', 'num_layers': 3, 'dropout': 0.3},
    {'nombre': 'lr_bajo', 'lr_factor': 0.5},
]


def ejecutar(script, output_name, extra_args, etiqueta):
  log_path = LOG_DIR / f"{script.replace('.py', '')}_{etiqueta}.txt"
  cmd = [sys.executable, '-u', str(BASE_DIR / script), '--output', output_name] + extra_args

  print(f"\n{'=' * 70}\n▶ {script} | {etiqueta} | salida={output_name}\n{'=' * 70}")

  inicio = datetime.now()
  with open(log_path, 'w', encoding='utf-8') as log_file:
    log_file.write(f"Comando: {' '.join(cmd)}\nInicio: {inicio}\n\n")
    log_file.flush()

    proceso = subprocess.Popen(
        cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', bufsize=1, env=ENV_SUBPROCESO,
    )
    for linea in proceso.stdout:
      print(linea, end='')
      log_file.write(linea)
      log_file.flush()
    codigo_salida = proceso.wait()

    fin = datetime.now()
    log_file.write(f"\nFin: {fin} | Duración: {fin - inicio} | Código salida: {codigo_salida}\n")

  if codigo_salida != 0:
    print(f"⚠️  {script} ({etiqueta}) terminó con código {codigo_salida}. Revisa {log_path}")
  else:
    print(f"✅ {script} ({etiqueta}) completado. Log: {log_path}")

  return codigo_salida == 0


def construir_plan():
  """Arma la lista completa y ordenada de corridas (fase 1 + fase 2)."""
  plan = []

  # FASE 1: barrido de ventana con configuración base
  for exp in EXPERIMENTOS:
    for w in WINDOW_SIZES:
      plan.append({
          'script': exp['script'],
          'output': exp['plantilla_f1'].format(w=w),
          'args': ['--window_size', str(w)],
          'etiqueta': f'w{w}',
      })

  # FASE 2: configuraciones alternativas en la ventana fija
  for exp in EXPERIMENTOS:
    for cfg in CONFIGS_EXTRA:
      extra_args = ['--window_size', str(WINDOW_FIJO_FASE2)]
      if 'hidden_size' in cfg:
        extra_args += ['--hidden_size', str(cfg['hidden_size'])]
      if 'num_layers' in cfg:
        extra_args += ['--num_layers', str(cfg['num_layers'])]
      if 'dropout' in cfg:
        extra_args += ['--dropout', str(cfg['dropout'])]
      if 'lr_factor' in cfg:
        extra_args += ['--lr', str(exp['lr_base'] * cfg['lr_factor'])]

      plan.append({
          'script': exp['script'],
          'output': exp['plantilla_f2'].format(w=WINDOW_FIJO_FASE2, cfg=cfg['nombre']),
          'args': extra_args,
          'etiqueta': f"w{WINDOW_FIJO_FASE2}_{cfg['nombre']}",
      })

  return plan


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--resume_from', type=str, default=None,
      help=(
          "Reanuda desde una combinación 'script.py:etiqueta' (p.ej. "
          "'train_v7_time.py:w75'), omitiendo todo lo anterior del plan "
          "por asumirse ya completado en una corrida previa."
      ),
  )
  args = parser.parse_args()

  plan = construir_plan()

  if args.resume_from:
    script_resume, _, etiqueta_resume = args.resume_from.partition(':')
    idx = next(
        (i for i, t in enumerate(plan)
         if t['script'] == script_resume and t['etiqueta'] == etiqueta_resume),
        None,
    )
    if idx is None:
      print(f"⚠️  No se encontró '{args.resume_from}' en el plan. Nada que ejecutar.")
      return
    omitidas = plan[:idx]
    plan = plan[idx:]
    print(
        f"⏭  Reanudando desde {args.resume_from}: se omiten {len(omitidas)} "
        'combinaciones asumidas como ya completadas.'
    )

  resumen = []
  for tarea in plan:
    ok = ejecutar(tarea['script'], tarea['output'], tarea['args'], tarea['etiqueta'])
    resumen.append((tarea['script'], tarea['etiqueta'], tarea['output'], ok))

  print(f"\n{'=' * 70}\nRESUMEN FINAL\n{'=' * 70}")
  for script, etiqueta, output_name, ok in resumen:
    estado = '✅' if ok else '❌'
    print(f'{estado} {script:<25} {etiqueta:<22} -> {output_name}')


if __name__ == '__main__':
  main()
