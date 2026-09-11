"""Barrido de 16 combinaciones (2^4) de ganancias de PINNLossMPC para
train_pred.py, una tras otra. Corre cada combinación con menos épocas que
un entrenamiento completo (para que el barrido entero sea viable), guarda
un checkpoint y un log por combinación, y al final imprime un resumen
ordenado por 'Mejor Val MSE' para elegir la mejor combinación antes de
hacer el entrenamiento completo (100 épocas) con esa.

4 factores binarios (2x2x2x2 = 16 combinaciones):
  A) w_direction_3d: {0.0 (apagado), 30.0 (encendido)} -- la pérdida de
     dirección/incremental en 3D completo (X, Y, Z).
  B) pesos_ejes_mse: {(1,1,1) uniforme -- "posición relativa a la base nada
     más", (1.5,1.0,2.2) ponderado priorizando X/Z}.
  C) w_speed + w_smooth: {(0,0) apagado, (5.0,2.0) encendido} -- la otra
     pareja de términos "incrementales" (velocidad máxima por paso + jerk).
  D) w_mse: {50.0, 100.0}.

Uso:
  python run_experimentos_pesos_loss.py
  python run_experimentos_pesos_loss.py --epochs 30
  python run_experimentos_pesos_loss.py --resume_from c09
"""
import argparse
import itertools
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / 'logs_experimentos_pesos_loss'
LOG_DIR.mkdir(exist_ok=True)

# Mismo problema de emojis en pipe que en run_experimentos_ventana.py -- forzar UTF-8.
ENV_SUBPROCESO = os.environ.copy()
ENV_SUBPROCESO['PYTHONIOENCODING'] = 'utf-8'
ENV_SUBPROCESO['PYTHONUTF8'] = '1'

DIRECTION_3D_OPCIONES = [0.0, 30.0]
PESOS_EJES_OPCIONES = [(1.0, 1.0, 1.0), (1.5, 1.0, 2.2)]
SPEED_SMOOTH_OPCIONES = [(0.0, 0.0), (5.0, 2.0)]
W_MSE_OPCIONES = [50.0, 100.0]

RE_BEST_VAL_MSE = re.compile(r'Mejor Val MSE:\s*([\d.eE+-]+)')


def construir_plan():
  plan = []
  combos = itertools.product(
      DIRECTION_3D_OPCIONES, PESOS_EJES_OPCIONES, SPEED_SMOOTH_OPCIONES, W_MSE_OPCIONES
  )
  for idx, (w_dir, pesos_ejes, (w_speed, w_smooth), w_mse) in enumerate(combos, start=1):
    etiqueta = f'c{idx:02d}'
    plan.append({
        'etiqueta': etiqueta,
        'output': f'best_mpc_pinn_predictor_sweep_{etiqueta}.pth',
        'w_mse': w_mse,
        'w_direction_3d': w_dir,
        'w_speed': w_speed,
        'w_smooth': w_smooth,
        'pesos_ejes_mse': pesos_ejes,
    })
  return plan


def ejecutar(tarea, epochs):
  etiqueta = tarea['etiqueta']
  log_path = LOG_DIR / f'{etiqueta}.txt'
  cmd = [
      sys.executable, '-u', str(BASE_DIR / 'train_pred.py'),
      '--output', tarea['output'],
      '--epochs', str(epochs),
      '--w-mse', str(tarea['w_mse']),
      '--w-direction-3d', str(tarea['w_direction_3d']),
      '--w-speed', str(tarea['w_speed']),
      '--w-smooth', str(tarea['w_smooth']),
      '--pesos-ejes-mse', *[str(v) for v in tarea['pesos_ejes_mse']],
  ]

  resumen_cfg = (
      f"w_mse={tarea['w_mse']} | w_direction_3d={tarea['w_direction_3d']} | "
      f"w_speed={tarea['w_speed']} | w_smooth={tarea['w_smooth']} | "
      f"pesos_ejes_mse={tarea['pesos_ejes_mse']}"
  )
  print(f"\n{'=' * 70}\n▶ {etiqueta}: {resumen_cfg}\n{'=' * 70}")

  inicio = datetime.now()
  best_val_mse = None
  with open(log_path, 'w', encoding='utf-8') as log_file:
    log_file.write(f"Comando: {' '.join(cmd)}\nConfig: {resumen_cfg}\nInicio: {inicio}\n\n")
    log_file.flush()

    proceso = subprocess.Popen(
        cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', bufsize=1, env=ENV_SUBPROCESO,
    )
    for linea in proceso.stdout:
      print(linea, end='')
      log_file.write(linea)
      log_file.flush()
      m = RE_BEST_VAL_MSE.search(linea)
      if m:
        best_val_mse = float(m.group(1))
    codigo_salida = proceso.wait()

    fin = datetime.now()
    log_file.write(f"\nFin: {fin} | Duración: {fin - inicio} | Código salida: {codigo_salida}\n")

  # El código de salida NO es un criterio confiable acá: PyTorch+CUDA en
  # Windows suele terminar el proceso con STATUS_ACCESS_VIOLATION
  # (0xC0000005 = 3221226505) durante la limpieza del contexto CUDA al
  # salir, AUNQUE el entrenamiento haya completado y guardado bien el
  # checkpoint -- por eso el criterio real de éxito es si se pudo parsear
  # 'Mejor Val MSE' del log (que solo se imprime tras terminar el loop
  # completo de épocas sin excepciones).
  ok = best_val_mse is not None
  if ok and codigo_salida != 0:
    print(f'✅ {etiqueta} completado (código de salida {codigo_salida} tras el cierre de CUDA, '
          f'ignorado). Mejor Val MSE: {best_val_mse} | Log: {log_path}')
  elif ok:
    print(f'✅ {etiqueta} completado. Mejor Val MSE: {best_val_mse} | Log: {log_path}')
  else:
    print(f'⚠️  {etiqueta} terminó con código {codigo_salida} SIN completar el entrenamiento '
          f'(no se encontró "Mejor Val MSE" en el log). Revisa {log_path}')

  return ok, best_val_mse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=30,
                       help='Épocas por combinación durante el barrido (por defecto 30, '
                            'menos que las 100 de un entrenamiento completo, para que el '
                            'barrido de 16 combinaciones sea viable en tiempo razonable).')
  parser.add_argument('--resume_from', type=str, default=None,
                       help="Reanuda desde una etiqueta ('c01'..'c16'), omitiendo las anteriores.")
  args = parser.parse_args()

  plan = construir_plan()

  if args.resume_from:
    idx = next((i for i, t in enumerate(plan) if t['etiqueta'] == args.resume_from), None)
    if idx is None:
      print(f"⚠️  No se encontró '{args.resume_from}' en el plan. Nada que ejecutar.")
      return
    omitidas = plan[:idx]
    plan = plan[idx:]
    print(f'⏭  Reanudando desde {args.resume_from}: se omiten {len(omitidas)} combinaciones.')

  print(f'\n🧪 Barrido de {len(plan)} combinaciones, {args.epochs} épocas cada una.\n')

  resultados = []
  for tarea in plan:
    ok, best_val_mse = ejecutar(tarea, args.epochs)
    resultados.append({**tarea, 'ok': ok, 'best_val_mse': best_val_mse})

  print(f"\n{'=' * 70}\nRESUMEN DEL BARRIDO (ordenado por Val MSE, mejor primero)\n{'=' * 70}")
  completados = [r for r in resultados if r['ok'] and r['best_val_mse'] is not None]
  fallidos = [r for r in resultados if not r['ok'] or r['best_val_mse'] is None]

  completados.sort(key=lambda r: r['best_val_mse'])
  for r in completados:
    print(f"{r['etiqueta']}: Val MSE={r['best_val_mse']:.6f} | w_mse={r['w_mse']} | "
          f"w_direction_3d={r['w_direction_3d']} | w_speed={r['w_speed']} | "
          f"w_smooth={r['w_smooth']} | pesos_ejes_mse={r['pesos_ejes_mse']} | "
          f"checkpoint={r['output']}")

  if fallidos:
    print(f"\n⚠️  {len(fallidos)} combinaciones sin resultado (revisar logs):")
    for r in fallidos:
      print(f"  {r['etiqueta']} -> {LOG_DIR / (r['etiqueta'] + '.txt')}")

  if completados:
    mejor = completados[0]
    print(f"\n🏆 Mejor combinación: {mejor['etiqueta']} (Val MSE={mejor['best_val_mse']:.6f})")
    print(f"   Para reentrenar completo (100 épocas) con esta combinación:")
    print(f"   python train_pred.py --output best_mpc_pinn_predictor_final.pth --epochs 100 "
          f"--w-mse {mejor['w_mse']} --w-direction-3d {mejor['w_direction_3d']} "
          f"--w-speed {mejor['w_speed']} --w-smooth {mejor['w_smooth']} "
          f"--pesos-ejes-mse {' '.join(str(v) for v in mejor['pesos_ejes_mse'])}")


if __name__ == '__main__':
  main()
