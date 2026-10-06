"""Main script: generates combinations and moves the Dynamixel motors, logging at high frequency (OptiTrack Hz).

Usage: run `python -m continuum_robot.main --no-simulate` from the project folder.
"""
import os
import sys

# Allow running `python continuum_robot/main.py` directly by adding
# the project root folder to sys.path when the package is not on PATH.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time
import argparse
import threading
import queue
from collections import deque
from datetime import datetime

from continuum_robot.config import robot_config as cfg
from continuum_robot.hardware.dynamixel import create_controller
from continuum_robot.control.trajectories import (
    all_combinations,
    candidatos_exploracion_local,
    generar_puntos_principales,
    resumen_tiempo_estimado,
)
from continuum_robot.data.logger import CSVLogger
from continuum_robot.utils.calibration_sequential import (
    calibrate_all_motors_sequential,
)
from continuum_robot.hardware.galga import PhidgetForceController

# 🎯 OFFICIAL NATNET CLIENT
from NatNetClient import NatNetClient

# ==============================================================================
# 🎯 UNIFIED FREQUENCY CONFIGURATION FOR THE EXPERIMENT
# ==============================================================================
FRECUENCIA_LOG = 60  # Hz
LOG_INTERVAL = 1.0 / FRECUENCIA_LOG
MOVE_TIMEOUT = 6.5  # Maximum wait time per instruction (seconds)

combo_actual_idx = 0
posiciones_objetivo_actuales = [0.0] * len(cfg.MOTOR_IDS)
controlador_dynamixel = None
logger_alta_frecuencia = None
time_start = 0.0

# 🔒 HARDWARE LOCK: Protects the serial data bus
hardware_lock = threading.Lock()

# 🚀 RIGID BODY VARIABLES (OPTITRACK)
ultima_pos_base = [0.0, 0.0, 0.0]
ultima_rot_base = [0.0, 0.0, 0.0, 1.0]
ultima_pos_efector = [0.0, 0.0, 0.0]
ultima_rot_efector = [0.0, 0.0, 0.0, 1.0]

ultimas_posiciones_validas = [0] * len(cfg.MOTOR_IDS)
ultimos_torques_validos = [0] * len(cfg.MOTOR_IDS)
ultimas_fuerzas_validas = [0.0] * 4
ultima_lectura_valida = 0

# 🔄 PRODUCER-CONSUMER PATTERN
log_queue = queue.Queue(maxsize=0)  # Thread-safe queue for log rows
consumer_running = False
sampler_running = False
controlador_phidget = None


def receive_rigid_body_frame(new_id, position, rotation):
    """PRODUCER: Receives frames from OptiTrack."""
    global ultima_pos_base, ultima_rot_base, ultima_pos_efector, ultima_rot_efector
    
    if new_id == 1:
        ultima_pos_base = position
        ultima_rot_base = rotation
    elif new_id == 2:
        ultima_pos_efector = position
        ultima_rot_efector = rotation


def data_sampler_thread(freq_hz=60):
    """Thread with a strict 60Hz clock. Reads Motors (Pos+Torque) + Phidget + OptiTrack."""
    global sampler_running, ultimas_posiciones_validas, ultimos_torques_validos, ultimas_fuerzas_validas, ultima_lectura_valida
    global combo_actual_idx, posiciones_objetivo_actuales, controlador_dynamixel, controlador_phidget, time_start, log_queue

    intervalo = 1.0 / freq_hz

    while sampler_running:
        t_loop_start = time.time()
        t_relativo = t_loop_start - time_start

        # 1. Dynamixel Motors reading (Position and Torque/Current)
        if controlador_dynamixel is not None:
            with hardware_lock:
                try:
                    # Use the new method defined in dynamixel.py
                    posiciones_reales, torques_reales, lectura_valida = controlador_dynamixel.sync_get_present_position_and_load(cfg.MOTOR_IDS)
                except Exception as e:
                    # Print the error if it fails again instead of silently hiding it
                    print(f"⚠️ Error leyendo Dynamixel: {e}")
                    posiciones_reales = list(ultimas_posiciones_validas)
                    torques_reales = list(ultimos_torques_validos)
                    lectura_valida = 0
        else:
            posiciones_reales = list(ultimas_posiciones_validas)
            torques_reales = list(ultimos_torques_validos)
            lectura_valida = 0

        if lectura_valida == 1:
            ultimas_posiciones_validas = list(posiciones_reales)
            ultimos_torques_validos = list(torques_reales)
            ultima_lectura_valida = 1
        else:
            ultima_lectura_valida = 0

        # 2. Phidget Sensor reading (4 load cells)
        if controlador_phidget is not None:
            try:
                fuerzas_g = controlador_phidget.leer_fuerzas_gramos()
                ultimas_fuerzas_validas = fuerzas_g
            except Exception:
                fuerzas_g = list(ultimas_fuerzas_validas)
        else:
            fuerzas_g = list(ultimas_fuerzas_validas)

        # 3. Pack extended row for the CSV
        log_row = (
            [t_loop_start, t_relativo, combo_actual_idx, lectura_valida]
            + posiciones_objetivo_actuales           # 4 Targets
            + posiciones_reales                      # 4 Real positions
            + torques_reales                         # 4 Real Torques/Currents
            + fuerzas_g                              # 4 Load cells (grams)
            + list(ultima_pos_base) + list(ultima_rot_base)     # OptiTrack Base (7)
            + list(ultima_pos_efector) + list(ultima_rot_efector) # OptiTrack End effector (7)
        )

        try:
            log_queue.put_nowait(log_row)
        except queue.Full:
            pass

        # 4. Sampling period control at 60 Hz
        t_ejecucion = time.time() - t_loop_start
        tiempo_espera = intervalo - t_ejecucion
        if tiempo_espera > 0:
            time.sleep(tiempo_espera)


def log_consumer_thread():
    """CONSUMER: Reads from the queue and writes to the CSV in a separate thread."""
    global logger_alta_frecuencia, log_queue, consumer_running
    
    while consumer_running:
        try:
            log_row = log_queue.get(timeout=0.1)
            if log_row is not None and logger_alta_frecuencia is not None:
                logger_alta_frecuencia.log(log_row)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error en log_consumer_thread: {e}")


def run_sequence(simulate: bool = True, delay: float = 0.1, dynamic_torque: bool = False):
    global combo_actual_idx, posiciones_objetivo_actuales, controlador_dynamixel, logger_alta_frecuencia, time_start
    global consumer_running, sampler_running, log_queue, controlador_phidget

    streaming_client = None

    try:
        controlador_dynamixel = create_controller(simulate=simulate, port=cfg.SERIAL_PORT, baudrate=cfg.BAUDRATE)
    except Exception as e:
        print(f'No se pudo inicializar el controlador Dynamixel: {e}')
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    logs_dir = os.path.join('continuum_robot', 'data')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f'logs_{timestamp}.csv')
    logger_alta_frecuencia = CSVLogger(log_path)
    
    ids = cfg.MOTOR_IDS
    time_start = time.time()

    # Start CSV Consumer
    consumer_running = True
    consumer_thread_obj = threading.Thread(target=log_consumer_thread, daemon=True)
    consumer_thread_obj.start()

    

    try:
        if not simulate:
            # 1. OptiTrack (NatNet) connection
            streaming_client = NatNetClient()
            streaming_client.set_client_address(cfg.OPTITRACK_HOST)
            streaming_client.set_server_address(cfg.OPTITRACK_HOST)
            streaming_client.rigid_body_listener = receive_rigid_body_frame
            streaming_client.run()

            # 2. PhidgetBridge Sensor initialization (4 force cells)
            try:
                controlador_phidget = PhidgetForceController()
                print("✓ PhidgetBridge (4 celdas) inicializado correctamente.")
            except Exception as e:
                print(f"⚠️ No se pudo inicializar Phidget: {e}")
                controlador_phidget = None

            # 3. Dynamixel scan and Torque enabling
            if hasattr(controlador_dynamixel, 'scan'):
                with hardware_lock:
                    found = controlador_dynamixel.scan(ids)
                    if found and hasattr(controlador_dynamixel, 'enable_torque'):
                        controlador_dynamixel.enable_torque(found)

        # Start 60 Hz Sampler
        sampler_running = True
        sampler_thread_obj = threading.Thread(target=data_sampler_thread, args=(60,), daemon=True)
        sampler_thread_obj.start()
        print("✓ Hilos de muestreo a 60 Hz y guardado iniciados correctamente.")

        # Wait until a valid signal is available or 1.5 seconds of startup
        start_wait = time.time()
        while time.time() - start_wait < 1.5:
            if ultima_lectura_valida == 1:
                break
            time.sleep(0.01)

        if ultima_lectura_valida != 1:
            print("⚠️ No se detectó lectura válida antes de 1.5 s; empezando la secuencia de todas formas.")

        points_per_motor = getattr(run_sequence, 'points_per_motor', 5)
        combos = list(all_combinations(ids, points_per_motor=points_per_motor))

        
        for i, combo in enumerate(combos, 1):
            positions = [combo[mid] for mid in ids]

            # Before sending the first target, leave the logs at point 0
            if i == 1:
                combo_actual_idx = 0
                posiciones_objetivo_actuales = [0.0] * len(cfg.MOTOR_IDS)
                print("✓ Iniciando secuencia: punto actual 0 sin movimiento de objetivo.")

            # Send movement command to the robot
            with hardware_lock:
                controlador_dynamixel.move(ids, positions, speed=cfg.DEFAULT_SPEED, wait_for_reached=False)

            # Update the real point after sending the command
            combo_actual_idx = i
            posiciones_objetivo_actuales = positions

            # ⏱️ SMART WAIT: Queries in-memory state at ~60Hz
            # History of latest readings to detect absence of change
            pos_history = deque(maxlen=40)

            llegado = False
            timeout_start = time.time()

            while not llegado and (time.time() - timeout_start < MOVE_TIMEOUT):
                time.sleep(0.016)

                pos_actuales = list(ultimas_posiciones_validas)
                # Record in history and detect whether there was no change in the last 40 readings
                try:
                    pos_history.append(tuple(pos_actuales))
                except Exception:
                    # If there is invalid data, continue without adding
                    pass

                if len(pos_history) == pos_history.maxlen:
                    first = pos_history[0]
                    if all(h == first for h in pos_history):
                        print(f"⚠️ Sin cambios en las últimas {pos_history.maxlen} lecturas para los motores; saltando al siguiente objetivo.")
                        llegado = True
                        break

                if pos_actuales and not any(p is None for p in pos_actuales):
                    cerca = all(abs(p - target) <= 20 for p, target in zip(pos_actuales, positions))
                    if cerca:
                        llegado = True

            time.sleep(delay)

    finally:
        # Stop sampling and CSV writing threads
        sampler_running = False
        consumer_running = False
        
        # Disconnect OptiTrack
        if streaming_client:
            streaming_client.shutdown()
        
        # Disconnect Phidget
        if controlador_phidget:
            try:
                controlador_phidget.close()
                print("✓ PhidgetBridge desconectado correctamente.")
            except Exception as e:
                print(f"⚠️ Error al cerrar Phidget: {e}")

        # Disconnect Dynamixel Motors
        if controlador_dynamixel:
            with hardware_lock:
                controlador_dynamixel.close()


def run_trayectoria_final(
    simulate: bool = True,
    delay: float = 0.1,
    niveles: int = 5,
    n_mixtas_exploracion: int = 4,
    offset_exploracion: int = 30,
    segundos_candidato: float = 0.25,
    segundos_punto_base: float = None,
    muestras_ventana_plateau: int = None,
):
    """Separate mode from run_sequence()/--points 3-5: walks through the sequence of
    "main points" from generar_puntos_principales() (shifted grid
    + nearest-neighbor ordering) and, at each one, performs a
    local micro-exploration of ~20 candidates (candidatos_exploracion_local)
    before moving on to the next point.

    `niveles=5` (default): 604 points (filtered by extreme/net push),
    starts near the center, waits up to 0.5s per point (or earlier if there are no
    changes in the last 30 readings -- at 60Hz that is already 0.5s, so
    both criteria coincide in practice).
    `niveles=3`: 81 points (NOT filtered, all are kept), starts far from the
    center (different traversal order from the 604), waits up to 0.8s
    per point (or earlier if there are no changes in the last 40 readings, ~0.67s
    at 60Hz -- here the plateau cutoff normally arrives before the 0.8s
    cap, unlike the 5-level case).
    `segundos_punto_base`/`muestras_ventana_plateau` force those values
    manually if you do not want the defaults above according to `niveles`.

    Each exploration candidate is held for a fixed `segundos_candidato`
    (0.25s by default = 15 steps at 60Hz), without checking arrival.
    """
    global combo_actual_idx, posiciones_objetivo_actuales, controlador_dynamixel, logger_alta_frecuencia, time_start
    global consumer_running, sampler_running, log_queue, controlador_phidget

    if niveles == 5:
        filtrar_puntos = True
        punto_inicio = 'centro'
        segundos_punto_base = 0.5 if segundos_punto_base is None else segundos_punto_base
        muestras_ventana_plateau = 30 if muestras_ventana_plateau is None else muestras_ventana_plateau
    elif niveles == 3:
        filtrar_puntos = False
        punto_inicio = 'extremo'
        segundos_punto_base = 0.8 if segundos_punto_base is None else segundos_punto_base
        muestras_ventana_plateau = 40 if muestras_ventana_plateau is None else muestras_ventana_plateau
    else:
        raise ValueError(f'niveles debe ser 3 o 5, no {niveles!r}')

    streaming_client = None

    try:
        controlador_dynamixel = create_controller(simulate=simulate, port=cfg.SERIAL_PORT, baudrate=cfg.BAUDRATE)
    except Exception as e:
        print(f'No se pudo inicializar el controlador Dynamixel: {e}')
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    logs_dir = os.path.join('continuum_robot', 'data')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f'logs_trayectoria_final_{timestamp}.csv')
    logger_alta_frecuencia = CSVLogger(log_path)

    ids = cfg.MOTOR_IDS
    time_start = time.time()

    # Start CSV Consumer
    consumer_running = True
    consumer_thread_obj = threading.Thread(target=log_consumer_thread, daemon=True)
    consumer_thread_obj.start()

    try:
        if not simulate:
            # 1. OptiTrack (NatNet) connection
            streaming_client = NatNetClient()
            streaming_client.set_client_address(cfg.OPTITRACK_HOST)
            streaming_client.set_server_address(cfg.OPTITRACK_HOST)
            streaming_client.rigid_body_listener = receive_rigid_body_frame
            streaming_client.run()

            # 2. PhidgetBridge Sensor initialization (4 force cells)
            try:
                controlador_phidget = PhidgetForceController()
                print("✓ PhidgetBridge (4 celdas) inicializado correctamente.")
            except Exception as e:
                print(f"⚠️ No se pudo inicializar Phidget: {e}")
                controlador_phidget = None

            # 3. Dynamixel scan and Torque enabling
            if hasattr(controlador_dynamixel, 'scan'):
                with hardware_lock:
                    found = controlador_dynamixel.scan(ids)
                    if found and hasattr(controlador_dynamixel, 'enable_torque'):
                        controlador_dynamixel.enable_torque(found)

        # Start 60 Hz Sampler
        sampler_running = True
        sampler_thread_obj = threading.Thread(target=data_sampler_thread, args=(60,), daemon=True)
        sampler_thread_obj.start()
        print("✓ Hilos de muestreo a 60 Hz y guardado iniciados correctamente.")

        # Wait until a valid signal is available or 1.5 seconds of startup
        start_wait = time.time()
        while time.time() - start_wait < 1.5:
            if ultima_lectura_valida == 1:
                break
            time.sleep(0.01)

        if ultima_lectura_valida != 1:
            print("⚠️ No se detectó lectura válida antes de 1.5 s; empezando la secuencia de todas formas.")

        puntos_principales = generar_puntos_principales(
            n_points=niveles,
            filtrar_por_extremo_o_empuje=filtrar_puntos,
            punto_inicio=punto_inicio,
        )
        resumen = resumen_tiempo_estimado(
            len(puntos_principales),
            n_candidatos_por_punto=16 + n_mixtas_exploracion,
            segundos_por_candidato=segundos_candidato,
            segundos_por_punto_base=segundos_punto_base,
        )
        print(
            f"✓ {resumen['puntos_principales']} puntos principales | "
            f"{resumen['candidatos_totales']} candidatos de exploración | "
            f"~{resumen['horas_total']:.2f}h estimadas (sin contar desplazamiento entre puntos)"
        )

        combo_idx_global = 0
        for i, punto in enumerate(puntos_principales, 1):
            positions = [punto[mid] for mid in ids]

            # Before sending the first target, leave the logs at point 0
            if i == 1:
                combo_actual_idx = 0
                posiciones_objetivo_actuales = [0.0] * len(cfg.MOTOR_IDS)
                print("✓ Iniciando secuencia: punto actual 0 sin movimiento de objetivo.")

            # Send movement command to the main point
            with hardware_lock:
                controlador_dynamixel.move(ids, positions, speed=cfg.DEFAULT_SPEED, wait_for_reached=False)

            combo_idx_global += 1
            combo_actual_idx = combo_idx_global
            posiciones_objetivo_actuales = positions

            # ⏱️ WAIT BOUNDED TO THE BASE POINT: up to segundos_punto_base,
            # cutting short if there are no changes in the last
            # muestras_ventana_plateau readings or if the target has already been reached
            # (same pattern as run_sequence, with a smaller cap and window).
            pos_history = deque(maxlen=muestras_ventana_plateau)
            llegado = False
            timeout_start = time.time()

            while not llegado and (time.time() - timeout_start < segundos_punto_base):
                time.sleep(0.016)

                pos_actuales = list(ultimas_posiciones_validas)
                try:
                    pos_history.append(tuple(pos_actuales))
                except Exception:
                    pass

                if len(pos_history) == pos_history.maxlen:
                    first = pos_history[0]
                    if all(h == first for h in pos_history):
                        llegado = True
                        break

                if pos_actuales and not any(p is None for p in pos_actuales):
                    cerca = all(abs(p - target) <= 20 for p, target in zip(pos_actuales, positions))
                    if cerca:
                        llegado = True

            # 🔬 LOCAL EXPLORATION: ~20 candidates around the main
            # point, each held for a fixed segundos_candidato (without
            # checking arrival).
            candidatos = candidatos_exploracion_local(
                punto, offset=offset_exploracion, n_mixtas=n_mixtas_exploracion,
            )
            for candidato in candidatos:
                cand_positions = [candidato[mid] for mid in ids]

                with hardware_lock:
                    controlador_dynamixel.move(ids, cand_positions, speed=cfg.DEFAULT_SPEED, wait_for_reached=False)

                combo_idx_global += 1
                combo_actual_idx = combo_idx_global
                posiciones_objetivo_actuales = cand_positions

                time.sleep(segundos_candidato)

            time.sleep(delay)

    finally:
        # Stop sampling and CSV writing threads
        sampler_running = False
        consumer_running = False

        # Disconnect OptiTrack
        if streaming_client:
            streaming_client.shutdown()

        # Disconnect Phidget
        if controlador_phidget:
            try:
                controlador_phidget.close()
                print("✓ PhidgetBridge desconectado correctamente.")
            except Exception as e:
                print(f"⚠️ Error al cerrar Phidget: {e}")

        # Disconnect Dynamixel Motors
        if controlador_dynamixel:
            with hardware_lock:
                controlador_dynamixel.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-simulate', action='store_true', help='Desactivar simulación y usar hardware real')
    parser.add_argument('--delay', type=float, default=0.1)
    parser.add_argument('--port', type=str, default=None, help='Puerto serie')
    parser.add_argument('--baud', type=int, default=None, help='Baudrate')
    parser.add_argument('--dynamic-torque', action='store_true', help='Desactivar torque en motores estáticos')
    parser.add_argument('--calibrate-all', action='store_true', help='Calibrar todos los motores')
    parser.add_argument('--calibrate-motor', type=int, choices=cfg.MOTOR_IDS, help='Calibrar un motor específico')
    parser.add_argument('--points', type=int, choices=[3,5], default=5, help='Número de puntos por motor (3 -> 81 combos, 5 -> 625 combos)')
    parser.add_argument('--trayectoria-final', action='store_true',
                         help='Modo aparte de --points: recorre generar_puntos_principales() '
                              '(grilla desplazada + orden por vecino cercano) y hace ~20 '
                              'candidatos de exploración local por punto.')
    parser.add_argument('--niveles-trayectoria-final', type=int, choices=[3, 5], default=5,
                         help='Solo con --trayectoria-final. 5 (default): 604 puntos '
                              'filtrados, arranca cerca del centro, 0.5s/30 muestras por '
                              'punto. 3: 81 puntos SIN filtrar, arranca lejos del centro '
                              '(orden distinto al de 604), 0.8s/40 muestras por punto.')
    args = parser.parse_args()
    simulate = not args.no_simulate
    
    if args.port:
        cfg.SERIAL_PORT = args.port
    if args.baud:
        cfg.BAUDRATE = args.baud
        
    if args.calibrate_all or args.calibrate_motor is not None:
        controller = None
        if not simulate:
            try:
                controller = create_controller(simulate=False, port=cfg.SERIAL_PORT, baudrate=cfg.BAUDRATE)
                with hardware_lock:
                    if hasattr(controller, 'enable_torque'):
                        try:
                            controller.enable_torque(cfg.MOTOR_IDS)
                        except Exception:
                            pass
                    try:
                        home_positions = [cfg.HOME_POSITION[mid] for mid in cfg.MOTOR_IDS]
                        controller.move(cfg.MOTOR_IDS, home_positions, speed=cfg.DEFAULT_SPEED, wait_for_reached=True)
                        print(f"Motores enviados a HOME_POSITION={home_positions}")
                    except Exception as e:
                        print(f"Advertencia: no se pudo mover a home antes de calibrar: {e}")
            except Exception as e:
                print(f'No se pudo inicializar el controlador para pre-movement: {e}')

        try:
            run_calibration(simulate=simulate, motor_id=args.calibrate_motor, all_motors=args.calibrate_all, controller=controller)
        finally:
            try:
                if controller:
                    controller.disable_torque(cfg.MOTOR_IDS)
            except Exception:
                pass
            try:
                if controller:
                    controller.close()
            except Exception:
                pass
    elif args.trayectoria_final:
        run_trayectoria_final(
            simulate=simulate, delay=args.delay, niveles=args.niveles_trayectoria_final,
        )
    else:
        run_sequence.points_per_motor = args.points
        run_sequence(simulate=simulate, delay=args.delay, dynamic_torque=args.dynamic_torque)


def run_calibration(simulate: bool = True, motor_id: int = None, all_motors: bool = False, controller=None):
    if simulate:
        print('Modo simulación: calibración desactivada')
        return

    created_controller = False
    if controller is None:
        try:
            controller = create_controller(simulate=False, port=cfg.SERIAL_PORT, baudrate=cfg.BAUDRATE)
            created_controller = True
        except Exception as e:
            print(f'No se pudo inicializar el controlador: {e}')
            return

    try:
        if motor_id is not None:
            calibrate_single_motor(controller, motor_id)
        elif all_motors:
            calibrate_all_motors_sequential(controller, cfg.MOTOR_IDS)
    finally:
        try:
            controller.disable_torque(cfg.MOTOR_IDS)
        except Exception:
            pass
        try:
            if created_controller:
                controller.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()