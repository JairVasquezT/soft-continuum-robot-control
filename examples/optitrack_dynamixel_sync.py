"""Ejemplo: Sincronización OptiTrack + Dynamixel con 5 puntos de muestreo.

Este script:
1. Se conecta a Motive 2.2.0 vía NatNet a 180Hz
2. Ejecuta movimientos de los 4 motores en los 5 puntos de muestreo por motor
3. Graba en 3 CSVs:
   - dynamixel_*.csv: posiciones de motores (una línea por movimiento, todos los motores)
   - optitrack_*.csv: datos de OptiTrack (una línea por frame)
   - sync_*.csv: datos alineados a 60Hz (OptiTrack + Dynamixel sincronizados)

Uso:
    python -m continuum_robot.examples.optitrack_dynamixel_sync [--no-simulate] [--sync-rate 60]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import time
import argparse
import csv
import threading
from datetime import datetime
from collections import deque

from continuum_robot.config import robot_config as cfg
from continuum_robot.hardware.dynamixel import create_controller
from continuum_robot.hardware.optitrack import create_natnet_client
from continuum_robot.control.trajectories import all_combinations


class DataBuffer:
    """Buffer thread-safe para datos de sensores."""
    def __init__(self, maxlen=100):
        self.buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()
    
    def append(self, data):
        with self._lock:
            self.buffer.append(data)
    
    def get_all(self):
        with self._lock:
            return list(self.buffer)
    
    def clear(self):
        with self._lock:
            self.buffer.clear()


def run_sync_experiment(sampling_rate: int = 180, sync_rate: int = None, simulate: bool = True, output_dir: str = None):
    """Ejecuta el experimento sincronizado.
    
    Args:
        sampling_rate: Frecuencia de OptiTrack en Hz
        sync_rate: Frecuencia de sincronización para CSV combinado (default: cfg.SYNC_SAMPLING_RATE)
        simulate: Si True, no usa hardware real (solo simulación)
        output_dir: Directorio para guardar los logs
    """
    # ========== Capturar timestamp al inicio para garantizar unicidad ==========
    start_time = datetime.now()
    timestamp = start_time.strftime('%Y%m%d_%H%M%S_%f')[:-3]  # %f da microsegundos, se trimea a milisegundos
    
    if sync_rate is None:
        sync_rate = cfg.SYNC_SAMPLING_RATE
    
    if output_dir is None:
        output_dir = 'continuum_robot/data/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    dynamixel_log = os.path.join(output_dir, f'dynamixel_{timestamp}.csv')
    optitrack_log = os.path.join(output_dir, f'optitrack_{timestamp}.csv')
    sync_log = os.path.join(output_dir, f'sync_{timestamp}.csv')
    
    print(f"=" * 80)
    print(f"EXPERIMENTO: OptiTrack (120Hz) + Dynamixel + 5 puntos por motor")
    print(f"=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Modo simulación: {simulate}")
    print(f"OptiTrack muestreo: {sampling_rate}Hz")
    print(f"Sync muestreo: {sync_rate}Hz (para CSV combinado)")
    print(f"Logs:")
    print(f"  - Dynamixel: {dynamixel_log}")
    print(f"  - OptiTrack: {optitrack_log}")
    print(f"  - Sincronizado: {sync_log}")
    print()
    
    # ========== Inicializar Dynamixel ==========
    print("Conectando Dynamixel...")
    try:
        dxl_controller = create_controller(simulate=simulate, port=cfg.SERIAL_PORT, baudrate=cfg.BAUDRATE)
        if not simulate:
            found = dxl_controller.scan(cfg.MOTOR_IDS)
            print(f"  ✓ Motores detectados: {found}")
            dxl_controller.enable_torque(found)
        else:
            print("  (Modo simulación)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return
    
    # ========== Inicializar OptiTrack ==========
    print(f"Conectando OptiTrack/Motive a {cfg.OPTITRACK_HOST}:{cfg.OPTITRACK_PORT} @ {sampling_rate}Hz...")
    optitrack_client = None
    try:
        optitrack_client = create_natnet_client(
            host=cfg.OPTITRACK_HOST, 
            port=cfg.OPTITRACK_PORT, 
            sampling_rate=sampling_rate
        )
        print(f"  ✓ Conectado a OptiTrack")
    except Exception as e:
        print(f"  ⚠ Warning: No se pudo conectar a OptiTrack - {e}")
        print(f"    (continuando solo con Dynamixel)")
    
    # ========== Preparar datos ==========
    combos = list(all_combinations(cfg.MOTOR_IDS))
    print(f"\n✓ Generadas {len(combos)} combinaciones de 5 puntos por motor")
    print(f"  Puntos por motor:")
    for mid in cfg.MOTOR_IDS:
        points = cfg.SAMPLING_POINTS[mid]
        print(f"    Motor {mid}: {points}")
    
    # ========== Buffers de datos ==========
    dxl_buffer = DataBuffer(maxlen=1000)
    optitrack_buffer = DataBuffer(maxlen=5000)
    
    # ========== CSV Dynamixel (una línea por muestreo) ==========
    dxl_file = open(dynamixel_log, 'w', newline='')
    dxl_writer = csv.writer(dxl_file)
    dxl_header = ['timestamp', 'combo_idx', 'combo_total']
    for mid in cfg.MOTOR_IDS:
        dxl_header.extend([f'motor{mid}_goal', f'motor{mid}_actual'])
    dxl_writer.writerow(dxl_header)
    dxl_file.flush()
    
    # ========== CSV OptiTrack (una línea por frame) ==========
    optitrack_file = open(optitrack_log, 'w', newline='')
    optitrack_writer = csv.writer(optitrack_file)
    # El encabezado se escribe cuando se recibe el primer frame
    
    # ========== CSV Sincronizado (alineado a sync_rate Hz) ==========
    sync_file = open(sync_log, 'w', newline='')
    sync_writer = csv.writer(sync_file)
    # El encabezado se escribe cuando se recibe el primer frame
    
    # ========== Loop principal ==========
    print(f"\n{'='*80}")
    print("Iniciando movimientos...")
    print(f"{'='*80}\n")
    
    loop_start_time = time.time()
    last_sync_write = loop_start_time
    sync_interval = 1.0 / sync_rate
    
    optitrack_writer_initialized = False
    sync_writer_initialized = False
    
    try:
        for i, combo in enumerate(combos, 1):
            positions = [combo[mid] for mid in cfg.MOTOR_IDS]
            
            # Mover SIN esperar (wait_for_reached=False)
            print(f"[{i:3d}/{len(combos)}] Moviendo a {positions}...", end='', flush=True)
            try:
                dxl_controller.move(
                    cfg.MOTOR_IDS, 
                    positions, 
                    speed=cfg.DEFAULT_SPEED, 
                    wait_for_reached=False,  # No esperar aqui, capturar datos mientras se mueve
                    timeout=5.0
                )
                print(" iniciado", flush=True)
            except Exception as e:
                print(f" error: {e}", flush=True)
                continue
            
            # ========== Loop de captura mientras el motor se mueve ==========
            move_start = time.time()
            move_timeout = 10.0  # Timeout para el movimiento
            reached = {mid: False for mid in cfg.MOTOR_IDS}
            
            while (time.time() - move_start) < move_timeout:
                now = time.time()
                
                # Leer posiciones actuales de Dynamixel a la frecuencia especificada
                actual_pos = []
                try:
                    for mid in cfg.MOTOR_IDS:
                        actual_pos.append(dxl_controller.get_present_position(mid))
                except Exception:
                    actual_pos = [None] * len(cfg.MOTOR_IDS)
                
                # Verificar si todos los motores llegaron a su posición
                for mid, goal, actual in zip(cfg.MOTOR_IDS, positions, actual_pos):
                    if actual is not None and abs(goal - actual) <= cfg.EPSILON:
                        reached[mid] = True
                
                # Grabar Dynamixel (una línea con todos los motores, a cada sampling)
                dxl_row = [now, i, len(combos)]
                for goal, actual in zip(positions, actual_pos):
                    dxl_row.extend([goal, actual])
                dxl_writer.writerow(dxl_row)
                dxl_file.flush()
                dxl_buffer.append({'timestamp': now, 'positions': actual_pos, 'goals': positions})
                
                # Grabar OptiTrack si hay datos disponibles
                if optitrack_client:
                    all_frames = optitrack_client.get_all_frames()  # Obtener TODOS los frames en el buffer
                    for frame in all_frames:
                        if frame:
                            if not optitrack_writer_initialized:
                                optitrack_header = ['frame_number', 'timestamp']
                                for rb in frame.rigid_bodies:
                                    optitrack_header.extend([
                                        f'RB{rb.id}_X', f'RB{rb.id}_Y', f'RB{rb.id}_Z',
                                        f'RB{rb.id}_Qx', f'RB{rb.id}_Qy', f'RB{rb.id}_Qz', f'RB{rb.id}_Qw'
                                    ])
                                optitrack_writer.writerow(optitrack_header)
                                optitrack_file.flush()
                                optitrack_writer_initialized = True
                            
                            optitrack_row = [frame.frame_number, frame.timestamp]
                            for rb in frame.rigid_bodies:
                                optitrack_row.extend(list(rb.position) + list(rb.rotation))
                            optitrack_writer.writerow(optitrack_row)
                            optitrack_file.flush()
                            optitrack_buffer.append({'frame': frame.frame_number, 'data': optitrack_row})
                    
                    # Limpiar buffer de OptiTrack para evitar duplicados en siguiente iteracion
                    optitrack_client._frame_buffer.clear()
                
                # Escribir CSV sincronizado a la frecuencia especificada
                if now - last_sync_write >= sync_interval:
                    dxl_data = dxl_buffer.get_all()
                    optitrack_data = optitrack_buffer.get_all()
                    
                    if dxl_data or optitrack_data:
                        if not sync_writer_initialized:
                            sync_header = ['timestamp_sync']
                            if dxl_data:
                                for mid in cfg.MOTOR_IDS:
                                    sync_header.extend([f'motor{mid}_goal', f'motor{mid}_actual'])
                            if optitrack_data:
                                frame_sample = optitrack_data[-1]['data']
                                for idx in range(1, len(frame_sample)):
                                    if idx % 7 == 1:
                                        rb_idx = (idx - 1) // 7
                                        sync_header.extend([
                                            f'RB{rb_idx}_X', f'RB{rb_idx}_Y', f'RB{rb_idx}_Z',
                                            f'RB{rb_idx}_Qx', f'RB{rb_idx}_Qy', f'RB{rb_idx}_Qz', f'RB{rb_idx}_Qw'
                                        ])
                                        break
                            sync_writer.writerow(sync_header)
                            sync_file.flush()
                            sync_writer_initialized = True
                        
                        sync_row = [now]
                        if dxl_data:
                            last_dxl = dxl_data[-1]
                            for goal, actual in zip(last_dxl['goals'], last_dxl['positions']):
                                sync_row.extend([goal, actual])
                        if optitrack_data:
                            last_optitrack = optitrack_data[-1]['data']
                            sync_row.extend(last_optitrack[1:])
                        
                        sync_writer.writerow(sync_row)
                        sync_file.flush()
                        last_sync_write = now
                
                # Salir si todos los motores llegaron a su posición
                if all(reached.values()):
                    print(f"  Motores alcanzaron posición objetivo")
                    break
                
                # Esperar antes de siguiente muestreo (según SAMPLING_INTERVAL)
                time.sleep(cfg.SAMPLING_INTERVAL)
            
            if not all(reached.values()):
                print(f"  Warning: timeout esperando motores en posición")
    
    except KeyboardInterrupt:
        print("\ninterrupcion del usuario - deteniendo...")
    
    finally:
        # ========== Cleanup ==========
        print(f"\n{'='*80}")
        print("Finalizando...")
        print(f"{'='*80}\n")
        
        dxl_file.close()
        optitrack_file.close()
        sync_file.close()
        
        print(f"✓ Log Dynamixel:      {dynamixel_log}")
        print(f"✓ Log OptiTrack:      {optitrack_log}")
        print(f"✓ Log Sincronizado:   {sync_log} ({sync_rate}Hz)")
        
        if optitrack_client:
            optitrack_client.disconnect()
        
        try:
            dxl_controller.disable_torque(cfg.MOTOR_IDS)
            dxl_controller.close()
        except Exception:
            pass
        
        print("\n✓ Experimento completado")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sincroniza OptiTrack (Motive 2.2.0) + Dynamixel con 5 puntos de muestreo'
    )
    parser.add_argument(
        '--sampling-rate', 
        type=int, 
        default=180,
        help='Frecuencia de muestreo OptiTrack en Hz (default: 180)'
    )
    parser.add_argument(
        '--sync-rate',
        type=int,
        default=None,
        help='Frecuencia de sincronización para CSV combinado en Hz (default: cfg.SYNC_SAMPLING_RATE)'
    )
    parser.add_argument(
        '--no-simulate',
        action='store_true',
        help='Usar hardware real en lugar de simulación'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directorio para guardar logs (default: continuum_robot/data/)'
    )
    
    args = parser.parse_args()
    
    run_sync_experiment(
        sampling_rate=args.sampling_rate,
        sync_rate=args.sync_rate,
        simulate=not args.no_simulate,
        output_dir=args.output_dir
    )
