"""Calibración secuencial de motores Dynamixel por búsqueda de punto medio basada en carga.

El Dynamixel EX-106+ reporta Present Load con signo: negativo para CW (horario), positivo para CCW (antihorario).
Se usa abs(present_load) >= 102 como umbral de detección (10% de carga).

Proceso:
1. Para cada motor (1, 2, 3, 4):
   - Desactiva torque en todos los demás motores
   - Activa torque en el motor actual
   - Ejecuta búsqueda de punto medio (giro horario/antihorario hasta 10% carga)
   - Guarda el punto medio como home_position
2. Activa torque en todos los motores
3. Mueve cada motor a su posición home calculada
"""

import time
from typing import Dict, List

from continuum_robot.config import robot_config as cfg


def rotate_until_load(controller, motor_id: int, direction: int = 1, 
                      max_speed: int = 20, timeout: float = 8.0, poll_interval: float = 0.02):
    """Gira el motor hasta que el cable de la polea alcance la tensión límite (~6%-7% carga)."""
    LOAD_THRESHOLD = 65  # Detecta el fin de la holgura sin deformar en exceso la estructura
    
    goal_pos = cfg.DXL_MAXIMUM_POSITION_VALUE if direction > 0 else cfg.DXL_MINIMUM_POSITION_VALUE
    
    controller.set_moving_speed(motor_id, max_speed)
    controller.set_goal_position(motor_id, goal_pos)
    
    start = time.time()
    while (time.time() - start) < timeout:
        try:
            raw_load, magnitude, direction_name, signed_load = controller.get_present_load_decoded(motor_id)
            pos = controller.get_present_position(motor_id)
            
            if magnitude >= LOAD_THRESHOLD:
                # Detener inmediatamente para congelar la posición del límite de tensión
                controller.set_goal_position(motor_id, pos)
                print(f'  ✓ Motor {motor_id}: Tensión en cable detectada a pos {pos} (mag={magnitude})')
                return pos
        except Exception as e:
            print(f'  Error leyendo motor {motor_id}: {e}')
            return None
        
        time.sleep(poll_interval)
    
    return None



def calibrate_single_motor(controller, motor_id: int) -> int:
    """Busca el punto medio entre la tensión del Cable A y del Cable B de un mismo motor."""
    print(f'\n--- Calibrando Motor {motor_id} ---')
    controller.enable_torque([motor_id])
    time.sleep(0.1)
    
    # 1. Tensión Cable A (Sentido Horario)
    pos_cw = rotate_until_load(controller, motor_id, direction=1, max_speed=20)
    time.sleep(0.2)
    
    # 2. Tensión Cable B (Sentido Antihorario)
    pos_ccw = rotate_until_load(controller, motor_id, direction=-1, max_speed=20)
    time.sleep(0.2)
    
    if pos_cw is None or pos_ccw is None:
        print(f'❌ ERROR en motor {motor_id}. Usando valor por defecto.')
        return cfg.HOME_POSITION[motor_id]
    
    # El centro exacto entre ambos cables
    home_calc = int((pos_cw + pos_ccw) // 2)
    print(f'  Resultado Motor {motor_id}: Límite CW={pos_cw}, Límite CCW={pos_ccw} -> HOME = {home_calc}')
    return home_calc


def calibrate_all_motors_sequential(controller, motor_ids: List[int] = None) -> Dict[int, int]:
    """Calibración en Cascada por Niveles para evitar acoplamiento de fricción."""
    home_positions = {}
    
    print('\n' + '='*60)
    print('INICIANDO CALIBRACIÓN POR NIVELES (CASCADA)')
    print('='*60)
    
    # Define la estructura de tu robot por niveles
    LEVEL_1_MOTORS = [1, 2]
    LEVEL_2_MOTORS = [3, 4]
    
    # -----------------------------------------------------------------
    # ETAPA 1: Calibrar Nivel 1 (Base)
    # -----------------------------------------------------------------
    print('\n>>> ETAPA 1: Calibrando Motores del Nivel 1 (Base)...')
    for mid in LEVEL_1_MOTORS:
        home_positions[mid] = calibrate_single_motor(controller, mid)
    
    # ⚠️ PASO CLAVE: Mover Nivel 1 a su HOME para enderezar la estructura
    print('\n[Alineando Nivel 1 a Home Neutro antes de continuar...]')
    for mid in LEVEL_1_MOTORS:
        controller.move([mid], [home_positions[mid]], speed=cfg.DEFAULT_SPEED, wait_for_reached=True)
    time.sleep(0.5)

    # -----------------------------------------------------------------
    # ETAPA 2: Calibrar Nivel 2 con la base recta
    # -----------------------------------------------------------------
    print('\n>>> ETAPA 2: Calibrando Motores del Nivel 2 (Efector superior)...')
    for mid in LEVEL_2_MOTORS:
        home_positions[mid] = calibrate_single_motor(controller, mid)
    
    # -----------------------------------------------------------------
    # ETAPA FINAL: Mover todo el robot a su posición Home global
    # -----------------------------------------------------------------
    print('\n' + '='*60)
    print(f'CALIBRACIÓN COMPLETA. Posiciones calculadas: {home_positions}')
    print('Moviendo todo el robot a Home definitivo...')
    
    all_mids = LEVEL_1_MOTORS + LEVEL_2_MOTORS
    final_targets = [home_positions[mid] for mid in all_mids]
    
    controller.move(all_mids, final_targets, speed=cfg.DEFAULT_SPEED, wait_for_reached=True)
    
    return home_positions