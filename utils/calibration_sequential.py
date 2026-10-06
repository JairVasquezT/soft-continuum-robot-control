"""Sequential calibration of Dynamixel motors by load-based midpoint search.

The Dynamixel EX-106+ reports Present Load with sign: negative for CW (clockwise), positive for CCW (counterclockwise).
abs(present_load) >= 102 is used as the detection threshold (10% load).

Process:
1. For each motor (1, 2, 3, 4):
   - Disable torque on all other motors
   - Enable torque on the current motor
   - Run midpoint search (clockwise/counterclockwise rotation up to 10% load)
   - Save the midpoint as home_position
2. Enable torque on all motors
3. Move each motor to its computed home position
"""

import time
from typing import Dict, List

from continuum_robot.config import robot_config as cfg


def rotate_until_load(controller, motor_id: int, direction: int = 1, 
                      max_speed: int = 20, timeout: float = 8.0, poll_interval: float = 0.02):
    """Rotates the motor until the pulley cable reaches the limit tension (~6%-7% load)."""
    LOAD_THRESHOLD = 65  # Detects the end of the slack without over-deforming the structure
    
    goal_pos = cfg.DXL_MAXIMUM_POSITION_VALUE if direction > 0 else cfg.DXL_MINIMUM_POSITION_VALUE
    
    controller.set_moving_speed(motor_id, max_speed)
    controller.set_goal_position(motor_id, goal_pos)
    
    start = time.time()
    while (time.time() - start) < timeout:
        try:
            raw_load, magnitude, direction_name, signed_load = controller.get_present_load_decoded(motor_id)
            pos = controller.get_present_position(motor_id)
            
            if magnitude >= LOAD_THRESHOLD:
                # Stop immediately to freeze the position of the tension limit
                controller.set_goal_position(motor_id, pos)
                print(f'  ✓ Motor {motor_id}: Tensión en cable detectada a pos {pos} (mag={magnitude})')
                return pos
        except Exception as e:
            print(f'  Error leyendo motor {motor_id}: {e}')
            return None
        
        time.sleep(poll_interval)
    
    return None



def calibrate_single_motor(controller, motor_id: int) -> int:
    """Finds the midpoint between Cable A's and Cable B's tension for a single motor."""
    print(f'\n--- Calibrando Motor {motor_id} ---')
    controller.enable_torque([motor_id])
    time.sleep(0.1)
    
    # 1. Cable A Tension (Clockwise)
    pos_cw = rotate_until_load(controller, motor_id, direction=1, max_speed=20)
    time.sleep(0.2)
    
    # 2. Cable B Tension (Counterclockwise)
    pos_ccw = rotate_until_load(controller, motor_id, direction=-1, max_speed=20)
    time.sleep(0.2)
    
    if pos_cw is None or pos_ccw is None:
        print(f'❌ ERROR en motor {motor_id}. Usando valor por defecto.')
        return cfg.HOME_POSITION[motor_id]
    
    # The exact center between both cables
    home_calc = int((pos_cw + pos_ccw) // 2)
    print(f'  Resultado Motor {motor_id}: Límite CW={pos_cw}, Límite CCW={pos_ccw} -> HOME = {home_calc}')
    return home_calc


def calibrate_all_motors_sequential(controller, motor_ids: List[int] = None) -> Dict[int, int]:
    """Cascaded calibration by Levels to avoid friction coupling."""
    home_positions = {}
    
    print('\n' + '='*60)
    print('INICIANDO CALIBRACIÓN POR NIVELES (CASCADA)')
    print('='*60)
    
    # Define your robot's structure by levels
    LEVEL_1_MOTORS = [1, 2]
    LEVEL_2_MOTORS = [3, 4]
    
    # -----------------------------------------------------------------
    # STAGE 1: Calibrate Level 1 (Base)
    # -----------------------------------------------------------------
    print('\n>>> ETAPA 1: Calibrando Motores del Nivel 1 (Base)...')
    for mid in LEVEL_1_MOTORS:
        home_positions[mid] = calibrate_single_motor(controller, mid)
    
    # ⚠️ KEY STEP: Move Level 1 to its HOME to straighten the structure
    print('\n[Alineando Nivel 1 a Home Neutro antes de continuar...]')
    for mid in LEVEL_1_MOTORS:
        controller.move([mid], [home_positions[mid]], speed=cfg.DEFAULT_SPEED, wait_for_reached=True)
    time.sleep(0.5)

    # -----------------------------------------------------------------
    # STAGE 2: Calibrate Level 2 with the straight base
    # -----------------------------------------------------------------
    print('\n>>> ETAPA 2: Calibrando Motores del Nivel 2 (Efector superior)...')
    for mid in LEVEL_2_MOTORS:
        home_positions[mid] = calibrate_single_motor(controller, mid)
    
    # -----------------------------------------------------------------
    # FINAL STAGE: Move the whole robot to its global home position
    # -----------------------------------------------------------------
    print('\n' + '='*60)
    print(f'CALIBRACIÓN COMPLETA. Posiciones calculadas: {home_positions}')
    print('Moviendo todo el robot a Home definitivo...')
    
    all_mids = LEVEL_1_MOTORS + LEVEL_2_MOTORS
    final_targets = [home_positions[mid] for mid in all_mids]
    
    controller.move(all_mids, final_targets, speed=cfg.DEFAULT_SPEED, wait_for_reached=True)
    
    return home_positions