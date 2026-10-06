#!/usr/bin/env python3
"""Script to send all motors to the home position.

Usage:
    python continuum_robot/go_home.py
    or
    python -m continuum_robot.go_home
    or from inside continuum_robot/:
    python go_home.py
"""

import sys
import time
import os

# Add the parent directory to the path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from continuum_robot.config import robot_config as cfg
from continuum_robot.hardware.dynamixel import create_controller


def go_home():
    """Sends all motors to their calibrated home position."""
    print("Inicializando controlador Dynamixel...")
    try:
        controller = create_controller(simulate=False)
    except Exception as e:
        print(f"❌ Error: No se pudo inicializar el controlador hardware: {e}")
        return False
    
    try:
        print(f"Activando torque en todos los motores...")
        controller.enable_torque(cfg.MOTOR_IDS)
        time.sleep(0.2)
        
        # Build list of home positions per motor
        home_positions = [cfg.HOME_POSITION[mid] for mid in cfg.MOTOR_IDS]
        

        # Build intermediate position: 200 ticks CW from home (respecting limits)
        pre_home_positions = []
        for mid, home_pos in zip(cfg.MOTOR_IDS, home_positions):
            max_limit = cfg.LIMITS[mid][1]
            pre_pos = min(home_pos + 200, max_limit)
            pre_home_positions.append(pre_pos)
        
        print(f"Moviendo motores primero a posición PRE-HOME (200 ticks CW desde HOME):")
        for mid, pos in zip(cfg.MOTOR_IDS, pre_home_positions):
            print(f"  Motor {mid} → {pos}")
        controller.move(cfg.MOTOR_IDS, pre_home_positions, speed=cfg.DEFAULT_SPEED,
                       wait_for_reached=True, timeout=10.0)
        
        # Build adjusted command position: HOME - 5 ticks
        command_home_positions = []
        for mid, home_pos in zip(cfg.MOTOR_IDS, home_positions):
            min_limit = cfg.LIMITS[mid][0]
            command_pos = max(home_pos - 6, min_limit)
            command_home_positions.append(command_pos)

        print(f"Moviendo motores a posición HOME ajustada (-5 ticks):")
        for mid, pos in zip(cfg.MOTOR_IDS, command_home_positions):
            print(f"  Motor {mid} → {pos}")
        controller.move(cfg.MOTOR_IDS, command_home_positions, speed=cfg.DEFAULT_SPEED,
                       wait_for_reached=True, timeout=10.0)
        
        print(f"✓ Todos los motores pasaron por PRE-HOME y recibieron el comando HOME-5")
        time.sleep(0.3)
        
        # Read final positions and show report
        print(f"\n{'='*50}")
        print(f"REPORTE DE POSICIONES FINALES")
        print(f"{'='*50}")
        
        all_ok = True
        for mid, target_pos in zip(cfg.MOTOR_IDS, home_positions):
            try:
                actual_pos = controller.get_present_position(mid)
                error = abs(actual_pos - target_pos)
                status = "✓" if error <= 5 else "⚠"
                print(f"{status} Motor {mid}: Objetivo={target_pos}, Actual={actual_pos}, Error={error}")
                if error > 5:
                    all_ok = False
            except Exception as e:
                print(f"❌ Motor {mid}: Error leyendo posición - {e}")
                all_ok = False
        
        print(f"{'='*50}")
        
        if not all_ok:
            print(f"⚠ Algunos motores tienen desviación > 5 unidades")
        
        # Disable torque to reduce heat/noise
        print("Desactivando torque...")
        controller.disable_torque(cfg.MOTOR_IDS)
        
        return all_ok
        
    except Exception as e:
        print(f"❌ Error durante movimiento: {e}")
        try:
            controller.disable_torque(cfg.MOTOR_IDS)
        except:
            pass
        return False


if __name__ == "__main__":
    success = go_home()
    sys.exit(0 if success else 1)
