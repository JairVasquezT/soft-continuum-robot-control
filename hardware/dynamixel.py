"""Simple control for Dynamixel using `dynamixel_sdk` if installed.
Includes a `Dummy` for testing without hardware.
"""
from time import sleep
import time
from typing import Iterable, Sequence, List

from continuum_robot.config import robot_config as cfg

try:
    from dynamixel_sdk import *
    HAS_SDK = True
except Exception:
    HAS_SDK = False


class DynamixelController:
    def __init__(self, port: str = None, baudrate: int = None):
        port = port or cfg.SERIAL_PORT
        baudrate = baudrate or cfg.BAUDRATE
        if not HAS_SDK:
            raise RuntimeError('dynamixel_sdk no disponible')
        self.portHandler = PortHandler(port)
        self.packetHandler = PacketHandler(int(cfg.PROTOCOL_VERSION))
        self.protocol = int(cfg.PROTOCOL_VERSION)
        try:
            if not self.portHandler.openPort():
                raise RuntimeError(f'No se pudo abrir puerto {port}')
        except Exception as e:
            raise RuntimeError(
                f'Error al abrir puerto {port}: {e}. ' \
                'Verifica que el puerto exista, no esté ocupado y que tengas permisos.')
        if not self.portHandler.setBaudRate(baudrate):
            raise RuntimeError(f'No se pudo configurar baudrate {baudrate}')
        
        # Hold Filter: Memory of the last valid position per motor
        self.ultimas_posiciones_validas = {mid: 0 for mid in cfg.MOTOR_IDS}
        
        if HAS_SDK:
            # To send multiple target positions at once (GroupSyncWrite does work in Proto 1.0)
            self.groupSyncWritePos = GroupSyncWrite(
                self.portHandler, 
                self.packetHandler, 
                cfg.ADDR_GOAL_POSITION, 
                4 if self.protocol == 2 else 2
            )

    def scan(self, ids: Sequence[int] = None) -> List[int]:
        """Tries to ping the ids and returns the list of those that respond."""
        ids = list(ids or cfg.MOTOR_IDS)
        found = []
        for mid in ids:
            try:
                # ping returns (model_number, result, error) in the SDK
                model_num, comm_result, error = self.packetHandler.ping(self.portHandler, int(mid))
                if comm_result == 0:
                    found.append(mid)
            except Exception:
                # ignores motors that do not respond
                continue
        return found

    def set_moving_speed(self, motor_id: int, speed: int):
        # Use 2 or 4 bytes depending on the protocol
        if self.protocol == 2:
            self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, cfg.ADDR_MOVING_SPEED, int(speed))
        else:
            self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, cfg.ADDR_MOVING_SPEED, int(speed))

    def set_goal_position(self, motor_id: int, position: int):
        if self.protocol == 2:
            self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, cfg.ADDR_GOAL_POSITION, int(position))
        else:
            self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, cfg.ADDR_GOAL_POSITION, int(position))

    def get_present_position(self, motor_id: int) -> int:
        if self.protocol == 2:
            pos, comm, err = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, cfg.ADDR_PRESENT_POSITION)
        else:
            pos, comm, err = self.packetHandler.read2ByteTxRx(self.portHandler, motor_id, cfg.ADDR_PRESENT_POSITION)
        return int(pos)

    def get_present_load(self, motor_id: int) -> int:
        """Reads the raw value of the Present Load register."""
        if self.protocol == 2:
            raw_load, comm, err = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, cfg.ADDR_PRESENT_LOAD)
        else:
            raw_load, comm, err = self.packetHandler.read2ByteTxRx(self.portHandler, motor_id, cfg.ADDR_PRESENT_LOAD)
        return int(raw_load)

    def decode_load(self, raw_load: int):
        """Decodes the Present Load register of the EX-106+.

        - bits 0-9: magnitude (0-1023)
        - bit 10: direction (0 = CCW, 1 = CW)
        """
        magnitude = raw_load & 0x3FF
        direction_bit = (raw_load >> 10) & 0x1
        direction = 'CW' if direction_bit else 'CCW'
        signed_load = -magnitude if direction == 'CW' else magnitude
        return raw_load, magnitude, direction, signed_load

    def get_present_load_decoded(self, motor_id: int):
        raw_load = self.get_present_load(motor_id)
        return self.decode_load(raw_load)

    def enable_torque(self, ids: Sequence[int]):
        for mid in ids:
            try:
                res = self.packetHandler.write1ByteTxRx(self.portHandler, int(mid), cfg.ADDR_MX_TORQUE_ENABLE, int(cfg.TORQUE_ENABLE))
            except Exception as e:
                print(f'Error enabling torque motor {mid}: {e}')

    def disable_torque(self, ids: Sequence[int]):
        for mid in ids:
            try:
                res = self.packetHandler.write1ByteTxRx(self.portHandler, int(mid), cfg.ADDR_MX_TORQUE_ENABLE, int(cfg.TORQUE_DISABLE))
            except Exception as e:
                print(f'Error disabling torque motor {mid}: {e}')

    def move(self, ids: Sequence[int], positions: Sequence[int], speed: int = None, wait_for_reached: bool = True, timeout: float = 5.0):
        speed = speed if speed is not None else cfg.DEFAULT_SPEED
        
        # Keep sending the speed (or pass it to SyncWrite too if your registers allow it)
        for mid in ids:
            self.set_moving_speed(mid, speed)
            
        # --- OPTIMIZATION WITH SYNC WRITE ---
        self.groupSyncWritePos.clearParam()
        is_p2 = (self.protocol == 2)
        
        for mid, pos in zip(ids, positions):
            # Convert the position into bytes readable by the SDK
            if is_p2:
                param_goal_pos = [DXL_LOBYTE(DXL_LOWORD(int(pos))), DXL_HIBYTE(DXL_LOWORD(int(pos))), 
                                  DXL_LOBYTE(DXL_HIWORD(int(pos))), DXL_HIBYTE(DXL_HIWORD(int(pos)))]
            else:
                param_goal_pos = [DXL_LOBYTE(int(pos)), DXL_HIBYTE(int(pos))]
                
            self.groupSyncWritePos.addParam(int(mid), param_goal_pos)
            
        # Transmit to all motors simultaneously
        self.groupSyncWritePos.txPacket()
        # -----------------------------------

        if not wait_for_reached:
            return

        # Wait-loop optimization using our new synchronous function
        # --- Wait loop until the target is reached ---
        start = time.time()
        pending = set(int(m) for m in ids)
        targets = {int(m): int(p) for m, p in zip(ids, positions)}

        while pending and (time.time() - start) < timeout:
            id_list = list(pending)
            current_positions, _ = self.sync_get_present_positions(id_list)
            
            for mid, now in zip(id_list, current_positions):
                # Validate that 'now' is a valid numeric value
                if now is not None and isinstance(now, (int, float)):
                    if abs(now - targets[mid]) <= getattr(cfg, 'EPSILON', 2):
                        pending.remove(mid)
                        
            sleep(getattr(cfg, 'SAMPLING_INTERVAL', 0.01))

        if pending:
            print(f"Warning: motores {sorted(list(pending))} no alcanzaron la posición objetivo en {timeout}s")

    def close(self):
        try:
            self.portHandler.closePort()
        except Exception:
            pass

    def sync_get_present_positions(self, ids: Sequence[int]):
        """Reads positions sequentially, compatible with Protocol 1.0 (EX-106+).
        
        Returns: tuple (positions: List[int], valid_reading: int)
            - positions: list of positions (backed up if a reading fails)
            - valid_reading: 1 if all readings were clean, 0 if the backup was used
        """
        if not ids:
            return [], 1

        posiciones = []
        lectura_valida = 1
        is_p2 = (self.protocol == 2)
        
        # Sequential reading compatible with Protocol 1.0
        for mid in ids:
            try:
                if is_p2:
                    pos, comm, err = self.packetHandler.read4ByteTxRx(
                        self.portHandler, int(mid), cfg.ADDR_PRESENT_POSITION)
                else:
                    pos, comm, err = self.packetHandler.read2ByteTxRx(
                        self.portHandler, int(mid), cfg.ADDR_PRESENT_POSITION)
                
                if comm == 0 and err == 0:
                    pos_int = int(pos)
                    posiciones.append(pos_int)
                    self.ultimas_posiciones_validas[int(mid)] = pos_int
                else:
                    # Communication failure: use backup
                    posiciones.append(self.ultimas_posiciones_validas[int(mid)])
                    lectura_valida = 0
            except Exception:
                # Internal exception: use backup
                posiciones.append(self.ultimas_posiciones_validas[int(mid)])
                lectura_valida = 0
        
        return posiciones, lectura_valida

    def sync_get_present_position_and_load(self, ids: Sequence[int]):
        """Reads position and torque/load sequentially, compatible with Protocol 1.0 (EX-106+).
        
        Returns: (positions: List[int], torques: List[int], valid_reading: int)
        """
        if not ids:
            return [], [], 1

        posiciones = []
        torques = []
        lectura_valida = 1
        is_p2 = (self.protocol == 2)

        for mid in ids:
            try:
                if is_p2:
                    pos, comm1, err1 = self.packetHandler.read4ByteTxRx(self.portHandler, int(mid), cfg.ADDR_PRESENT_POSITION)
                    load, comm2, err2 = self.packetHandler.read4ByteTxRx(self.portHandler, int(mid), cfg.ADDR_PRESENT_LOAD)
                else:
                    pos, comm1, err1 = self.packetHandler.read2ByteTxRx(self.portHandler, int(mid), cfg.ADDR_PRESENT_POSITION)
                    load, comm2, err2 = self.packetHandler.read2ByteTxRx(self.portHandler, int(mid), cfg.ADDR_PRESENT_LOAD)

                if comm1 == 0 and err1 == 0 and comm2 == 0 and err2 == 0:
                    pos_int = int(pos)
                    _, _, _, signed_load = self.decode_load(int(load))
                    posiciones.append(pos_int)
                    torques.append(signed_load)
                    self.ultimas_posiciones_validas[int(mid)] = pos_int
                else:
                    posiciones.append(self.ultimas_posiciones_validas[int(mid)])
                    torques.append(0)
                    lectura_valida = 0
            except Exception:
                posiciones.append(self.ultimas_posiciones_validas[int(mid)])
                torques.append(0)
                lectura_valida = 0

        return posiciones, torques, lectura_valida



class DummyDynamixel:
    """Dummy controller that prints the actions (useful for testing)."""
    def __init__(self, *args, **kwargs):
        print('DummyDynamixel: modo simulación (sin hardware)')

    def set_moving_speed(self, motor_id: int, speed: int):
        print(f'[Dummy] set speed motor {motor_id} -> {speed}')

    def set_goal_position(self, motor_id: int, position: int):
        print(f'[Dummy] set goal motor {motor_id} -> {position}')

    def get_present_load(self, motor_id: int) -> int:
        raw_load = 0
        print(f'[Dummy] get_present_load motor {motor_id} -> {raw_load}')
        return raw_load

    def decode_load(self, raw_load: int):
        magnitude = raw_load & 0x3FF
        direction_bit = (raw_load >> 10) & 0x1
        direction = 'CW' if direction_bit else 'CCW'
        signed_load = -magnitude if direction == 'CW' else magnitude
        return raw_load, magnitude, direction, signed_load

    def get_present_load_decoded(self, motor_id: int):
        raw_load = self.get_present_load(motor_id)
        return self.decode_load(raw_load)

    def move(self, ids: Sequence[int], positions: Sequence[int], speed: int = None, wait_for_reached: bool = True, timeout: float = 5.0):
        print(f'[Dummy] move ids={ids} positions={positions} speed={speed} (simulated)')
        # Simulate movement time proportional to the difference
        try:
            avg_diff = sum(abs(int(p) - cfg.MIDDLE_POSITION) for p in positions) / max(1, len(positions))
            sim_time = min(0.5 + avg_diff / 2000.0, 2.0)
        except Exception:
            sim_time = 0.2
        sleep(sim_time)

    def close(self):
        print('[Dummy] close')

    def enable_torque(self, ids: Sequence[int]):
        print(f'[Dummy] enable_torque {list(ids)}')

    def disable_torque(self, ids: Sequence[int]):
        print(f'[Dummy] disable_torque {list(ids)}')


def create_controller(simulate: bool = False, port: str = None, baudrate: int = None):
    if simulate:
        return DummyDynamixel()
    if not HAS_SDK:
        raise RuntimeError(
            'dynamixel_sdk no está instalado en el intérprete de Python activo. '
            'Instálalo con "pip install dynamixel_sdk" en ese entorno o ejecuta con --simulate.'
        )
    return DynamixelController(port=port, baudrate=baudrate)





