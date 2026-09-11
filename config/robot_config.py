"""Configuración del robot: IDs, límites y parámetros por defecto."""

# IDs de los motores (ajusta según tu robot)
MOTOR_IDS = [1, 2, 3, 4]


# Rango permitido alrededor de HOME para cada motor
MOTOR_HOME_RANGES = {
    1: 550,
    2: 550,
    3: 650,
    4: 650,
}

# Posición home por motor (calibrada)
HOME_POSITION = {
    1: 1871,
    2: 1951,
    3: 1485,
    4: 1712,
}

# Límites calculados desde HOME_POSITION usando el rango definido
LIMITS = {
    mid: (
        max(HOME_POSITION[mid] - MOTOR_HOME_RANGES[mid], 0),
        min(HOME_POSITION[mid] + MOTOR_HOME_RANGES[mid], 4095),
    )
    for mid in MOTOR_IDS
}

# Función para generar 5 puntos por rango (mín, 3 intermedios, máx)
def generate_5_points(limit_tuple):
    """Genera 5 puntos uniformemente distribuidos en un rango.
    
    Ejemplo: (1550, 2500) -> [1550, 1787, 2025, 2262, 2500]
    """
    min_val, max_val = limit_tuple
    step = (max_val - min_val) / 4
    return [int(min_val + step * i) for i in range(5)]

# Puntos de muestreo para OptiTrack (5 puntos por motor)
SAMPLING_POINTS = {
    motor_id: generate_5_points(LIMITS[motor_id])
    for motor_id in [1, 2, 3, 4]
}

# Ángulo (grados, plano XZ) de la dirección de tracción de cada motor --
# usado por trajectories.py para estimar el empuje cartesiano neto de una
# combinación de niveles y descartar combinaciones que se cancelan entre sí
# (p.ej. dos motores enfrentados tirando con niveles opuestos).
MOTOR_ANGULOS_TRACCION = {1: 0, 2: 90, 3: 30, 4: 120}

# Rango MÁS ANCHO que MOTOR_HOME_RANGES/LIMITS, usado SOLO como límite de
# recorte (clip) para la micro-exploración local alrededor de un punto
# principal ya alcanzado (candidatos_exploracion_local en trajectories.py)
# -- la grilla de puntos principales en sí sigue generándose sobre LIMITS,
# sin cambios. Coincide con RANGOS_MANUALES que ya usa el pipeline de ML
# (dataset_pred_filt.py), para no clipear a 0 efecto una exploración que
# cae cerca del borde de la grilla original.
MOTOR_EXPLORATION_RANGES = {1: 650, 2: 650, 3: 750, 4: 750}
LIMITS_EXPLORACION = {
    mid: (
        max(HOME_POSITION[mid] - MOTOR_EXPLORATION_RANGES[mid], 0),
        min(HOME_POSITION[mid] + MOTOR_EXPLORATION_RANGES[mid], 4095),
    )
    for mid in MOTOR_IDS
}


# Velocidad por defecto
DEFAULT_SPEED = 30

# Dynamixel SDK / puerto serial (ajusta a tu sistema)
SERIAL_PORT = 'COM3'
BAUDRATE = 2000000

# Direcciones de registro para Dynamixel MX (protocolo 1.0)
ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_MOVING_SPEED = 32
ADDR_MX_PRESENT_POSITION = 36
ADDR_MX_PRESENT_LOAD = 40

# Compatibilidad: nombres genéricos usados por el código
ADDR_GOAL_POSITION = ADDR_MX_GOAL_POSITION
ADDR_MOVING_SPEED = ADDR_MX_MOVING_SPEED
ADDR_PRESENT_POSITION = ADDR_MX_PRESENT_POSITION
ADDR_PRESENT_LOAD = ADDR_MX_PRESENT_LOAD

PROTOCOL_VERSION = 1.0

# Constantes adicionales sugeridas por usuario
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_MAXIMUM_POSITION_VALUE = 4095
DXL_MINIMUM_POSITION_VALUE = 0
MIDDLE_POSITION = DXL_MAXIMUM_POSITION_VALUE // 2
EPSILON = 10
MAX_SPEED = 40
SAMPLING_INTERVAL = 0.01666

# ============ OPTITRACK / NATNET ============
# Configuración para Motive 2.2.0 (Misma PC vía Loopback)
OPTITRACK_HOST = '192.168.1.178'   # 🎯 CAMBIADO: '127.0.0.1' es la IP exacta para "loopback"
OPTITRACK_PORT = 1510          # Puerto NatNet de datos por defecto (Motive usa 1510)
OPTITRACK_COMMAND_PORT = 1511  # Puerto NatNet de comandos por defecto
OPTITRACK_SAMPLING_RATE = 60   # 🎯 Ajustado a 60Hz (la frecuencia de muestreo que quieres)
USE_MULTICAST = False          # 🎯 Forzamos Unicast por defecto en tu configuración

# Frecuencia de sincronización para grabar datos alineados
SYNC_SAMPLING_RATE = 60

