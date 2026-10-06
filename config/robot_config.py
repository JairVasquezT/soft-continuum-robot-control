"""Robot configuration: IDs, limits and default parameters."""

# Motor IDs (adjust according to your robot)
MOTOR_IDS = [1, 2, 3, 4]


# Allowed range around HOME for each motor
MOTOR_HOME_RANGES = {
    1: 550,
    2: 550,
    3: 650,
    4: 650,
}

# Home position per motor (calibrated)
HOME_POSITION = {
    1: 1871,
    2: 1951,
    3: 1485,
    4: 1712,
}

# Limits computed from HOME_POSITION using the defined range
LIMITS = {
    mid: (
        max(HOME_POSITION[mid] - MOTOR_HOME_RANGES[mid], 0),
        min(HOME_POSITION[mid] + MOTOR_HOME_RANGES[mid], 4095),
    )
    for mid in MOTOR_IDS
}

# Function to generate 5 points per range (min, 3 intermediate, max)
def generate_5_points(limit_tuple):
    """Generates 5 uniformly distributed points over a range.
    
    Example: (1550, 2500) -> [1550, 1787, 2025, 2262, 2500]
    """
    min_val, max_val = limit_tuple
    step = (max_val - min_val) / 4
    return [int(min_val + step * i) for i in range(5)]

# Sampling points for OptiTrack (5 points per motor)
SAMPLING_POINTS = {
    motor_id: generate_5_points(LIMITS[motor_id])
    for motor_id in [1, 2, 3, 4]
}

# Angle (degrees, XZ plane) of each motor's pulling direction --
# used by trajectories.py to estimate the net Cartesian push of a
# combination of levels and discard combinations that cancel each other out
# (e.g. two opposing motors pulling with opposite levels).
MOTOR_ANGULOS_TRACCION = {1: 0, 2: 90, 3: 30, 4: 120}

# WIDER range than MOTOR_HOME_RANGES/LIMITS, used ONLY as the clip
# limit for the local micro-exploration around an already reached
# main point (candidatos_exploracion_local in trajectories.py)
# -- the grid of main points itself is still generated over LIMITS,
# unchanged. It matches RANGOS_MANUALES already used by the ML pipeline
# (dataset_pred_filt.py), so as not to clip to zero effect an exploration that
# falls near the edge of the original grid.
MOTOR_EXPLORATION_RANGES = {1: 650, 2: 650, 3: 750, 4: 750}
LIMITS_EXPLORACION = {
    mid: (
        max(HOME_POSITION[mid] - MOTOR_EXPLORATION_RANGES[mid], 0),
        min(HOME_POSITION[mid] + MOTOR_EXPLORATION_RANGES[mid], 4095),
    )
    for mid in MOTOR_IDS
}


# Default speed
DEFAULT_SPEED = 30

# Dynamixel SDK / serial port (adjust to your system)
SERIAL_PORT = 'COM3'
BAUDRATE = 2000000

# Register addresses for Dynamixel MX (protocol 1.0)
ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_MOVING_SPEED = 32
ADDR_MX_PRESENT_POSITION = 36
ADDR_MX_PRESENT_LOAD = 40

# Compatibility: generic names used by the code
ADDR_GOAL_POSITION = ADDR_MX_GOAL_POSITION
ADDR_MOVING_SPEED = ADDR_MX_MOVING_SPEED
ADDR_PRESENT_POSITION = ADDR_MX_PRESENT_POSITION
ADDR_PRESENT_LOAD = ADDR_MX_PRESENT_LOAD

PROTOCOL_VERSION = 1.0

# Additional constants suggested by the user
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_MAXIMUM_POSITION_VALUE = 4095
DXL_MINIMUM_POSITION_VALUE = 0
MIDDLE_POSITION = DXL_MAXIMUM_POSITION_VALUE // 2
EPSILON = 10
MAX_SPEED = 40
SAMPLING_INTERVAL = 0.01666

# ============ OPTITRACK / NATNET ============
# Configuration for Motive 2.2.0 (Same PC via Loopback)
OPTITRACK_HOST = '192.168.1.178'   # 🎯 CHANGED: '127.0.0.1' is the exact IP for "loopback"
OPTITRACK_PORT = 1510          # Default NatNet data port (Motive uses 1510)
OPTITRACK_COMMAND_PORT = 1511  # Default NatNet command port
OPTITRACK_SAMPLING_RATE = 60   # 🎯 Adjusted to 60Hz (the sampling rate you want)
USE_MULTICAST = False          # 🎯 We force Unicast by default in your configuration

# Synchronization frequency for recording aligned data
SYNC_SAMPLING_RATE = 60

