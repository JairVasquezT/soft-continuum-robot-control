from dynamixel_sdk import *
import time

print("SDK instalado correctamente")

# ======================
# Configuration
# ======================

# PORT = "/dev/ttyUSB0"      # Linux
PORT = "COM3"            # Windows

BAUDRATE =2000000         # Check that it matches your motors' baudrate

PROTOCOL_VERSION = 1.0

# EX-106+ addresses
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_PRESENT_POSITION = 36
ADDR_MOVING_SPEED = 32

TORQUE_ENABLE = 1

IDS = [1,2,3,4]

portHandler = PortHandler(PORT)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    quit()

if not portHandler.setBaudRate(BAUDRATE):
    quit()

# Enable torque
for motor in IDS:
    packetHandler.write1ByteTxRx(
        portHandler,
        motor,
        ADDR_TORQUE_ENABLE,
        TORQUE_ENABLE
    )

print("Motores listos")

LIMITES = {
    1:(1550,2500),
    2:(1550,2500),
    3:(1400,2650),
    4:(1400,2650)
}

def mover_motor(ID, posicion):

    minimo, maximo = LIMITES[ID]

    posicion = max(minimo, min(maximo, posicion))

    packetHandler.write2ByteTxRx(
        portHandler,
        ID,
        ADDR_GOAL_POSITION,
        posicion
    )

def leer_posicion(ID):

    posicion, _, _ = packetHandler.read2ByteTxRx(
        portHandler,
        ID,
        ADDR_PRESENT_POSITION
    )

    return posicion

POS_NEUTRA = {
    1:2025,
    2:2025,
    3:2025,
    4:2025
}

def set_speed(ID, speed):

    packetHandler.write2ByteTxRx(
        portHandler,
        ID,
        ADDR_MOVING_SPEED,
        speed
    )