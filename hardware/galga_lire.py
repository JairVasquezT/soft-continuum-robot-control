import time
from Phidget22.Devices.VoltageRatioInput import *
from Phidget22.PhidgetException import *

# =====================================================================
# LOAD CELL (STRAIN GAUGE) CONFIGURATION
# =====================================================================
# The PhidgetBridge 4-Input has 4 channels (0 to 3)
CANALES_A_LEER = [0, 1, 2, 3] # Change this according to how many gauges you have connected (e.g. [0, 1, 2, 3])

# Per-channel calibration based on your measurements at 0 g, 500 g and 1910 g.
# A linear approximation is used: grams = slope * voltage_ratio + intercept
CALIBRACION_CELDAS = {
    0: {'slope': -4642909.486589756, 'intercept': -290.0718507585548},
    1: {'slope': -4821786.234979635, 'intercept': -137.17214442022848},
    2: {'slope': -5136768.852164714, 'intercept': -29.88481961346418},
    3: {'slope': -4821810.751993669, 'intercept': -749.9326369088943},
}
MAX_GRAMOS = 5000.0
MIN_GRAMOS = 0.0

celdas = {}


def convertir_a_gramos(canal, voltage_ratio):
    datos = CALIBRACION_CELDAS.get(canal)
    if datos is None:
        return None
    gramos = datos['slope'] * voltage_ratio + datos['intercept']
    gramos = max(MIN_GRAMOS, min(MAX_GRAMOS, gramos))
    return gramos


def al_recibir_cambio_voltaje(self, voltageRatio):
    """ Function that runs automatically every time the gauge reads a value """
    canal = self.getChannel()
    gramos = convertir_a_gramos(canal, voltageRatio)
    celdas[canal] = {
        'voltage_ratio': voltageRatio,
        'grams': gramos,
    }

def inicializar_galgas():
    """ Initializes and connects the load cells in parallel """
    for canal in CANALES_A_LEER:
        try:
            ch = VoltageRatioInput()
            ch.setChannel(canal)
            
            # We assign the function that will handle the data when they change
            ch.setOnVoltageRatioChangeHandler(al_recibir_cambio_voltaje)
            
            # We open the channel with a 5-second timeout
            ch.openWaitForAttachment(5000)
            
            # Performance configuration (sampling rate in ms)
            # 32ms is ~31.25 Hz, ideal to align with your 30 Hz OptiTrack
            ch.setDataInterval(32) 
            
            celdas[canal] = {'voltage_ratio': 0.0, 'grams': 0.0}
            print(f"-> Galga en Canal {canal} conectada y configurada exitosamente.")
            
        except PhidgetException as e:
            print(f"❌ Error al abrir el Canal {canal}: {e.description} (Código: {e.code})")

# =====================================================================
# DIRECT READING TEST
# =====================================================================
if __name__ == "__main__":
    print("Inicializando puente de galgas Phidget...")
    inicializar_galgas()
    
    print("\nComenzando lectura. Presiona Ctrl+C para detener...")
    try:
        while True:
            # Here you have the clean values in a dictionary ready for your code
            # celdas[0] is the reading of channel 0, celdas[1] of channel 1, etc.
            line = []
            for ch in CANALES_A_LEER:
                info = celdas.get(ch, {'voltage_ratio': 0.0, 'grams': 0.0})
                line.append(f"G{ch}: {info['voltage_ratio']:.6e} V/V ({info['grams']:.1f} g)")
            print("Lecturas actuales -> " + " | ".join(line), end="\r")
            
            # This delay simulates the cycle of your acquisition loop (30 Hz)
            time.sleep(0.033)
            
    except KeyboardInterrupt:
        print("\nLectura finalizada por el usuario.")