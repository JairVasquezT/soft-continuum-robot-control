import time
from Phidget22.Devices.VoltageRatioInput import *
from Phidget22.PhidgetException import *

# =====================================================================
# CONFIGURACIÓN DE LAS CELDAS DE CARGA (GALGAS)
# =====================================================================
# El PhidgetBridge 4-Input tiene 4 canales (del 0 al 3)
CANALES_A_LEER = [0, 1, 2, 3] # Modifica esto según cuántas galgas tengas conectadas (ej. [0, 1, 2, 3])

# Calibración por canal basada en tus medidas de 0 g, 500 g y 1910 g.
# Se usa una aproximación lineal: gramos = slope * voltage_ratio + intercept
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
    """ Función que se ejecuta automáticamente cada vez que la galga lee un valor """
    canal = self.getChannel()
    gramos = convertir_a_gramos(canal, voltageRatio)
    celdas[canal] = {
        'voltage_ratio': voltageRatio,
        'grams': gramos,
    }

def inicializar_galgas():
    """ Inicializa y conecta las celdas de carga en paralelo """
    for canal in CANALES_A_LEER:
        try:
            ch = VoltageRatioInput()
            ch.setChannel(canal)
            
            # Asignamos la función que manejará los datos cuando cambien
            ch.setOnVoltageRatioChangeHandler(al_recibir_cambio_voltaje)
            
            # Abrimos el canal con un tiempo de espera de 5 segundos
            ch.openWaitForAttachment(5000)
            
            # Configuración de rendimiento (frecuencia de muestreo en ms)
            # 32ms equivale a ~31.25 Hz, ideal para alinearse con tus 30 Hz del OptiTrack
            ch.setDataInterval(32) 
            
            celdas[canal] = {'voltage_ratio': 0.0, 'grams': 0.0}
            print(f"-> Galga en Canal {canal} conectada y configurada exitosamente.")
            
        except PhidgetException as e:
            print(f"❌ Error al abrir el Canal {canal}: {e.description} (Código: {e.code})")

# =====================================================================
# PRUEBA DE LECTURA DIRECTA
# =====================================================================
if __name__ == "__main__":
    print("Inicializando puente de galgas Phidget...")
    inicializar_galgas()
    
    print("\nComenzando lectura. Presiona Ctrl+C para detener...")
    try:
        while True:
            # Aquí tienes los valores limpios en un diccionario listos para tu código
            # celdas[0] es la lectura del canal 0, celdas[1] del canal 1, etc.
            line = []
            for ch in CANALES_A_LEER:
                info = celdas.get(ch, {'voltage_ratio': 0.0, 'grams': 0.0})
                line.append(f"G{ch}: {info['voltage_ratio']:.6e} V/V ({info['grams']:.1f} g)")
            print("Lecturas actuales -> " + " | ".join(line), end="\r")
            
            # Este delay simula el ciclo de tu bucle de adquisición (30 Hz)
            time.sleep(0.033)
            
    except KeyboardInterrupt:
        print("\nLectura finalizada por el usuario.")