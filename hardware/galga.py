from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *

CANALES_A_LEER = [0, 1, 2, 3]

CALIBRACION_CELDAS = {
    0: {'slope': -4642909.486589756, 'intercept': -290.0718507585548},
    1: {'slope': -4821786.234979635, 'intercept': -137.17214442022848},
    2: {'slope': -5136768.852164714, 'intercept': -29.88481961346418},
    3: {'slope': -4821810.751993669, 'intercept': -749.9326369088943},
}
MAX_GRAMOS = 5000.0
MIN_GRAMOS = 0.0

class PhidgetForceController:
    def __init__(self, canales=CANALES_A_LEER, calibracion=CALIBRACION_CELDAS):
        self.canales_ids = canales
        self.calibracion = calibracion
        self.celdas = {}
        self._inicializar_sensores()

    def _inicializar_sensores(self):
        for ch in self.canales_ids:
            try:
                ch_obj = VoltageRatioInput()
                ch_obj.setChannel(ch)
                ch_obj.openWaitForAttachment(1000)
                ch_obj.setDataInterval(16)  # ~60 Hz internal refresh rate in Phidget
                self.celdas[ch] = ch_obj
            except Exception as e:
                print(f"⚠️ Error al conectar canal Phidget {ch}: {e}")

    def leer_fuerzas_gramos(self):
        fuerzas = []
        for ch in self.canales_ids:
            if ch in self.celdas:
                try:
                    ratio = self.celdas[ch].getVoltageRatio()
                    cal = self.calibracion[ch]
                    gramos = (cal['slope'] * ratio) + cal['intercept']
                    gramos = max(MIN_GRAMOS, min(MAX_GRAMOS, gramos))
                    fuerzas.append(round(gramos, 2))
                except Exception:
                    fuerzas.append(0.0)
            else:
                fuerzas.append(0.0)
        return fuerzas

    def close(self):
        for ch_obj in self.celdas.values():
            try:
                ch_obj.close()
            except Exception:
                pass