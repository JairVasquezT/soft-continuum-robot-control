import sys
import os
import threading
from NatNetClient import NatNetClient

class MiClienteOptiTrack:
    def __init__(self, client_ip="127.0.0.1", server_ip="127.0.0.1", multicast=False):
        self.streaming_client = NatNetClient()
        self.streaming_client.set_client_address(client_ip)
        self.streaming_client.set_server_address(server_ip)
        self.streaming_client.set_use_multicast(multicast)
        
    def start(self, rigid_body_callback):
        # 🎯 Vinculamos el callback que vendrá desde tu MAIN
        self.streaming_client.rigid_body_listener = rigid_body_callback
        is_running = self.streaming_client.run()
        if not is_running or not self.streaming_client.connected():
            print("ERROR: No se pudo conectar con Motive")
            return False
        return True
        
    def stop(self):
        self.streaming_client.shutdown()

        