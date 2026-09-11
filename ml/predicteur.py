import time
import torch
import torch.nn as nn


class EncoderDecoderSoftRobot(nn.Module):

  def __init__(
      self,
      hist_input_dim=13,  # Dimensiones del pasado (ej. V7: 9 cinemática + 4 tensión)
      future_action_dim=4,  # Acciones futuras candidatas (ej. 4 delta_meta_m)
      hidden_dim=128,
      num_layers=2,
      output_dim=3,
      dropout=0.1,
  ):
    super(EncoderDecoderSoftRobot, self).__init__()

    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    # 1. ENCODER: Procesa los 20 pasos de historial
    self.encoder = nn.LSTM(
        hist_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
    )

    # 2. DECODER: Recibe las acciones futuras proyectadas (10 pasos) + hidden state del encoder
    self.decoder = nn.LSTM(
        future_action_dim,
        hidden_dim,
        num_layers,
        batch_first=True,
        dropout=dropout,
    )

    # 3. CABEZA DE SALIDA: Proyecta la memoria a coordenadas (x, y, z) 3D
    self.fc_out = nn.Sequential(
        nn.Linear(hidden_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),  # 3D: rel_x, rel_y, rel_z
    )

  def encode_history(self, x_hist):
    """Procesa el historial pasado de 20 pasos.

    x_hist shape: (Batch_Size, 20, hist_input_dim)
    """
    _, (h_n, c_n) = self.encoder(x_hist)
    return h_n, c_n

  def decode_future(self, x_future_actions, h_state, c_state):
    """Predice los 10 pasos futuros en base a las acciones candidatas.

    x_future_actions shape: (N_candidates, 10, future_action_dim)
    """
    out_dec, _ = self.decoder(x_future_actions, (h_state, c_state))
    # out_dec shape: (N_candidates, 10, hidden_dim)
    predictions = self.fc_out(out_dec)
    # predictions shape: (N_candidates, 10, 3)
    return predictions

  def forward(self, x_hist, x_future_actions):
    """Entrenamiento estándar Batch a Batch."""
    h_n, c_n = self.encode_history(x_hist)
    return self.decode_future(x_future_actions, h_n, c_n)

  def predict_candidates(self, x_hist_single, x_candidates):
    """Inferencia ultrarrápida para MPC en tiempo real.

    - x_hist_single: (1, 20, hist_input_dim) -> Historial actual de la planta.
    - x_candidates:  (N_cand, 10, future_action_dim) -> N secuencias de control a evaluar.
    """
    N_cand = x_candidates.size(0)

    # 1. Codificar el pasado UNA SOLA VEZ
    h_single, c_single = self.encode_history(x_hist_single)

    # 2. Replicar el estado latente para los N candidatos en GPU (Memory Expand)
    # Shape resultante: (num_layers, N_cand, hidden_dim)
    h_expanded = h_single.expand(-1, N_cand, -1).contiguous()
    c_expanded = c_single.expand(-1, N_cand, -1).contiguous()

    # 3. Decodificar todas las trayectorias en paralelo
    return self.decode_future(x_candidates, h_expanded, c_expanded)


# =====================================================================
# DEMOSTRACIÓN Y PRUEBA DE RENDIMIENTO (BENCHMARK EN PARALELO)
# =====================================================================
if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'⚡ Evaluando Predictor MPC en: {device}')

  # Hiperparámetros de la prueba
  T_IN = 20  # Historial pasado
  T_OUT = 10  # Horizonte de predicción
  N_CANDIDATES = 2000  # ¡2,000 secuencias candidatas en paralelo!

  model = EncoderDecoderSoftRobot(
      hist_input_dim=13, future_action_dim=4, hidden_dim=128, num_layers=2
  ).to(device)

  model.eval()

  # Simulación de entradas
  # 1 historial real del robot (1 batch, 20 pasos, 13 variables)
  dummy_hist = torch.randn(1, T_IN, 13, device=device)

  # 2,000 candidatos de control futuro generados por un MPC (2000 batch, 10 pasos, 4 motores)
  dummy_candidates = torch.randn(N_CANDIDATES, T_OUT, 4, device=device)

  # Warmup de GPU
  with torch.no_grad():
    _ = model.predict_candidates(dummy_hist, dummy_candidates)

  # Medición de tiempo de inferencia
  start_time = time.time()
  with torch.no_grad():
    # Matriz de salida: (2000 candidatos, 10 pasos futuros, 3D posiciones)
    predicciones_3d = model.predict_candidates(dummy_hist, dummy_candidates)

  if device.type == 'cuda':
    torch.cuda.synchronize()

  elapsed_ms = (time.time() - start_time) * 1000.0

  print('\n✅ Test completado con éxito:')
  print(f' ▫ Shape de salida: {predicciones_3d.shape} (N_candidatos, T_out, 3D)')
  print(
      f' ▫ Tiempo de evaluación para {N_CANDIDATES} candidatos:'
      f' {elapsed_ms:.2f} ms'
  )