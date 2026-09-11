import time
from collections import deque

import torch
import torch.nn as nn


# ==============================================================================
# 1. GENERADOR DE CANDIDATOS MULTI-ESCALA CON ZONA MUERTA SUAVE
# ==============================================================================
class CandidateGenerator:

  def __init__(self, metadata_json=None, device="cuda"):
    self.device = device

    # 1. Cargar HOMEs reales desde el JSON
    if metadata_json and "home_motores" in metadata_json:
      homes = metadata_json["home_motores"]
      self.homes = torch.tensor(
          [homes["m1"], homes["m2"], homes["m3"], homes["m4"]], device=device
      )
    else:
      # Valores fallback
      self.homes = torch.tensor(
          [1871.0, 1951.0, 1485.0, 1712.0], device=device
      )

    # 2. Rangos dinámicos: ANCHO TOTAL por motor (no semi-amplitud). DEBE
    # coincidir EXACTO con la escala que usó dataset_filtre.py/dataset_pred_filt.py
    # para delta_real_mX/delta_meta_mX (RANGOS_MANUALES: radio 650 para
    # m1/m2, 750 para m3/m4 -> ancho 1300/1500) -- esto no es un margen de
    # seguridad ajustable, es la escala con la que el predictor fue
    # entrenado. Si no coincide, cada u_cand que recibe el modelo queda mal
    # interpretado (una magnitud de comando distinta a la que el generador
    # de candidatos cree que está mandando). El margen de seguridad real
    # (dónde el CEM debería EVITAR converger, aunque el clamp lo permita) va
    # en margen_confort_ticks/w_limite (NeuralMPCController), no acá.
    self.ranges = torch.tensor([1300.0, 1300.0, 1500.0, 1500.0], device=device)

    # 3. Zonas muertas en ticks
    self.deadbands = torch.tensor(
        [[-6.0, 4.0], [-13.0, 8.0], [-6.0, 5.0], [-6.0, 4.0]], device=device
    )

  def desescalar_u_ticks(self, u_scaled):
    """Convierte de [-1, 1] a Ticks Absolutos [HOME - Range/2, HOME + Range/2]."""
    return self.homes + u_scaled * (self.ranges / 2.0)

  def escalar_u_ticks(self, u_ticks):
    """Convierte de Ticks Absolutos a [-1, 1]."""
    return (u_ticks - self.homes) / (self.ranges / 2.0)

  def generar_candidatos_cem(self, mean_ticks, std_ticks, num_samples, frac_global=0.1):
    """Genera candidatos para UNA iteración de CEM, en ticks absolutos.

    Mayoría muestreada de N(mean_ticks, std_ticks) (elite gaussiano), más una
    fracción pequeña de exploración global uniforme en todo el rango físico
    para no quedar atrapado en un mínimo local si la media converge mal.
    """
    num_global = max(1, int(num_samples * frac_global))
    num_local = num_samples - num_global

    # A. Muestreo local: Gaussiana centrada en la media del CEM
    delta_local_ticks = torch.randn(num_local, 4, device=self.device) * std_ticks.unsqueeze(0)
    u_local_ticks = mean_ticks.unsqueeze(0) + delta_local_ticks

    # B. Muestreo global uniforme (en todo el rango físico [-1, 1])
    u_global_scaled = torch.rand(num_global, 4, device=self.device) * 2.0 - 1.0
    u_global_ticks = self.desescalar_u_ticks(u_global_scaled)

    # C. Unir todos los candidatos en ticks absolutos
    u_cands_ticks = torch.cat([u_local_ticks, u_global_ticks], dim=0)

    # D. Aplicar límites físicos estrictos por motor
    min_ticks = self.homes - (self.ranges / 2.0)
    max_ticks = self.homes + (self.ranges / 2.0)
    u_cands_ticks = torch.clamp(
        u_cands_ticks, min_ticks.unsqueeze(0), max_ticks.unsqueeze(0)
    )

    # El candidato 0 siempre es la media actual (garantiza no perder la mejor estimación)
    u_cands_ticks[0] = mean_ticks

    return u_cands_ticks


# ==============================================================================
# 2. FUNCIÓN DE COSTE CORREGIDA
# ==============================================================================
class MPCCostFunction:

    def __init__(self, w_pos=1.5, w_smooth=0.1, w_terminal=5.0,
               error_axis_weights=(1.0, 1.0, 1.0), deadband_mm=0.0):
        # NOTA: Se aumentó w_pos/w_terminal y se redujo w_smooth para dar prioridad al Target
        self.w_pos = w_pos
        self.w_smooth = w_smooth
        self.w_terminal = w_terminal
        if len(error_axis_weights) != 3:
            raise ValueError('error_axis_weights debe tener 3 valores: [X, Y, Z]')
        self.error_axis_weights = tuple(float(weight) for weight in error_axis_weights)
        if deadband_mm < 0:
            raise ValueError('deadband_mm debe ser mayor o igual que cero')
        self.deadband_mm = float(deadband_mm)

    def compute_cost(
        self, y_preds_mm, y_ref_mm, u_cands_scaled, u_current_scaled
    ):
      # Error cuadrático ponderado por eje con zona muerta en mm
      axis_weights = torch.tensor(
          self.error_axis_weights, device=y_preds_mm.device, dtype=y_preds_mm.dtype
      )
      errors = torch.abs(y_preds_mm - y_ref_mm.unsqueeze(0))
      error_outside_deadband = torch.relu(errors - self.deadband_mm)
      axis_costs = error_outside_deadband.square() * axis_weights
      err_pos = torch.sum(axis_costs, dim=-1)

      # Costes de seguimiento
      cost_tracking = torch.mean(err_pos[:, :-1], dim=1)
      cost_terminal = err_pos[:, -1]

      # Coste de esfuerzo CORREGIDO: Evaluado contra la posición actual del motor
      cost_effort = torch.norm(
          u_cands_scaled - u_current_scaled.unsqueeze(0), dim=-1
      )

      # Coste Total
      total_cost = (
          self.w_pos * cost_tracking
          + self.w_terminal * cost_terminal
          + self.w_smooth * cost_effort
      )

      return total_cost


# ==============================================================================
# 3. CONTROLADOR NEURAL MPC
# ==============================================================================
class NeuralMPCController:

  def __init__(
      self,
      model_predictor,
      metadata_json,
      num_samples=300,
      t_out=10,
      device="cuda",
      n_cem_iters=2,
      elite_frac=0.15,
      std_floor_min=9.0,
      max_step_ticks=150.0,
      stagnation_window=15,
      stagnation_min_improvement_mm=3.0,
      w_persistencia=0.16,
      w_limite=0.05,
      margen_confort_ticks=100.0,
      target_tolerance_mm=10.0,
      error_chico_factor=2.0,
      anclas_fijas_ticks=None,
      modo_control='hibrido',
      k_iters_jacobiano=2,
  ):
    self.predictor = model_predictor.to(device).eval()
    self.device = device
    self.num_samples = num_samples
    self.t_out = t_out
    self.stagnation_counter = 0

    # 🎛️ Modo de control -- tres opciones excluyentes:
    #   'cem': solo Cross-Entropy Method (multi-arranque con anclas
    #          aleatorias/home/fijas, SIN el ancla de Jacobiano).
    #   'jacobiano': solo cinemática inversa vía Jacobiano local (autograd) +
    #          pseudo-inversa amortiguada, un paso por ciclo -- sin CEM ni
    #          multi-arranque, para aislar qué tanto aporta cada método por
    #          separado.
    #   'hibrido' (default, comportamiento de siempre): Jacobiano como UNA
    #          ancla más del multi-arranque del CEM, que compite por costo
    #          con las demás y luego se refina con las iteraciones de CEM.
    if modo_control not in ('cem', 'jacobiano', 'hibrido'):
      raise ValueError(
          f"modo_control debe ser 'cem', 'jacobiano' o 'hibrido', no {modo_control!r}"
      )
    self.modo_control = modo_control

    # 🔁 Cantidad de iteraciones de re-linealización local del ancla de
    # Jacobiano (Gauss-Newton local): K=1 recalcula el Jacobiano una sola
    # vez en el punto de partida (comportamiento histórico); K=2-3 vuelve a
    # evaluar el predictor Y el Jacobiano en el punto YA CORREGIDO por el
    # paso anterior, acercando la aproximación lineal al comportamiento real
    # del predictor conforme el candidato se aleja del punto inicial. Ver
    # _ancla_jacobiano_ik.
    self.k_iters_jacobiano = max(1, int(k_iters_jacobiano))

    # 🧪 Anclas fijas EXTRA para el multi-arranque (diagnóstico): candidatos
    # de ticks conocidos-buenos (p.ej. semillas confirmadas a mano para
    # puntos difíciles) que compiten con las demás anclas en pie de
    # igualdad -- se evalúan con el MISMO costo (incluida la barrera de
    # límites) y solo "ganan" si de verdad tienen el menor costo ese ciclo.
    # A diferencia de SEED_COMMANDS_TICKS/USAR_SEMILLAS_FIJAS (que bypassea
    # el CEM por completo), esto no fuerza nada: es una opción más en el menú.
    if anclas_fijas_ticks:
      self.anclas_fijas_ticks = torch.tensor(
          anclas_fijas_ticks, device=device, dtype=torch.float32
      )
    else:
      self.anclas_fijas_ticks = None

    # 📏 Umbral de "error chico" ATADO a la tolerancia de llegada real, no a
    # un número fijo. Antes era un corte duro en 30mm sin importar cuál
    # fuera target_tolerance_mm -- con tolerancia de 8mm, un error de 27-30mm
    # (3.4-3.75x la tolerancia real) caía igual en el bucket "ya casi
    # llegaste, no ensanches la búsqueda", desactivando el escape de
    # estancamiento justo cuando más hacía falta (visto en producción: se
    # quedaba plano en ~27-30mm sin que ni el ensanche de std ni el
    # multi-arranque se activaran nunca, porque error_chico=True los
    # bloquea a ambos). Con esto, "error chico" es relativo a qué tan lejos
    # de la tolerancia real está el robot.
    self.error_chico_umbral_mm = float(target_tolerance_mm) * float(error_chico_factor)

    # 🪤 Detección de estancamiento por VENTANA (no ciclo-a-ciclo): comparar
    # solo contra el error del ciclo anterior es sensible al ruido de medición
    # (típico del observador LSTM/compliance del cable) -- si el error real
    # está estancado pero oscilando (12->14->11->13mm), la diferencia entre
    # ciclos consecutivos casi siempre supera el umbral y el contador nunca
    # llega a disparar el ensanche de búsqueda. En vez de eso, se guarda el
    # historial de error de los últimos `stagnation_window` ciclos y se mide
    # si el MEJOR error dentro de esa ventana mejoró lo suficiente respecto al
    # arranque de la ventana.
    self.stagnation_min_improvement_mm = float(stagnation_min_improvement_mm)
    self.error_history = deque(maxlen=max(2, int(stagnation_window)))
    self.last_target_pos_mm = None  # para detectar cambio de waypoint

    # 🧲 Persistencia direccional: penaliza candidatos que REVIERTEN la
    # dirección del último movimiento neto aplicado. En un sistema de cables
    # con holgura/backlash, un comando que tira y afloja sin comprometerse
    # con una dirección puede no traducirse en movimiento físico real (visto
    # en producción: Opti real casi congelado mientras el comando oscilaba
    # fuerte de un lado a otro). Esto empuja al optimizador a consolidar el
    # movimiento en vez de dudar.
    self.w_persistencia = float(w_persistencia)
    self.last_delta_ticks = None  # dirección neta del último ciclo, persistida

    # 🚧 Evitar soluciones que exploran más allá de lo que el robot REAL
    # visitó durante la recolección de datos. margen_confort_ticks=100
    # confirmado con los máximos/mínimos reales logueados por motor: los 4
    # motores muestran la MISMA brecha de ~100 ticks entre el radio asumido
    # por RANGOS_MANUALES (650/750, usado para escalar el entrenamiento) y
    # el radio real efectivamente visitado (~550/~650) -- la rutina de
    # recolección dejó ese margen de seguridad pero el script de
    # preprocesamiento nunca lo restó. No es una barrera de redundancia
    # cinemática (eso resultó contraproducente, ver commit anterior con
    # w_limite=0) -- es evitar candidatos en territorio que el modelo nunca
    # vio de verdad, aunque la escala en teoría lo permita. Penaliza
    # suavemente (barrera, no
    # límite duro -- eso ya lo hace el clamp en generar_candidatos_cem) a los
    # candidatos que entran en los últimos `margen_confort_ticks` antes de
    # cualquiera de los dos límites de cada motor.
    self.w_limite = float(w_limite)
    self.margen_confort_ticks = float(margen_confort_ticks)

    # 🎯 CEM (Cross-Entropy Method): en vez de argmin sobre candidatos i.i.d.
    # nuevos cada ciclo, se mantiene una distribución (media/std) que se
    # refina en `n_cem_iters` iteraciones por ciclo, y cuya STD persiste
    # entre ciclos de control (memoria) para converger suave en vez de saltar
    # entre "empates" aleatorios cerca del target (la causa del chattering).
    self.n_cem_iters = max(1, int(n_cem_iters))
    self.elite_frac = elite_frac
    self.std_floor_min = std_floor_min
    self.cem_std_ticks = None  # persistido entre llamadas a optimize()

    # 🛑 Límite de salto por ciclo (ticks, norma euclídea sobre los 4 motores):
    # la media del CEM no puede alejarse más de esto del punto de partida del
    # ciclo, ni siquiera en iteraciones intermedias. Evita que el predictor
    # (mal extrapolado fuera de lo entrenado) empuje la solución a "otra rama"
    # de la cinemática inversa de un salto, disparando el error real aunque
    # el coste PREDICHO parezca bajo.
    self.max_step_ticks = float(max_step_ticks)

    self.candidate_generator = CandidateGenerator(metadata_json, device=device)
    # Pesos reajustados para obligar al robot a romper la resistencia de los $45\text{ mm}$
    self.cost_fn = MPCCostFunction(
        w_pos=1.5,
        w_smooth=0.2,
        w_terminal=5.0,
        error_axis_weights=(1.0, 1.0, 1.0),#1-2.5-1
        deadband_mm=0.0,#5
    )

    y_trans = metadata_json["Y_transformer"]
    self.min_y = torch.tensor(
        [
            y_trans["rel_x"]["min_t"],
            y_trans["rel_y"]["min_t"],
            y_trans["rel_z"]["min_t"],
        ],
        device=device,
    )
    self.max_y = torch.tensor(
        [
            y_trans["rel_x"]["max_t"],
            y_trans["rel_y"]["max_t"],
            y_trans["rel_z"]["max_t"],
        ],
        device=device,
    )

  def _desescalar_y_mm(self, y_scaled_tensor):
    pos_m = (
        self.min_y + (y_scaled_tensor + 1.0) * (self.max_y - self.min_y) / 2.0
    )
    return pos_m * 1000.0

  def _ancla_jacobiano_ik(self, x_hist_input, mean_ticks_inicial, target_pos_mm, lam=1e-2, k_iters=None):
    """Candidato de cinemática inversa vía Jacobiano local del predictor
    (autograd) + pseudo-inversa amortiguada (Damped Least Squares), con
    RE-LINEALIZACIÓN iterativa (Gauss-Newton local, `k_iters` pasos, por
    defecto `self.k_iters_jacobiano`):

      u_0 = mean_ticks_inicial
      Para step = 0 .. k_iters-1:
        x̂_step = Predictor(u_step)                       (forward, en u_step)
        J_step = d(Predictor(u))/du |_{u=u_step}           (Jacobiano en u_step)
        Δq = J_step⁺ (target_pos_mm - x̂_step)              (pseudo-inversa amortiguada)
        u_{step+1} = u_step + Δq
      devuelve u_{k_iters}

    Con k_iters=1 es el comportamiento histórico (un solo paso). Con 2-3,
    cada paso vuelve a evaluar el predictor Y el Jacobiano en el punto YA
    CORREGIDO por el paso anterior -- la aproximación lineal se mantiene
    fiel al predictor real incluso si el candidato se aleja bastante del
    punto de partida (a diferencia de un solo paso, que usa una linealización
    congelada en u_0 para todo el desplazamiento).

    NOTA: x̂_step SIEMPRE sale del forward del predictor (nunca de una
    medición externa como current_pos_mm) -- así el paso queda
    autoconsistente con el modelo que el propio Jacobiano describe, en
    cada re-linealización, no solo en la primera.

    `optimize()` corre bajo @torch.inference_mode(), que es incompatible
    con autograd -- por eso este método reabre gradientes explícitamente y
    clona los tensores de entrada (los tensores creados en inference_mode
    no pueden usarse en autograd ni siquiera reabriendo el contexto). Cada
    iteración vuelve a hacer `.detach()` del resultado antes de la
    siguiente relinealización, para no acumular el grafo de autograd de
    los pasos previos (cada Jacobiano se calcula independiente, no
    diferenciando "a través" de las iteraciones anteriores).

    Si algo falla numéricamente (Jacobiano casi singular, etc.) en
    cualquier paso, devuelve None y el llamador simplemente no suma esta
    ancla. Trabaja sobre un solo candidato (vector de 4 ticks) -- si en el
    futuro se calculan varias anclas de Jacobiano en batch, las mismas
    operaciones matriciales (@, .T, torch.linalg.solve) siguen siendo
    válidas agregando una dimensión de batch al frente.
    """
    if k_iters is None:
      k_iters = self.k_iters_jacobiano
    try:
      with torch.inference_mode(False), torch.enable_grad():
        x_hist_ik = x_hist_input.clone()
        target_ik = target_pos_mm.clone().detach()

        def f(ticks_):
          u_scaled_ = self.candidate_generator.escalar_u_ticks(ticks_)
          y_scaled = self.predictor(x_hist_ik, u_scaled_.unsqueeze(0))[0, -1, :3]
          return self._desescalar_y_mm(y_scaled)

        u_step = mean_ticks_inicial.clone().detach()
        for step in range(k_iters):
          # Relinealizar exactamente en u_step: gradiente propio de este
          # paso, no heredado del anterior (por eso el .detach() de arriba
          # y el nuevo requires_grad_ acá).
          u_step = u_step.detach().requires_grad_(True)

          x_hat_step = f(u_step)  # x̂_step = Predictor(u_step), en ESTE punto
          J_step = torch.autograd.functional.jacobian(f, u_step)  # [3,4] mm/tick, en u_step

          dx = (target_ik - x_hat_step).detach()
          JJt = J_step @ J_step.T + lam * torch.eye(3, device=self.device)
          dq = J_step.T @ torch.linalg.solve(JJt, dx)

          u_step = u_step.detach() + dq.detach()  # u_{step+1}, sin arrastrar el grafo

      resultado = u_step.detach()
      if not torch.isfinite(resultado).all():
        return None
      return resultado
    except Exception:
      return None

  @torch.inference_mode()
  def optimize(self, x_hist_tensor, y_ref_mm, u_current_scaled, current_pos_mm):
    t_start = time.perf_counter()

    # Asegurar dimensión de batch en x_hist_tensor [1, seq_len, features]
    if x_hist_tensor.dim() == 2:
        x_hist_input = x_hist_tensor.unsqueeze(0)
    else:
        x_hist_input = x_hist_tensor

    # --------------------------------------------------------------------------
    # A. CÁLCULO DEL RADIO ADAPTATIVO Y DETECCIÓN DE ESTANCAMIENTO
    # --------------------------------------------------------------------------
    # Error real percibido por la visión/sensores: compara la posición REAL
    # actual (OptiTrack u observador, la misma que cierra el lazo de control,
    # pasada por el llamador) contra el target, ambos en mm/Cartesiano. Antes
    # se usaba una pasada extra del predictor "adivinando" la posición actual
    # a partir de su propio historial — eso no es el error real, es lo que el
    # modelo CREE, y puede estar sesgado respecto a lo que el sensor mide.
    target_pos_mm = y_ref_mm[0] if y_ref_mm.dim() > 1 else y_ref_mm
    err_actual_mm = torch.norm(target_pos_mm - current_pos_mm).item()

    # 🔄 Cambio de waypoint: limpiar la ventana de estancamiento y la memoria
    # de STD del CEM. Sin esto, al pasar de un target ya convergido (error
    # chico) a uno nuevo lejano, la ventana queda con una mezcla de errores
    # viejos (chicos) y nuevos (grandes) -> "mejora≈0" falso -> ensanche de
    # búsqueda innecesario justo al llegar al nuevo punto (comportamiento
    # errático). cem_std_ticks también se limpia para no arrastrar una STD
    # ya angosta del waypoint anterior (aunque el clamp con std_floor ya
    # protege esto, limpiar es más claro y evita dudas).
    es_target_nuevo = (
        self.last_target_pos_mm is None
        or torch.norm(target_pos_mm - self.last_target_pos_mm).item() > 1.0
    )
    if es_target_nuevo:
        self.error_history.clear()
        self.stagnation_counter = 0
        self.cem_std_ticks = None
        self.last_delta_ticks = None  # no arrastrar "dirección buena" de otro target
    self.last_target_pos_mm = target_pos_mm.clone()

    # Piso de la desviación estándar (STD) según la magnitud del error: define
    # cuánto puede explorar el CEM como mínimo, incluso si ya convergió a algo
    # más angosto en un ciclo anterior (evita quedar "demasiado seguro" si
    # aparece un target nuevo lejano).
    # frac_global: fracción de candidatos de exploración global (uniforme en
    # todo el rango físico). Útil con error grande para escapar mínimos
    # locales; con error chico es precisamente el vector que mete un
    # candidato "de otra rama" cuando ya casi se llegó -> se reduce casi a 0.
    if err_actual_mm > 70.0:  # Error > 7 cm
        std_floor = 45.0
        frac_global = 0.10
        error_chico = False
    elif err_actual_mm > self.error_chico_umbral_mm:  # entre el umbral real y 7cm
        std_floor = 32.0
        frac_global = 0.05
        error_chico = False
    else:  # Error genuinamente chico (relativo a target_tolerance_mm real)
        std_floor = 23.0
        frac_global = 0.01
        error_chico = True

    # Detección de Estancamiento por ventana: se guarda el error de este
    # ciclo, y una vez llena la ventana se compara el error del INICIO de la
    # ventana contra el MEJOR (mínimo) error visto dentro de ella. Si la
    # mejora es menor a stagnation_min_improvement_mm, está genuinamente
    # estancado (no es solo ruido puntual entre dos ciclos consecutivos).
    self.error_history.append(err_actual_mm)
    if len(self.error_history) == self.error_history.maxlen:
        mejora = self.error_history[0] - min(self.error_history)
        estancado = mejora < self.stagnation_min_improvement_mm
    else:
        estancado = False

    # OJO: con error ya chico (<3cm) NO se ensancha por estancamiento. Ahí
    # normalmente ya estás cerca de lo mejor que el modelo puede lograr;
    # ensanchar la búsqueda no "escapa un mínimo local", empuja a explorar
    # candidatos más lejanos que el predictor puede sobrestimar y termina
    # alejando al robot de un punto que ya era razonablemente bueno (visto
    # en la práctica: quedaba dando vueltas en ~15mm y terminaba en 40mm).
    if estancado and not error_chico:
        self.stagnation_counter += 1
        # Tope absoluto: nunca ensanchar más allá del bucket "error grande"
        # (90) * 1.5, para que un falso disparo no vuelva la búsqueda tan
        # amplia que el candidato elegido termine siendo casi ruido puro.
        std_floor = min(std_floor * 1.8, 135.0)
    else:
        self.stagnation_counter = 0

    std_floor = max(std_floor, self.std_floor_min)

    # --------------------------------------------------------------------------
    # B. CEM (Cross-Entropy Method): refina media/std en `n_cem_iters`
    # iteraciones. La media SIEMPRE arranca en el estado real actual (lo que
    # MPC.py ya reporta como u_current_scaled tras EMA/histéresis externas),
    # así el optimizador nunca "cree" estar en un lugar distinto al robot
    # real. La STD sí se hereda entre ciclos de control (memoria): en vez del
    # salto discreto de "radio adaptativo" de antes, converge suave y no
    # colapsa por debajo de std_floor.
    # --------------------------------------------------------------------------
    mean_ticks_inicial = self.candidate_generator.desescalar_u_ticks(u_current_scaled)

    # 🧭 MODO JACOBIANO PURO: self.k_iters_jacobiano pasos de cinemática
    # inversa vía Jacobiano local (re-linealizado en cada paso) + pseudo-
    # inversa amortiguada, recalculados desde la posición ACTUAL cada ciclo
    # -- sin CEM, sin multi-arranque. Retorna temprano; el resto de
    # optimize() (CEM/multi-arranque) no aplica acá.
    if self.modo_control == 'jacobiano':
        ancla_jacobiano = self._ancla_jacobiano_ik(
            x_hist_input, mean_ticks_inicial, target_pos_mm
        )
        # Si el Jacobiano es numéricamente inválido este ciclo (casi
        # singular, etc.), no mover -- más seguro que un salto arbitrario.
        mean_ticks_jac = (
            mean_ticks_inicial.clone() if ancla_jacobiano is None else ancla_jacobiano
        )

        # Mismo límite de salto por ciclo que el modo CEM/híbrido: nunca
        # "teletransportar" aunque el Jacobiano proponga un salto grande.
        delta_jac = mean_ticks_jac - mean_ticks_inicial
        dist_jac = torch.norm(delta_jac)
        if dist_jac > self.max_step_ticks:
            mean_ticks_jac = mean_ticks_inicial + delta_jac * (self.max_step_ticks / dist_jac)

        u_best_jac = torch.clamp(
            self.candidate_generator.escalar_u_ticks(mean_ticks_jac), -1.0, 1.0
        )
        y_pred_final_scaled_jac = self.predictor(x_hist_input, u_best_jac.unsqueeze(0))
        y_pred_final_mm_jac = self._desescalar_y_mm(y_pred_final_scaled_jac[0, :, :3])
        best_cost_jac = self.cost_fn.compute_cost(
            y_pred_final_mm_jac.unsqueeze(0), y_ref_mm, u_best_jac.unsqueeze(0), u_current_scaled
        )[0].item()

        self.last_delta_ticks = delta_jac.clone()
        t_calc_ms_jac = (time.perf_counter() - t_start) * 1000.0
        return (
            u_best_jac,
            best_cost_jac,
            t_calc_ms_jac,
            u_best_jac.unsqueeze(0),
            torch.tensor([best_cost_jac], device=self.device),
            y_pred_final_mm_jac,
        )

    # 🎯 Multi-arranque: al llegar a un target NUEVO, o si el CEM está
    # genuinamente estancado (misma condición que dispara el ensanche de
    # std arriba), evaluar unas pocas "anclas" (estado actual, home, varios
    # puntos repartidos por el rango entrenado, y el Jacobiano local vía
    # autograd) para elegir la región más prometedora ANTES de iterar el
    # CEM local. En estancamiento, el Jacobiano se recalcula en la posición
    # ACTUAL (más cerca del target que al principio, aproximación local más
    # precisa ahí) -- intenta escapar de una meseta en vez de solo ensanchar
    # ruido alrededor del mismo punto atascado. Reemplaza la necesidad de
    # sembrar a mano una combinación conocida-buena: si existe una rama
    # cinemática redundante mejor, alguna ancla cae cerca y el CEM converge
    # hacia ahí en los ciclos siguientes -- el comando real sigue acotado
    # por max_step_ticks más abajo, no hay teletransporte.
    intentar_multiarranque = es_target_nuevo or (estancado and not error_chico)
    if intentar_multiarranque:
        n_anclas_extra = 6
        anclas_extra_scaled = torch.rand(n_anclas_extra, 4, device=self.device) * 2.0 - 1.0
        anclas_extra_ticks = self.candidate_generator.desescalar_u_ticks(anclas_extra_scaled)
        anclas_lista = [
            mean_ticks_inicial.unsqueeze(0),
            self.candidate_generator.homes.unsqueeze(0),
            anclas_extra_ticks,
        ]
        # 🧭 Ancla de cinemática inversa vía Jacobiano (autograd) + pseudo-
        # inversa amortiguada: en vez de anclas puramente aleatorias/fijas,
        # una aproximación informada por la sensibilidad real del modelo
        # en el estado actual (ver discusión con el profesor sobre J⁺).
        # En modo_control='cem' se omite a propósito -- ese modo evalúa el
        # CEM SIN ninguna ayuda del Jacobiano, ni siquiera como ancla.
        if self.modo_control != 'cem':
          ancla_jacobiano = self._ancla_jacobiano_ik(
              x_hist_input, mean_ticks_inicial, target_pos_mm
          )
        else:
          ancla_jacobiano = None
        if ancla_jacobiano is not None:
          anclas_lista.append(ancla_jacobiano.unsqueeze(0))
        if self.anclas_fijas_ticks is not None:
          anclas_lista.append(self.anclas_fijas_ticks)
        anclas_ticks = torch.cat(anclas_lista, dim=0)
        anclas_scaled = self.candidate_generator.escalar_u_ticks(anclas_ticks)

        x_hist_anclas = x_hist_input.repeat(anclas_ticks.shape[0], 1, 1)
        y_preds_anclas = self.predictor(x_hist_anclas, anclas_scaled)
        y_preds_anclas_mm = self._desescalar_y_mm(y_preds_anclas[:, :, :3])
        costs_anclas = self.cost_fn.compute_cost(
            y_preds_anclas_mm, y_ref_mm, anclas_scaled, u_current_scaled
        )
        # Mismo término de límites que se aplica en el loop normal, para que
        # el puntaje de las anclas sea comparable (evita elegir una ancla
        # que solo se ve bien porque ignora la penalización de límites).
        min_ticks_anclas = self.candidate_generator.homes - self.candidate_generator.ranges / 2.0
        max_ticks_anclas = self.candidate_generator.homes + self.candidate_generator.ranges / 2.0
        margen_anclas = torch.minimum(
            anclas_ticks - min_ticks_anclas.unsqueeze(0),
            max_ticks_anclas.unsqueeze(0) - anclas_ticks,
        )
        costs_anclas = costs_anclas + self.w_limite * torch.sum(
            torch.relu(self.margen_confort_ticks - margen_anclas), dim=1
        )

        mejor_ancla_idx = int(torch.argmin(costs_anclas).item())
        mean_ticks = anclas_ticks[mejor_ancla_idx].clone()

        # 📣 Log visible: sin esto no hay forma de saber desde afuera si el
        # multi-arranque se disparó por target nuevo o por estancamiento, ni
        # qué ancla ganó -- necesario para diagnosticar si está actuando.
        n_anclas_fijas = 0 if self.anclas_fijas_ticks is None else self.anclas_fijas_ticks.shape[0]
        etiquetas_anclas = (
            ['estado_actual', 'home']
            + [f'aleatoria_{i}' for i in range(n_anclas_extra)]
            + (['jacobiano_ik'] if ancla_jacobiano is not None else [])
            + [f'fija_{i}' for i in range(n_anclas_fijas)]
        )
        motivo = 'target nuevo' if es_target_nuevo else 'estancamiento'
        print(f'🧭 Multi-arranque ({motivo}): ancla ganadora = '
              f'"{etiquetas_anclas[mejor_ancla_idx]}" (costo={costs_anclas[mejor_ancla_idx].item():.2f})')
    else:
        mean_ticks = mean_ticks_inicial

    if self.cem_std_ticks is None:
        std_ticks = torch.full((4,), std_floor, device=self.device)
    else:
        std_ticks = torch.clamp(self.cem_std_ticks, min=std_floor)

    n_por_iter = max(8, self.num_samples // self.n_cem_iters)
    top_u_scaled, top_costs = None, None

    for it in range(self.n_cem_iters):
        cands_ticks = self.candidate_generator.generar_candidatos_cem(
            mean_ticks, std_ticks, num_samples=n_por_iter, frac_global=frac_global
        )
        cands_scaled = self.candidate_generator.escalar_u_ticks(cands_ticks)

        x_hist_batch = x_hist_input.repeat(n_por_iter, 1, 1)
        y_preds_scaled = self.predictor(x_hist_batch, cands_scaled)
        y_preds_mm = self._desescalar_y_mm(y_preds_scaled[:, :, :3])

        costs = self.cost_fn.compute_cost(
            y_preds_mm, y_ref_mm, cands_scaled, u_current_scaled
        )

        # 🧲 Persistencia direccional: penalizar candidatos que revierten la
        # dirección del último movimiento neto aplicado (ver __init__). Solo
        # castiga la componente que se OPONE (alineamiento negativo); seguir
        # en la misma dirección o moverse en una dirección "neutral" no se
        # penaliza. Sin esto, el CEM puede oscilar de un lado a otro sin
        # comprometerse, y en un sistema con holgura de cable eso puede no
        # traducirse en movimiento físico real.
        if self.last_delta_ticks is not None:
          last_norm = torch.norm(self.last_delta_ticks)
          if last_norm > 1e-3:
            last_dir = self.last_delta_ticks / last_norm
            cand_delta = cands_ticks - mean_ticks_inicial
            cand_delta_norm = torch.norm(cand_delta, dim=1)
            cand_delta_safe = cand_delta_norm.clamp(min=1e-6)
            alineamiento = (cand_delta / cand_delta_safe.unsqueeze(1)) @ last_dir
            costs = costs + self.w_persistencia * torch.clamp(-alineamiento, min=0.0) * cand_delta_norm

        # 🚧 Penalización por acercarse a los límites físicos (resolución de
        # redundancia): margen = distancia de cada motor al límite MÁS
        # cercano (bajo o alto). Si algún motor entra en la zona de confort
        # (margen < margen_confort_ticks), se penaliza proporcional a cuánto
        # se metió -- 0 si está a margen_confort_ticks o más de ambos límites.
        min_ticks_t = self.candidate_generator.homes - self.candidate_generator.ranges / 2.0
        max_ticks_t = self.candidate_generator.homes + self.candidate_generator.ranges / 2.0
        margen = torch.minimum(
            cands_ticks - min_ticks_t.unsqueeze(0),
            max_ticks_t.unsqueeze(0) - cands_ticks,
        )
        costs = costs + self.w_limite * torch.sum(
            torch.relu(self.margen_confort_ticks - margen), dim=1
        )

        # Élite: el `elite_frac` con menor coste actualiza la media/std
        k_elite = max(5, int(n_por_iter * self.elite_frac))
        elite_costs, elite_idx = torch.topk(costs, k_elite, largest=False)
        elite_ticks = cands_ticks[elite_idx]

        mean_ticks = elite_ticks.mean(dim=0)
        std_ticks = torch.clamp(elite_ticks.std(dim=0), min=std_floor)

        # 🛑 Límite de salto: no dejar que la media se aleje más de
        # max_step_ticks del punto de partida de ESTE ciclo, ni siquiera en
        # iteraciones intermedias (si no, la iteración siguiente exploraría
        # alrededor de un salto ya "adoptado" internamente).
        delta = mean_ticks - mean_ticks_inicial
        dist = torch.norm(delta)
        if dist > self.max_step_ticks:
            mean_ticks = mean_ticks_inicial + delta * (self.max_step_ticks / dist)

        if it == self.n_cem_iters - 1:
            # Top-K de la última iteración, para diagnóstico/logging
            k_top = min(3, n_por_iter)
            top_costs, top_idx = torch.topk(costs, k_top, largest=False)
            top_u_scaled = cands_scaled[top_idx]
            best_cost = elite_costs[0].item()  # aprox: coste del mejor crudo, no de la media

    self.cem_std_ticks = std_ticks  # persistir para el próximo ciclo (memoria)
    self.last_delta_ticks = (mean_ticks - mean_ticks_inicial).clone()  # dirección neta de este ciclo

    # --------------------------------------------------------------------------
    # C. SALIDA DE CONTROL: la MEDIA del último élite (no el mejor candidato
    # crudo) -> acción suavizada, promedio de varias soluciones "empatadas"
    # en vez de saltar de forma discreta entre ellas ciclo a ciclo.
    # --------------------------------------------------------------------------
    u_best = torch.clamp(
        self.candidate_generator.escalar_u_ticks(mean_ticks), -1.0, 1.0
    )

    # 🔭 Trayectoria que el PREDICTOR cree que va a resultar del comando
    # elegido (u_best), para poder comparar en el log lo que el modelo
    # "cree" contra lo que realmente mide el sensor unos ciclos después --
    # distingue un problema de calibración/sesgo del modelo (predice bien
    # pero el error real persiste igual) de un problema real de alcance
    # físico (el propio modelo ya predice que no va a llegar).
    y_pred_final_scaled = self.predictor(x_hist_input, u_best.unsqueeze(0))
    y_pred_final_mm = self._desescalar_y_mm(y_pred_final_scaled[0, :, :3])

    t_calc_ms = (time.perf_counter() - t_start) * 1000.0

    return u_best, best_cost, t_calc_ms, top_u_scaled, top_costs, y_pred_final_mm