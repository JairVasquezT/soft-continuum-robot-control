import time
from collections import deque

import torch
import torch.nn as nn


# ==============================================================================
# 1. MULTI-SCALE CANDIDATE GENERATOR WITH SOFT DEAD ZONE
# ==============================================================================
class CandidateGenerator:

  def __init__(self, metadata_json=None, device="cuda"):
    self.device = device

    # 1. Load the real HOMEs from the JSON
    if metadata_json and "home_motores" in metadata_json:
      homes = metadata_json["home_motores"]
      self.homes = torch.tensor(
          [homes["m1"], homes["m2"], homes["m3"], homes["m4"]], device=device
      )
    else:
      # Fallback values
      self.homes = torch.tensor(
          [1871.0, 1951.0, 1485.0, 1712.0], device=device
      )

    # 2. Dynamic ranges: TOTAL WIDTH per motor (not half-amplitude). MUST
    # match EXACTLY the scale used by dataset_filtre.py/dataset_pred_filt.py
    # for delta_real_mX/delta_meta_mX (RANGOS_MANUALES: radius 650 for
    # m1/m2, 750 for m3/m4 -> width 1300/1500) -- this is not an adjustable
    # safety margin, it is the scale the predictor was trained with. If it
    # does not match, every u_cand fed to the model is misinterpreted (a
    # different command magnitude than the one the candidate generator
    # believes it is sending). The real safety margin (where the CEM should
    # AVOID converging, even if the clamp allows it) lives in
    # margen_confort_ticks/w_limite (NeuralMPCController), not here.
    self.ranges = torch.tensor([1300.0, 1300.0, 1500.0, 1500.0], device=device)

    # 3. Dead zones in ticks
    self.deadbands = torch.tensor(
        [[-6.0, 4.0], [-13.0, 8.0], [-6.0, 5.0], [-6.0, 4.0]], device=device
    )

  def desescalar_u_ticks(self, u_scaled):
    """Converts from [-1, 1] to Absolute Ticks [HOME - Range/2, HOME + Range/2]."""
    return self.homes + u_scaled * (self.ranges / 2.0)

  def escalar_u_ticks(self, u_ticks):
    """Converts from Absolute Ticks to [-1, 1]."""
    return (u_ticks - self.homes) / (self.ranges / 2.0)

  def generar_candidatos_cem(self, mean_ticks, std_ticks, num_samples, frac_global=0.1):
    """Generates candidates for ONE CEM iteration, in absolute ticks.

    Mostly sampled from N(mean_ticks, std_ticks) (Gaussian elite), plus a
    small fraction of uniform global exploration over the whole physical range
    so as not to get trapped in a local minimum if the mean converges badly.
    """
    num_global = max(1, int(num_samples * frac_global))
    num_local = num_samples - num_global

    # A. Local sampling: Gaussian centered on the CEM mean
    delta_local_ticks = torch.randn(num_local, 4, device=self.device) * std_ticks.unsqueeze(0)
    u_local_ticks = mean_ticks.unsqueeze(0) + delta_local_ticks

    # B. Uniform global sampling (over the whole physical range [-1, 1])
    u_global_scaled = torch.rand(num_global, 4, device=self.device) * 2.0 - 1.0
    u_global_ticks = self.desescalar_u_ticks(u_global_scaled)

    # C. Merge all candidates into absolute ticks
    u_cands_ticks = torch.cat([u_local_ticks, u_global_ticks], dim=0)

    # D. Apply strict physical limits per motor
    min_ticks = self.homes - (self.ranges / 2.0)
    max_ticks = self.homes + (self.ranges / 2.0)
    u_cands_ticks = torch.clamp(
        u_cands_ticks, min_ticks.unsqueeze(0), max_ticks.unsqueeze(0)
    )

    # Candidate 0 is always the current mean (guarantees the best estimate is not lost)
    u_cands_ticks[0] = mean_ticks

    return u_cands_ticks


# ==============================================================================
# 2. CORRECTED COST FUNCTION
# ==============================================================================
class MPCCostFunction:

    def __init__(self, w_pos=1.5, w_smooth=0.1, w_terminal=5.0,
               error_axis_weights=(1.0, 1.0, 1.0), deadband_mm=0.0):
        # NOTE: w_pos/w_terminal were increased and w_smooth was reduced to prioritize the Target
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
      # Per-axis weighted squared error with dead zone in mm
      axis_weights = torch.tensor(
          self.error_axis_weights, device=y_preds_mm.device, dtype=y_preds_mm.dtype
      )
      errors = torch.abs(y_preds_mm - y_ref_mm.unsqueeze(0))
      error_outside_deadband = torch.relu(errors - self.deadband_mm)
      axis_costs = error_outside_deadband.square() * axis_weights
      err_pos = torch.sum(axis_costs, dim=-1)

      # Tracking costs
      cost_tracking = torch.mean(err_pos[:, :-1], dim=1)
      cost_terminal = err_pos[:, -1]

      # CORRECTED effort cost: evaluated against the motor's current position
      cost_effort = torch.norm(
          u_cands_scaled - u_current_scaled.unsqueeze(0), dim=-1
      )

      # Total Cost
      total_cost = (
          self.w_pos * cost_tracking
          + self.w_terminal * cost_terminal
          + self.w_smooth * cost_effort
      )

      return total_cost


# ==============================================================================
# 3. NEURAL MPC CONTROLLER
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

    # 🎛️ Control mode -- three mutually exclusive options:
    #   'cem': Cross-Entropy Method only (multi-start with random/home/fixed
    #          anchors, WITHOUT the Jacobian anchor).
    #   'jacobiano': local inverse kinematics via Jacobian (autograd) +
    #          damped pseudo-inverse only, one step per cycle -- no CEM or
    #          multi-start, to isolate how much each method contributes
    #          on its own.
    #   'hibrido' (default, the usual behavior): Jacobian as ONE more
    #          multi-start anchor of the CEM, competing on cost with the
    #          others and then refined by the CEM iterations.
    if modo_control not in ('cem', 'jacobiano', 'hibrido'):
      raise ValueError(
          f"modo_control debe ser 'cem', 'jacobiano' o 'hibrido', no {modo_control!r}"
      )
    self.modo_control = modo_control

    # 🔁 Number of local re-linearization iterations of the Jacobian
    # anchor (local Gauss-Newton): K=1 recomputes the Jacobian only once
    # at the starting point (historical behavior); K=2-3 re-evaluates the
    # predictor AND the Jacobian at the point ALREADY CORRECTED by the
    # previous step, bringing the linear approximation closer to the real
    # behavior of the predictor as the candidate moves away from the
    # starting point. See _ancla_jacobiano_ik.
    self.k_iters_jacobiano = max(1, int(k_iters_jacobiano))

    # 🧪 EXTRA fixed anchors for the multi-start (diagnostic): candidates
    # of known-good ticks (e.g. hand-confirmed seeds for hard points) that
    # compete with the other anchors on equal terms -- they are evaluated
    # with the SAME cost (including the limits barrier) and only "win" if
    # they truly have the lowest cost that cycle. Unlike
    # SEED_COMMANDS_TICKS/USAR_SEMILLAS_FIJAS (which bypasses the CEM
    # entirely), this forces nothing: it is just one more option on the menu.
    if anclas_fijas_ticks:
      self.anclas_fijas_ticks = torch.tensor(
          anclas_fijas_ticks, device=device, dtype=torch.float32
      )
    else:
      self.anclas_fijas_ticks = None

    # 📏 "Small error" threshold TIED to the real arrival tolerance, not to
    # a fixed number. It used to be a hard cutoff at 30mm regardless of
    # target_tolerance_mm -- with an 8mm tolerance, an error of 27-30mm
    # (3.4-3.75x the real tolerance) still fell into the "you're almost
    # there, don't widen the search" bucket, disabling the stagnation
    # escape exactly when it was most needed (seen in production: it stayed
    # flat at ~27-30mm without the std widening or the multi-start ever
    # triggering, because error_chico=True blocks both). With this,
    # "small error" is relative to how far the robot is from the real
    # tolerance.
    self.error_chico_umbral_mm = float(target_tolerance_mm) * float(error_chico_factor)

    # 🪤 WINDOW-based stagnation detection (not cycle-to-cycle): comparing
    # only against the previous cycle's error is sensitive to measurement
    # noise (typical of the LSTM observer/cable compliance) -- if the real
    # error is stagnant but oscillating (12->14->11->13mm), the difference
    # between consecutive cycles almost always exceeds the threshold and
    # the counter never reaches the point of triggering the search
    # widening. Instead, the error history of the last `stagnation_window`
    # cycles is stored and we measure whether the BEST error within that
    # window improved enough with respect to the start of the window.
    self.stagnation_min_improvement_mm = float(stagnation_min_improvement_mm)
    self.error_history = deque(maxlen=max(2, int(stagnation_window)))
    self.last_target_pos_mm = None  # for detecting a waypoint change

    # 🧲 Directional persistence: penalizes candidates that REVERSE the
    # direction of the last applied net movement. In a cable system with
    # slack/backlash, a command that pulls and releases without committing
    # to a direction may not translate into real physical movement (seen
    # in production: real Opti almost frozen while the command oscillated
    # strongly from side to side). This pushes the optimizer to consolidate
    # the movement instead of hesitating.
    self.w_persistencia = float(w_persistencia)
    self.last_delta_ticks = None  # net direction of the last cycle, persisted

    # 🚧 Avoid solutions that explore beyond what the REAL robot visited
    # during data collection. margen_confort_ticks=100 confirmed with the
    # real maxima/minima logged per motor: all 4 motors show the SAME gap
    # of ~100 ticks between the radius assumed by RANGOS_MANUALES (650/750,
    # used to scale the training) and the real radius actually visited
    # (~550/~650) -- the collection routine left that safety margin but the
    # preprocessing script never subtracted it. This is not a kinematic
    # redundancy barrier (that proved counterproductive, see previous
    # commit with w_limite=0) -- it is about avoiding candidates in
    # territory the model never truly saw, even if the scale theoretically
    # allows it. It softly penalizes (barrier, not
    # hard limit -- the clamp in generar_candidatos_cem already does that)
    # candidates that enter the last `margen_confort_ticks` before
    # either of the two limits
    # of each motor.
    self.w_limite = float(w_limite)
    self.margen_confort_ticks = float(margen_confort_ticks)

    # 🎯 CEM (Cross-Entropy Method): instead of an argmin over fresh i.i.d.
    # candidates every cycle, a distribution (mean/std) is maintained and
    # refined in `n_cem_iters` iterations per cycle, and its STD persists
    # across control cycles (memory) to converge smoothly instead of jumping
    # between random "ties" near the target (the cause of chattering).
    self.n_cem_iters = max(1, int(n_cem_iters))
    self.elite_frac = elite_frac
    self.std_floor_min = std_floor_min
    self.cem_std_ticks = None  # persisted across optimize() calls

    # 🛑 Per-cycle jump limit (ticks, Euclidean norm over the 4 motors):
    # the CEM mean cannot move farther than this from the cycle's starting
    # point, not even in intermediate iterations. It prevents the predictor
    # (badly extrapolated outside what it was trained on) from pushing the
    # solution to "another branch" of the inverse kinematics in one jump,
    # spiking the real error even though the PREDICTED cost looks low.
    self.max_step_ticks = float(max_step_ticks)

    self.candidate_generator = CandidateGenerator(metadata_json, device=device)
    # Reweighted to force the robot to break the resistance of the $45\text{ mm}$
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
    """Inverse-kinematics candidate via the predictor's local Jacobian
    (autograd) + damped pseudo-inverse (Damped Least Squares), with
    iterative RE-LINEARIZATION (local Gauss-Newton, `k_iters` steps, by
    default `self.k_iters_jacobiano`):

      u_0 = mean_ticks_inicial
      For step = 0 .. k_iters-1:
        x̂_step = Predictor(u_step)                       (forward, at u_step)
        J_step = d(Predictor(u))/du |_{u=u_step}           (Jacobian at u_step)
        Δq = J_step⁺ (target_pos_mm - x̂_step)              (damped pseudo-inverse)
        u_{step+1} = u_step + Δq
      returns u_{k_iters}

    With k_iters=1 it is the historical behavior (a single step). With 2-3,
    each step re-evaluates the predictor AND the Jacobian at the point ALREADY
    CORRECTED by the previous step -- the linear approximation stays
    faithful to the real predictor even if the candidate moves quite far from
    the starting point (unlike a single step, which uses a linearization
    frozen at u_0 for the whole displacement).

    NOTE: x̂_step ALWAYS comes from the predictor's forward pass (never from an
    external measurement such as current_pos_mm) -- this way the step is
    self-consistent with the model the Jacobian itself describes, at
    every re-linearization, not just the first.

    `optimize()` runs under @torch.inference_mode(), which is incompatible
    with autograd -- hence this method explicitly re-enables gradients and
    clones the input tensors (tensors created in inference_mode
    cannot be used in autograd even when re-entering the context). Each
    iteration `.detach()`es the result again before the
    next re-linearization, so as not to accumulate the autograd graph of
    the previous steps (each Jacobian is computed independently, not
    differentiating "through" the previous iterations).

    If something fails numerically (nearly singular Jacobian, etc.) at
    any step, it returns None and the caller simply does not add this
    anchor. It works on a single candidate (vector of 4 ticks) -- if in the
    future several Jacobian anchors are computed in batch, the same
    matrix operations (@, .T, torch.linalg.solve) remain
    valid by adding a batch dimension at the front.
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
          # Re-linearize exactly at u_step: this step's own gradient,
          # not inherited from the previous one (hence the .detach() above
          # and the new requires_grad_ here).
          u_step = u_step.detach().requires_grad_(True)

          x_hat_step = f(u_step)  # x̂_step = Predictor(u_step), at THIS point
          J_step = torch.autograd.functional.jacobian(f, u_step)  # [3,4] mm/tick, at u_step

          dx = (target_ik - x_hat_step).detach()
          JJt = J_step @ J_step.T + lam * torch.eye(3, device=self.device)
          dq = J_step.T @ torch.linalg.solve(JJt, dx)

          u_step = u_step.detach() + dq.detach()  # u_{step+1}, without carrying the graph

      resultado = u_step.detach()
      if not torch.isfinite(resultado).all():
        return None
      return resultado
    except Exception:
      return None

  @torch.inference_mode()
  def optimize(self, x_hist_tensor, y_ref_mm, u_current_scaled, current_pos_mm):
    t_start = time.perf_counter()

    # Ensure batch dimension in x_hist_tensor [1, seq_len, features]
    if x_hist_tensor.dim() == 2:
        x_hist_input = x_hist_tensor.unsqueeze(0)
    else:
        x_hist_input = x_hist_tensor

    # --------------------------------------------------------------------------
    # A. ADAPTIVE RADIUS COMPUTATION AND STAGNATION DETECTION
    # --------------------------------------------------------------------------
    # Real error perceived by vision/sensors: compares the current REAL
    # position (OptiTrack or observer, the same one that closes the control
    # loop, passed in by the caller) against the target, both in mm/Cartesian.
    # It used to use an extra predictor pass "guessing" the current position
    # from its own history -- that is not the real error, it is what the
    # model BELIEVES, and it can be biased with respect to what the sensor measures.
    target_pos_mm = y_ref_mm[0] if y_ref_mm.dim() > 1 else y_ref_mm
    err_actual_mm = torch.norm(target_pos_mm - current_pos_mm).item()

    # 🔄 Waypoint change: clear the stagnation window and the CEM STD
    # memory. Without this, when going from an already converged target
    # (small error) to a new distant one, the window holds a mix of
    # old (small) and new (large) errors -> false "improvement≈0" ->
    # unnecessary search widening right when reaching the new point
    # (erratic behavior). cem_std_ticks is also cleared so as not to
    # carry over an already narrow STD from the previous waypoint (although
    # the clamp with std_floor already protects this, clearing is clearer and avoids doubts).
    es_target_nuevo = (
        self.last_target_pos_mm is None
        or torch.norm(target_pos_mm - self.last_target_pos_mm).item() > 1.0
    )
    if es_target_nuevo:
        self.error_history.clear()
        self.stagnation_counter = 0
        self.cem_std_ticks = None
        self.last_delta_ticks = None  # do not carry over a "good direction" from another target
    self.last_target_pos_mm = target_pos_mm.clone()

    # Standard deviation (STD) floor according to the error magnitude: it defines
    # the minimum the CEM can explore, even if it already converged to
    # something narrower in a previous cycle (avoids being "too sure" if
    # a new distant target appears).
    # frac_global: fraction of global exploration candidates (uniform over
    # the whole physical range). Useful with large error to escape local
    # minima; with small error it is precisely the vector that injects a
    # candidate "from another branch" when the target is almost reached -> reduced to almost 0.
    if err_actual_mm > 70.0:  # Error > 7 cm
        std_floor = 45.0
        frac_global = 0.10
        error_chico = False
    elif err_actual_mm > self.error_chico_umbral_mm:  # between the real threshold and 7cm
        std_floor = 32.0
        frac_global = 0.05
        error_chico = False
    else:  # Genuinely small error (relative to the real target_tolerance_mm)
        std_floor = 23.0
        frac_global = 0.01
        error_chico = True

    # Window-based stagnation detection: this cycle's error is stored,
    # and once the window is full the error at the START of the window is
    # compared against the BEST (minimum) error seen within it. If the
    # improvement is less than stagnation_min_improvement_mm, it is
    # genuinely stagnant (not just momentary noise between two consecutive cycles).
    self.error_history.append(err_actual_mm)
    if len(self.error_history) == self.error_history.maxlen:
        mejora = self.error_history[0] - min(self.error_history)
        estancado = mejora < self.stagnation_min_improvement_mm
    else:
        estancado = False

    # NOTE: with an already small error (<3cm) we do NOT widen on stagnation. There
    # you are normally already close to the best the model can achieve;
    # widening the search does not "escape a local minimum", it pushes the
    # exploration toward farther candidates that the predictor may
    # overestimate and ends up moving the robot away from a point that was
    # already reasonably good (seen in practice: it kept circling at ~15mm and ended at 40mm).
    if estancado and not error_chico:
        self.stagnation_counter += 1
        # Absolute cap: never widen beyond the "large error" bucket
        # (90) * 1.5, so that a false trigger does not make the search so
        # wide that the chosen candidate ends up being almost pure noise.
        std_floor = min(std_floor * 1.8, 135.0)
    else:
        self.stagnation_counter = 0

    std_floor = max(std_floor, self.std_floor_min)

    # --------------------------------------------------------------------------
    # B. CEM (Cross-Entropy Method): refines mean/std in `n_cem_iters`
    # iterations. The mean ALWAYS starts at the current real state (what
    # MPC.py already reports as u_current_scaled after external EMA/hysteresis),
    # so the optimizer never "believes" it is somewhere other than the real
    # robot. The STD is inherited across control cycles (memory): instead of
    # the discrete "adaptive radius" jump used before, it converges smoothly and does not
    # collapse below std_floor.
    # --------------------------------------------------------------------------
    mean_ticks_inicial = self.candidate_generator.desescalar_u_ticks(u_current_scaled)

    # 🧭 PURE JACOBIAN MODE: self.k_iters_jacobiano steps of inverse
    # kinematics via local Jacobian (re-linearized at every step) + damped pseudo-
    # inverse, recomputed from the CURRENT position every cycle
    # -- no CEM, no multi-start. Returns early; the rest of
    # optimize() (CEM/multi-start) does not apply here.
    if self.modo_control == 'jacobiano':
        ancla_jacobiano = self._ancla_jacobiano_ik(
            x_hist_input, mean_ticks_inicial, target_pos_mm
        )
        # If the Jacobian is numerically invalid this cycle (nearly
        # singular, etc.), do not move -- safer than an arbitrary jump.
        mean_ticks_jac = (
            mean_ticks_inicial.clone() if ancla_jacobiano is None else ancla_jacobiano
        )

        # Same per-cycle jump limit as the CEM/hybrid mode: never
        # "teleport" even if the Jacobian proposes a large jump.
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

    # 🎯 Multi-start: on reaching a NEW target, or if the CEM is
    # genuinely stagnant (same condition that triggers the std
    # widening above), evaluate a few "anchors" (current state, home, several
    # points spread across the trained range, and the local Jacobian via
    # autograd) to pick the most promising region BEFORE iterating the
    # local CEM. On stagnation, the Jacobian is recomputed at the CURRENT
    # position (closer to the target than at the start, so the local
    # approximation is more accurate there) -- it tries to escape a plateau instead of just widening
    # noise around the same stuck point. It replaces the need to
    # hand-seed a known-good combination: if a better redundant
    # kinematic branch exists, some anchor lands near it and the CEM converges
    # toward it over the following cycles -- the real command is still bounded
    # by max_step_ticks below, there is no teleporting.
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
        # 🧭 Inverse-kinematics anchor via Jacobian (autograd) + damped pseudo-
        # inverse: instead of purely random/fixed anchors,
        # an approximation informed by the model's real sensitivity
        # at the current state (see discussion with the professor about J⁺).
        # In modo_control='cem' it is deliberately omitted -- that mode evaluates the
        # CEM WITHOUT any Jacobian help, not even as an anchor.
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
        # Same limits term applied in the normal loop, so that
        # the anchors' scores are comparable (avoids picking an anchor
        # that only looks good because it ignores the limits penalty).
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

        # 📣 Visible log: without this there is no way to tell from outside whether the
        # multi-start was triggered by a new target or by stagnation, nor
        # which anchor won -- needed to diagnose whether it is acting.
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

        # 🧲 Directional persistence: penalize candidates that reverse the
        # direction of the last applied net movement (see __init__). It only
        # punishes the component that OPPOSES (negative alignment); continuing
        # in the same direction or moving in a "neutral" direction is not
        # penalized. Without this, the CEM can oscillate from side to side
        # without committing, and in a system with cable slack that may not
        # translate into real physical movement.
        if self.last_delta_ticks is not None:
          last_norm = torch.norm(self.last_delta_ticks)
          if last_norm > 1e-3:
            last_dir = self.last_delta_ticks / last_norm
            cand_delta = cands_ticks - mean_ticks_inicial
            cand_delta_norm = torch.norm(cand_delta, dim=1)
            cand_delta_safe = cand_delta_norm.clamp(min=1e-6)
            alineamiento = (cand_delta / cand_delta_safe.unsqueeze(1)) @ last_dir
            costs = costs + self.w_persistencia * torch.clamp(-alineamiento, min=0.0) * cand_delta_norm

        # 🚧 Penalty for approaching the physical limits (redundancy resolution):
        # margin = distance of each motor to the NEAREST limit (low or high). If any
        # motor enters the comfort zone
        # (margin < margen_confort_ticks), it is penalized proportionally to how far it
        # went in -- 0 if it is margen_confort_ticks or more away from both limits.
        min_ticks_t = self.candidate_generator.homes - self.candidate_generator.ranges / 2.0
        max_ticks_t = self.candidate_generator.homes + self.candidate_generator.ranges / 2.0
        margen = torch.minimum(
            cands_ticks - min_ticks_t.unsqueeze(0),
            max_ticks_t.unsqueeze(0) - cands_ticks,
        )
        costs = costs + self.w_limite * torch.sum(
            torch.relu(self.margen_confort_ticks - margen), dim=1
        )

        # Elite: the `elite_frac` with the lowest cost updates the mean/std
        k_elite = max(5, int(n_por_iter * self.elite_frac))
        elite_costs, elite_idx = torch.topk(costs, k_elite, largest=False)
        elite_ticks = cands_ticks[elite_idx]

        mean_ticks = elite_ticks.mean(dim=0)
        std_ticks = torch.clamp(elite_ticks.std(dim=0), min=std_floor)

        # 🛑 Jump limit: do not let the mean move farther than
        # max_step_ticks from THIS cycle's starting point, not even in
        # intermediate iterations (otherwise the next iteration would explore
        # around a jump already "adopted" internally).
        delta = mean_ticks - mean_ticks_inicial
        dist = torch.norm(delta)
        if dist > self.max_step_ticks:
            mean_ticks = mean_ticks_inicial + delta * (self.max_step_ticks / dist)

        if it == self.n_cem_iters - 1:
            # Top-K of the last iteration, for diagnostics/logging
            k_top = min(3, n_por_iter)
            top_costs, top_idx = torch.topk(costs, k_top, largest=False)
            top_u_scaled = cands_scaled[top_idx]
            best_cost = elite_costs[0].item()  # approx: cost of the raw best, not of the mean

    self.cem_std_ticks = std_ticks  # persist for the next cycle (memory)
    self.last_delta_ticks = (mean_ticks - mean_ticks_inicial).clone()  # net direction of this cycle

    # --------------------------------------------------------------------------
    # C. CONTROL OUTPUT: the MEAN of the last elite (not the raw best
    # candidate) -> smoothed action, average of several "tied" solutions
    # instead of jumping discretely between them cycle to cycle.
    # --------------------------------------------------------------------------
    u_best = torch.clamp(
        self.candidate_generator.escalar_u_ticks(mean_ticks), -1.0, 1.0
    )

    # 🔭 Trajectory that the PREDICTOR believes will result from the chosen
    # command (u_best), to be able to compare in the log what the model
    # "believes" against what the sensor actually measures a few cycles later --
    # it distinguishes a model calibration/bias problem (predicts well
    # but the real error persists all the same) from a real physical-reach problem
    # (the model itself already predicts it will not get there).
    y_pred_final_scaled = self.predictor(x_hist_input, u_best.unsqueeze(0))
    y_pred_final_mm = self._desescalar_y_mm(y_pred_final_scaled[0, :, :3])

    t_calc_ms = (time.perf_counter() - t_start) * 1000.0

    return u_best, best_cost, t_calc_ms, top_u_scaled, top_costs, y_pred_final_mm