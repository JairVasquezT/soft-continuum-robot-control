"""Generation of trajectories/combinations with uniform sampling points.

Generates 5 points per motor (min, 3 intermediate, max) for mapping with OptiTrack.
Generates all possible combinations (Cartesian product).
"""
import random
from itertools import product
from typing import Dict, List, Iterable

import numpy as np

from continuum_robot.config import robot_config as cfg


def generate_n_points(limit_tuple, n: int) -> List[int]:
    """Generates `n` uniformly distributed points over a range (inclusive).

    For example n=3 -> [min, mid, max]
    """
    min_val, max_val = limit_tuple
    if n <= 1:
        return [int(min_val)]
    step = (max_val - min_val) / (n - 1)
    return [int(min_val + step * i) for i in range(n)]


def sample_positions_for_motor(motor_id: int, points: int = 5) -> List[int]:
    """Returns the uniformly distributed sampling `points` for the motor.

    For compatibility, `points=5` uses `cfg.SAMPLING_POINTS` if defined.
    """
    if points == 5 and motor_id in cfg.SAMPLING_POINTS:
        return cfg.SAMPLING_POINTS[motor_id]
    limits = cfg.LIMITS[motor_id]
    return generate_n_points(limits, points)


def all_combinations(motor_ids: Iterable[int] = None, points_per_motor: int = 5):
    """Generates all combinations of `points_per_motor` per motor.

    By default produces 5^4 = 625 for 4 motors. If `points_per_motor=3` it produces 3^4 = 81.
    """
    motor_ids = list(motor_ids or cfg.MOTOR_IDS)
    lists = [sample_positions_for_motor(mid, points=points_per_motor) for mid in motor_ids]
    for combo in product(*lists):
        yield dict(zip(motor_ids, combo))


# =====================================================================
# "FINAL TRAJECTORY" GENERATOR: shifted grid (genuinely new points
# relative to --points 5) + net Cartesian push filter + nearest-neighbor
# ordering + local micro-exploration per point. A mode separate
# from all_combinations()/--points 3/5, which remains unchanged.
# =====================================================================
NIVELES = [-2, -1, 0, 1, 2]  # for n_points=5 -- see _niveles_simetricos for n_points=3

OFFSET_FRACCION = 0.5              # half a step of grid offset
OFFSET_EXPLORACION_TICKS = 30      # micro-exploration range per point
MARGEN_CERCA_LIMITE_TICKS = 30     # at this distance or less from the limit, bias toward it


def generate_n_points_offset(limit_tuple, n: int, offset_fraccion: float = OFFSET_FRACCION) -> List[int]:
    """Generates `n` uniformly distributed points, shifted by `offset_fraccion`
    of a step relative to `generate_n_points`/`SAMPLING_POINTS` -- so that they are
    genuinely new points, not the ones already sampled with --points {n}.

    The step is reduced proportionally to the offset (it is not kept fixed), so
    the last point NEVER exceeds max_val -- it is clipped just in
    case (rounding errors), not because it is needed in the normal case.
    """
    min_val, max_val = limit_tuple
    step = (max_val - min_val) / (n - 1 + offset_fraccion)
    inicio = min_val + step * offset_fraccion
    return [int(np.clip(inicio + step * i, min_val, max_val)) for i in range(n)]


def generate_5_points_offset(limit_tuple, offset_fraccion: float = OFFSET_FRACCION) -> List[int]:
    """Special case of `generate_n_points_offset` with n=5 (compatibility)."""
    return generate_n_points_offset(limit_tuple, 5, offset_fraccion)


def _sampling_points_offset(n_points: int, offset_fraccion: float = OFFSET_FRACCION):
    return {
        m: generate_n_points_offset(cfg.LIMITS[m], n_points, offset_fraccion)
        for m in cfg.MOTOR_IDS
    }


def _direccion_unitaria(angulo_grados):
    r = np.radians(angulo_grados)
    return np.array([np.cos(r), np.sin(r)])


def _niveles_simetricos(n_points: int) -> List[float]:
    """Symmetric levels around 0 for `n_points` (e.g. n=5 ->
    [-2,-1,0,1,2], n=3 -> [-1,0,1]) -- used only to estimate the net
    Cartesian push of a combination, not as real ticks."""
    centro = (n_points - 1) / 2.0
    return [i - centro for i in range(n_points)]


def _magnitud_neta(idx_tuple, vectores_motor, niveles_valores: List[float]):
    niveles = {1 + i: niveles_valores[idx_tuple[i]] for i in range(4)}
    return np.linalg.norm(sum(niveles[m] * vectores_motor[m] for m in cfg.MOTOR_IDS))


def _tiene_extremo(idx_tuple, n_points: int) -> bool:
    return any(i == 0 or i == n_points - 1 for i in idx_tuple)


def _ordenar_vecino_cercano(puntos, tope_salto_por_motor, inicio):
    """Sorts `puntos` (tuples of indices 0-4) starting at `inicio`, each
    next one being the nearest (Chebyshev distance) among those at
    most `tope_salto_por_motor` levels of jump per motor away -- if none
    satisfies that, the constraint is relaxed and the nearest among
    all the remaining ones is taken (to avoid getting stuck)."""
    restantes = [p for p in puntos if p != inicio]
    actual = inicio
    secuencia = [actual]
    while restantes:
        candidatos_validos = [
            p for p in restantes
            if max(abs(actual[i] - p[i]) for i in range(4)) <= tope_salto_por_motor
        ]
        if not candidatos_validos:
            candidatos_validos = restantes
        siguiente = min(
            candidatos_validos,
            key=lambda p: max(abs(actual[i] - p[i]) for i in range(4)),
        )
        secuencia.append(siguiente)
        restantes.remove(siguiente)
        actual = siguiente
    return secuencia


def generar_puntos_principales(
    n_points: int = 5,
    offset_fraccion: float = OFFSET_FRACCION,
    tope_salto_por_motor: int = 2,
    filtrar_por_extremo_o_empuje: bool = True,
    punto_inicio: str = 'centro',
) -> List[Dict[int, int]]:
    """Generates the ordered sequence of "main points": grid of
    `n_points` points per motor shifted (`generate_n_points_offset`)
    over `cfg.LIMITS` (the usual range, NOT the widened
    exploration one), ordered by nearest neighbor (maximum jump of
    `tope_salto_por_motor` levels per motor between consecutive points).

    `n_points=5` (default, compatibility with the original 604-point
    mode): filters the 625 combinations to those that are "at the extreme of
    some motor" or with net Cartesian push >= 1.0 (to discard
    combinations that cancel each other out, such as opposite levels on
    opposing motors according to `cfg.MOTOR_ANGULOS_TRACCION`), and starts at
    the real point closest to the center.

    `n_points=3` (81 combinations): intended to be used with
    `filtrar_por_extremo_o_empuje=False` (none is discarded, all
    81 remain) and `punto_inicio='extremo'` (starts at the point FARTHEST
    from the center, instead of the closest) -- so that the path
    starts in a different order from the 604-point sequence.

    Returns a list of dicts {motor_id: ticks}.
    """
    sampling_points = _sampling_points_offset(n_points, offset_fraccion)
    vectores_motor = {
        m: _direccion_unitaria(a) for m, a in cfg.MOTOR_ANGULOS_TRACCION.items()
    }
    niveles_valores = _niveles_simetricos(n_points)

    todas = list(product(range(n_points), repeat=4))
    if filtrar_por_extremo_o_empuje:
        filtradas = [
            c for c in todas
            if _tiene_extremo(c, n_points)
            or _magnitud_neta(c, vectores_motor, niveles_valores) >= 1.0
        ]
    else:
        filtradas = todas

    centro_idx = (n_points - 1) / 2.0
    punto_centro_teorico = tuple([centro_idx] * 4)
    if punto_inicio == 'centro':
        inicio = min(
            filtradas, key=lambda p: max(abs(p[i] - punto_centro_teorico[i]) for i in range(4))
        )
    elif punto_inicio == 'extremo':
        inicio = max(
            filtradas, key=lambda p: max(abs(p[i] - punto_centro_teorico[i]) for i in range(4))
        )
    else:
        raise ValueError(f"punto_inicio debe ser 'centro' o 'extremo', no {punto_inicio!r}")

    secuencia_idx = _ordenar_vecino_cercano(filtradas, tope_salto_por_motor, inicio)

    return [
        {1 + i: sampling_points[1 + i][idx[i]] for i in range(4)}
        for idx in secuencia_idx
    ]


def candidatos_exploracion_local(
    punto_ticks: Dict[int, int],
    offset: int = OFFSET_EXPLORACION_TICKS,
    margen_limite: int = MARGEN_CERCA_LIMITE_TICKS,
    n_mixtas: int = 4,
) -> List[Dict[int, int]]:
    """Generates local micro-exploration candidates around an already
    reached main point: up to 16 "intense" combinations (all 4 motors
    move ±offset ticks, no zeros -- fewer if some motor is near
    a limit, see below) + `n_mixtas` "soft" combinations (some
    motor stays at 0, but never all 4 at once). By default
    16 + 4 = 20 candidates per point.

    The result is ordered so that there are never more than 2 intense candidates
    in a row without a soft one in between, if the point is near some
    physical limit (to avoid forcing the motor repeatedly against the stop).
    If no motor is near a limit, the order does not matter and it is
    simply shuffled.

    Clips against `cfg.LIMITS_EXPLORACION` (wider than `cfg.LIMITS`, the one
    for the main-point grid) so as not to clip the exploration to zero
    effect when the point is already near the edge of the original grid.

    Returns a list of dicts {motor_id: ticks}.
    """
    opciones_por_motor = {}
    motor_cerca_limite = False
    for m in cfg.MOTOR_IDS:
        min_m, max_m = cfg.LIMITS_EXPLORACION[m]
        pos = punto_ticks[m]
        dist_min = pos - min_m
        dist_max = max_m - pos

        if dist_min <= 0:
            # Already AT the lower limit (or beyond) -- it can only move away
            opciones_por_motor[m] = [+offset]
            motor_cerca_limite = True
        elif dist_max <= 0:
            # Already AT the upper limit -- it can only move away
            opciones_por_motor[m] = [-offset]
            motor_cerca_limite = True
        elif dist_min <= margen_limite:
            # Near the lower limit, but with margin -- explore toward it
            opciones_por_motor[m] = [-offset]
            motor_cerca_limite = True
        elif dist_max <= margen_limite:
            # Near the upper limit, but with margin -- explore toward it
            opciones_por_motor[m] = [+offset]
            motor_cerca_limite = True
        else:
            opciones_por_motor[m] = [-offset, +offset]

    combos_intensos = list(product(*[opciones_por_motor[m] for m in cfg.MOTOR_IDS]))
    combos_intensos = [c for c in combos_intensos if any(v != 0 for v in c)]

    combos_suaves = []
    intentos = 0
    while len(combos_suaves) < n_mixtas and intentos < 300:
        intentos += 1
        c = tuple(random.choice(opciones_por_motor[m] + [0]) for m in cfg.MOTOR_IDS)
        if any(v != 0 for v in c) and c not in combos_intensos and c not in combos_suaves:
            combos_suaves.append(c)

    if motor_cerca_limite:
        # Interleave: at most 2 intense ones in a row, separated by a soft one
        random.shuffle(combos_intensos)
        random.shuffle(combos_suaves)
        orden_final = []
        suaves_restantes = list(combos_suaves)
        i = 0
        while i < len(combos_intensos):
            orden_final.append(combos_intensos[i])
            if i + 1 < len(combos_intensos):
                orden_final.append(combos_intensos[i + 1])
            i += 2
            if suaves_restantes and i < len(combos_intensos):
                orden_final.append(suaves_restantes.pop(0))
        orden_final.extend(suaves_restantes)  # the leftovers, at the end
    else:
        # No motor near a limit -- order does not matter, just shuffle
        orden_final = combos_intensos + combos_suaves
        random.shuffle(orden_final)

    candidatos_ticks = []
    for delta in orden_final:
        cand = {
            m: int(np.clip(
                punto_ticks[m] + delta[i],
                cfg.LIMITS_EXPLORACION[m][0], cfg.LIMITS_EXPLORACION[m][1],
            ))
            for i, m in enumerate(cfg.MOTOR_IDS)
        }
        candidatos_ticks.append(cand)
    return candidatos_ticks


def resumen_tiempo_estimado(
    n_puntos_principales: int,
    n_candidatos_por_punto: int = 20,
    segundos_por_candidato: float = 0.25,
    segundos_por_punto_base: float = 0.5,
) -> Dict[str, float]:
    """Estimate of total time (not counting travel between main
    points, which varies). Returns a dict with the breakdown in hours."""
    candidatos_totales = n_puntos_principales * n_candidatos_por_punto
    horas_exploracion = candidatos_totales * segundos_por_candidato / 3600
    horas_puntos_base = n_puntos_principales * segundos_por_punto_base / 3600
    return {
        'puntos_principales': n_puntos_principales,
        'candidatos_totales': candidatos_totales,
        'horas_exploracion': horas_exploracion,
        'horas_puntos_base': horas_puntos_base,
        'horas_total': horas_exploracion + horas_puntos_base,
    }
