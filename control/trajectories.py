"""Generación de trayectorias/combinaciones con puntos de muestreo uniformes.

Genera 5 puntos por motor (mín, 3 intermedios, máx) para mapeo con OptiTrack.
Genera todas las combinaciones posibles (producto cartesiano).
"""
import random
from itertools import product
from typing import Dict, List, Iterable

import numpy as np

from continuum_robot.config import robot_config as cfg


def generate_n_points(limit_tuple, n: int) -> List[int]:
    """Genera `n` puntos uniformemente distribuidos en un rango (inclusive).

    Por ejemplo n=3 -> [min, mid, max]
    """
    min_val, max_val = limit_tuple
    if n <= 1:
        return [int(min_val)]
    step = (max_val - min_val) / (n - 1)
    return [int(min_val + step * i) for i in range(n)]


def sample_positions_for_motor(motor_id: int, points: int = 5) -> List[int]:
    """Retorna los `points` de muestreo uniformemente distribuidos para el motor.

    Por compatibilidad, `points=5` usa `cfg.SAMPLING_POINTS` si está definido.
    """
    if points == 5 and motor_id in cfg.SAMPLING_POINTS:
        return cfg.SAMPLING_POINTS[motor_id]
    limits = cfg.LIMITS[motor_id]
    return generate_n_points(limits, points)


def all_combinations(motor_ids: Iterable[int] = None, points_per_motor: int = 5):
    """Genera todas las combinaciones de `points_per_motor` por motor.

    Por defecto produce 5^4 = 625 para 4 motores. Si `points_per_motor=3` produce 3^4 = 81.
    """
    motor_ids = list(motor_ids or cfg.MOTOR_IDS)
    lists = [sample_positions_for_motor(mid, points=points_per_motor) for mid in motor_ids]
    for combo in product(*lists):
        yield dict(zip(motor_ids, combo))


# =====================================================================
# GENERADOR "TRAYECTORIA FINAL": grilla desplazada (puntos genuinamente
# nuevos respecto a --points 5) + filtro de empuje cartesiano neto + orden
# por vecino más cercano + micro-exploración local por punto. Modo aparte
# de all_combinations()/--points 3/5, que queda sin cambios.
# =====================================================================
NIVELES = [-2, -1, 0, 1, 2]  # para n_points=5 -- ver _niveles_simetricos para n_points=3

OFFSET_FRACCION = 0.5              # medio paso de desplazamiento de la grilla
OFFSET_EXPLORACION_TICKS = 30      # rango de la micro-exploración por punto
MARGEN_CERCA_LIMITE_TICKS = 30     # a esta distancia o menos del límite, sesgar hacia ahí


def generate_n_points_offset(limit_tuple, n: int, offset_fraccion: float = OFFSET_FRACCION) -> List[int]:
    """Genera `n` puntos uniformemente distribuidos, desplazados `offset_fraccion`
    de paso respecto a `generate_n_points`/`SAMPLING_POINTS` -- para que sean
    puntos genuinamente nuevos, no los ya muestreados con --points {n}.

    El paso se reduce proporcionalmente al offset (no se mantiene fijo), así
    que el último punto NUNCA excede max_val -- se recorta con clip por las
    dudas (errores de redondeo), no porque haga falta en el caso normal.
    """
    min_val, max_val = limit_tuple
    step = (max_val - min_val) / (n - 1 + offset_fraccion)
    inicio = min_val + step * offset_fraccion
    return [int(np.clip(inicio + step * i, min_val, max_val)) for i in range(n)]


def generate_5_points_offset(limit_tuple, offset_fraccion: float = OFFSET_FRACCION) -> List[int]:
    """Caso particular de `generate_n_points_offset` con n=5 (compatibilidad)."""
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
    """Niveles simétricos alrededor de 0 para `n_points` (p.ej. n=5 ->
    [-2,-1,0,1,2], n=3 -> [-1,0,1]) -- usados solo para estimar el empuje
    cartesiano neto de una combinación, no como ticks reales."""
    centro = (n_points - 1) / 2.0
    return [i - centro for i in range(n_points)]


def _magnitud_neta(idx_tuple, vectores_motor, niveles_valores: List[float]):
    niveles = {1 + i: niveles_valores[idx_tuple[i]] for i in range(4)}
    return np.linalg.norm(sum(niveles[m] * vectores_motor[m] for m in cfg.MOTOR_IDS))


def _tiene_extremo(idx_tuple, n_points: int) -> bool:
    return any(i == 0 or i == n_points - 1 for i in idx_tuple)


def _ordenar_vecino_cercano(puntos, tope_salto_por_motor, inicio):
    """Ordena `puntos` (tuplas de índices 0-4) empezando en `inicio`, cada
    siguiente es el más cercano (distancia Chebyshev) entre los que estén a
    lo sumo `tope_salto_por_motor` niveles de salto por motor -- si ninguno
    cumple eso, se relaja la restricción y se toma el más cercano entre
    todos los restantes (para no trabarse)."""
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
    """Genera la secuencia ordenada de "puntos principales": grilla de
    `n_points` puntos por motor desplazada (`generate_n_points_offset`)
    sobre `cfg.LIMITS` (el rango de siempre, NO el ampliado de
    exploración), ordenada por vecino más cercano (salto máximo de
    `tope_salto_por_motor` niveles por motor entre puntos consecutivos).

    `n_points=5` (default, compatibilidad con el modo original de 604
    puntos): filtra las 625 combinaciones a las que estén "en el extremo de
    algún motor" o con empuje cartesiano neto >= 1.0 (para descartar
    combinaciones que se cancelan entre sí, como niveles opuestos en
    motores enfrentados según `cfg.MOTOR_ANGULOS_TRACCION`), y arranca en
    el punto real más próximo al centro.

    `n_points=3` (81 combinaciones): pensado para usarse con
    `filtrar_por_extremo_o_empuje=False` (no se descarta ninguna, quedan
    las 81 completas) y `punto_inicio='extremo'` (arranca en el punto MÁS
    ALEJADO del centro, en vez de el más cercano) -- para que el recorrido
    salga en un orden distinto al de la secuencia de 604 puntos.

    Devuelve una lista de dicts {motor_id: ticks}.
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
    """Genera candidatos de micro-exploración local alrededor de un punto
    principal ya alcanzado: hasta 16 combinaciones "intensas" (los 4 motores
    se desplazan ±offset ticks, sin ceros -- menos si algún motor está cerca
    de un límite, ver abajo) + `n_mixtas` combinaciones "suaves" (algún
    motor se queda en 0, pero nunca los 4 a la vez). Por defecto
    16 + 4 = 20 candidatos por punto.

    El resultado se ordena para que nunca haya más de 2 candidatos intensos
    seguidos sin un suave de por medio, si el punto está cerca de algún
    límite físico (para no forzar el motor repetidamente contra el tope).
    Si ningún motor está cerca de un límite, el orden no importa y se
    mezcla sin más.

    Recorta contra `cfg.LIMITS_EXPLORACION` (más ancho que `cfg.LIMITS`, el
    de la grilla de puntos principales) para no clipear la exploración a 0
    efecto cuando el punto ya está cerca del borde de la grilla original.

    Devuelve una lista de dicts {motor_id: ticks}.
    """
    opciones_por_motor = {}
    motor_cerca_limite = False
    for m in cfg.MOTOR_IDS:
        min_m, max_m = cfg.LIMITS_EXPLORACION[m]
        pos = punto_ticks[m]
        dist_min = pos - min_m
        dist_max = max_m - pos

        if dist_min <= 0:
            # Ya está EN el límite inferior (o más allá) -- solo puede alejarse
            opciones_por_motor[m] = [+offset]
            motor_cerca_limite = True
        elif dist_max <= 0:
            # Ya está EN el límite superior -- solo puede alejarse
            opciones_por_motor[m] = [-offset]
            motor_cerca_limite = True
        elif dist_min <= margen_limite:
            # Cerca del límite inferior, pero con margen -- explorar hacia ahí
            opciones_por_motor[m] = [-offset]
            motor_cerca_limite = True
        elif dist_max <= margen_limite:
            # Cerca del límite superior, pero con margen -- explorar hacia ahí
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
        # Intercalar: máximo 2 intensos seguidos, separados por un suave
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
        orden_final.extend(suaves_restantes)  # los que sobraron, al final
    else:
        # Sin motor cerca de límite -- el orden no importa, mezclar y listo
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
    """Estimación de tiempo total (sin contar desplazamiento entre puntos
    principales, que varía). Devuelve un dict con el desglose en horas."""
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
