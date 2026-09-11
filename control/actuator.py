"""Funciones de movimiento de alto nivel (antagonista, etc.)."""

def antagonistic_move(controller, ids, positions_a, positions_b, speed=None):
    """Ejemplo simple: mueve dos conjuntos de posiciones alternativamente."""
    # positions_a and positions_b are lists of same length as ids
    controller.move(ids, positions_a, speed=speed)
    controller.move(ids, positions_b, speed=speed)
