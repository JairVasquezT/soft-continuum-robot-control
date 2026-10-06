"""High-level movement functions (antagonist, etc.)."""

def antagonistic_move(controller, ids, positions_a, positions_b, speed=None):
    """Simple example: moves two sets of positions alternately."""
    # positions_a and positions_b are lists of same length as ids
    controller.move(ids, positions_a, speed=speed)
    controller.move(ids, positions_b, speed=speed)
