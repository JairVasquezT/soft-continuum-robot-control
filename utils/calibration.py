"""Calibración y utilidades varias."""

def calibrate_zero(controller, ids):
    # Ejemplo: llevar a home
    for mid in ids:
        controller.set_goal_position(mid, 2048)
