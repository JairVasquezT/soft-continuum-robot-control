"""Calibration and miscellaneous utilities."""

def calibrate_zero(controller, ids):
    # Example: take to home
    for mid in ids:
        controller.set_goal_position(mid, 2048)
