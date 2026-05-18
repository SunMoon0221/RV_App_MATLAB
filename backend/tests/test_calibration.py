import numpy as np

from app.services.calibration import compute_ratios, pixels_to_coordinates


def test_calibration_ratios():
    xr, yr = compute_ratios((0, 0, 100, 0), (0, 0, 0, 50), 1.0, 30.0)
    assert abs(xr - 0.01) < 1e-9
    assert abs(yr - 0.6) < 1e-9


def test_coordinate_conversion():
    x_px = np.array([0, 100], dtype=float)
    y_px = np.array([50.0, 0.0])
    t, p = pixels_to_coordinates(x_px, y_px, 0, 50, 0.01, 0.6)
    assert abs(t[1] - 1.0) < 1e-9
    assert abs(p[1] - 30.0) < 1e-9
