"""Axis calibration: pixel guides to time (s) and pressure (mmHg)."""

from __future__ import annotations

import numpy as np

from app.session_store import CalibrationParams


def compute_ratios(
    horizontal_line: tuple[float, float, float, float],
    vertical_line: tuple[float, float, float, float],
    time_span_seconds: float,
    pressure_span_mmhg: float,
) -> tuple[float, float]:
    """
    x_ratio = time_span_seconds / horizontal_pixel_distance
    y_ratio = pressure_span_mmHg / vertical_pixel_distance
    """
    hx1, hy1, hx2, hy2 = horizontal_line
    vx1, vy1, vx2, vy2 = vertical_line
    h_dist = float(np.hypot(hx2 - hx1, hy2 - hy1))
    v_dist = float(np.hypot(vx2 - vx1, vy2 - vy1))
    if h_dist < 1e-6 or v_dist < 1e-6:
        raise ValueError("Calibration line distance too small")
    x_ratio = time_span_seconds / h_dist
    y_ratio = pressure_span_mmhg / v_dist
    return x_ratio, y_ratio


def pixels_to_coordinates(
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    origin_x: float,
    origin_y: float,
    x_ratio: float,
    y_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    x_coord = (x_pixel - origin_x) * x_ratio
    y_coord = (origin_y - y_pixel) * y_ratio   # image y increases downward
    """
    valid = np.isfinite(y_pixels)
    x_coord = np.full_like(x_pixels, np.nan, dtype=np.float64)
    y_coord = np.full_like(y_pixels, np.nan, dtype=np.float64)
    x_coord[valid] = (x_pixels[valid].astype(np.float64) - origin_x) * x_ratio
    y_coord[valid] = (origin_y - y_pixels[valid]) * y_ratio
    return x_coord, y_coord


def apply_calibration(
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    params: CalibrationParams,
) -> tuple[np.ndarray, np.ndarray]:
    return pixels_to_coordinates(
        x_pixels,
        y_pixels,
        params.origin_x,
        params.origin_y,
        params.x_ratio,
        params.y_ratio,
    )


def replace_line_segment(
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> np.ndarray:
    """Linear interpolation of median rows between x1 and x2."""
    out = y_pixels.copy()
    x1, y1 = p1
    x2, y2 = p2
    if x2 < x1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    x1_i = int(round(x1))
    x2_i = int(round(x2))
    if x2_i <= x1_i:
        return out
    xs = np.arange(x1_i, x2_i + 1)
    ys = y1 + (y2 - y1) * (xs - x1) / max(x2 - x1, 1e-9)
    for i, x in enumerate(xs):
        if 0 <= x < len(out):
            out[x] = ys[i]
    return out
