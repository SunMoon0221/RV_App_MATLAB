"""Pmax estimation methods — each returns identical schema; failures are isolated."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, signal

from app.services.derivatives import (
    DEFAULT_FS,
    butterworth_lowpass,
    compute_derivatives,
    find_dpdt_extrema,
    find_second_derivative_landmarks,
    find_third_derivative_inflection,
    gaussian_smooth_pressure,
)

METHOD_NAMES = [
    "Original Piecewise Sinusoid",
    "Brimioulle Sine",
    "RV IsoMax Sine",
    "Second Derivative Sine",
    "Kremer/Shih Tangent",
]


def empty_result(
    name: str,
    short_name: str,
    kind: str,
    success: bool,
    message: str,
    scale_factor: float = 1.0,
    is_selected: bool = False,
) -> dict[str, Any]:
    """Standard schema for success and failure."""
    return {
        "name": name,
        "short_name": short_name,
        "kind": kind,
        "success": success,
        "message": message,
        "raw_pmax": None,
        "scaled_pmax": None,
        "scale_factor": scale_factor,
        "fit_r2": None,
        "pt1": None,
        "pt2": None,
        "pt3": None,
        "pt4": None,
        "t_fit": None,
        "p_fit_raw": None,
        "p_fit_scaled": None,
        "t_used1": None,
        "p_used1": None,
        "t_used2": None,
        "p_used2": None,
        "peak_time": None,
        "t_cross": None,
        "p_cross": None,
        "pad_mean": None,
        "ees": None,
        "ea": None,
        "ees_ea": None,
        "esv": None,
        "edv": None,
        "is_selected": is_selected,
    }


def _pt(t: float, p: float) -> dict[str, float]:
    return {"t": float(t), "p": float(p)}


def run_all_pmax_methods(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float = 1.0,
    selected_method: str = "Original Piecewise Sinusoid",
    fs: float = DEFAULT_FS,
    esp: float | None = None,
    edp: float | None = None,
    stroke_volume: float | None = None,
) -> list[dict[str, Any]]:
    """Run every method; one failure must not affect others."""
    runners: list[tuple[str, str, str, Any]] = [
        ("Original Piecewise Sinusoid", "Orig", "piecewise", original_piecewise_sinusoid),
        ("Brimioulle Sine", "Brim", "sine", brimioulle_sine),
        ("RV IsoMax Sine", "IsoMax", "sine", rv_isomax_sine),
        ("Second Derivative Sine", "2ndSin", "sine", second_derivative_sine),
        ("Kremer/Shih Tangent", "K/S", "tangent", kremer_shih_tangent),
    ]
    results: list[dict[str, Any]] = []
    for name, short, kind, runner in runners:
        try:
            r = runner(time_s, pressure, scale_factor, fs, esp, edp, stroke_volume)
        except Exception as exc:  # noqa: BLE001 — research: isolate method failures
            r = empty_result(name, short, kind, False, str(exc), scale_factor)
        r["is_selected"] = r["name"] == selected_method
        if r["success"] and scale_factor != 1.0 and r.get("raw_pmax") is not None:
            r["scaled_pmax"] = r["raw_pmax"] * scale_factor
            r["scale_factor"] = scale_factor
        results.append(r)
    return results


def _prep_signals(
    time_s: np.ndarray, pressure: np.ndarray, fs: float, cutoff: float = 25.0, sigma_ms: float = 70.0
):
    p_filt = butterworth_lowpass(pressure, fs, cutoff)
    p_smooth = gaussian_smooth_pressure(p_filt, fs, sigma_ms)
    dpdt, d2, _ = compute_derivatives(time_s, p_smooth, fs)
    d3 = np.gradient(d2, 1.0 / fs)
    i_max, i_min = find_dpdt_extrema(dpdt)
    return p_filt, p_smooth, dpdt, d2, d3, i_max, i_min


# --- Brimioulle core: P(t) = a + b*(1 + sin(c*t + d)), Pmax = a + 2b ---


def _brimioulle_model(t: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a + b * (1.0 + np.sin(c * t + d))


def _fit_brimioulle_sine(
    t_win: np.ndarray,
    p_win: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Nonlinear least squares: P(t) = a + b*(1+sin(c*t+d)).
    Pmax = a + 2b. Returns (a,b,c,d, pmax, r2).
    """
    if len(t_win) < 5:
        raise ValueError("Insufficient points in fitting window")
    t0 = t_win - t_win[0]
    p0 = float(np.mean(p_win))
    amp = max(float(np.ptp(p_win)) / 4, 0.5)
    p0_guess = [p0, amp, np.pi / max(t0[-1], 0.1), 0.0]

    def residual(params):
        a, b, c, d = params
        return _brimioulle_model(t0, a, b, c, d) - p_win

    try:
        res = optimize.least_squares(residual, p0_guess, method="lm", max_nfev=5000)
    except Exception:
        res = optimize.least_squares(residual, p0_guess, method="trf", max_nfev=5000)
    a, b, c, d = res.x
    pred = _brimioulle_model(t0, a, b, c, d)
    ss_res = float(np.sum((p_win - pred) ** 2))
    ss_tot = float(np.sum((p_win - np.mean(p_win)) ** 2)) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    pmax = a + 2.0 * b
    return a, b, c, d, float(pmax), float(r2)


def _sine_method_result(
    name: str,
    short: str,
    time_s: np.ndarray,
    p_smooth: np.ndarray,
    win1: tuple[int, int],
    win2: tuple[int, int],
    scale_factor: float,
    pt_labels: tuple[int, int, int, int],
) -> dict[str, Any]:
    i1a, i1b = win1
    i2a, i2b = win2
    t1, p1 = time_s[i1a : i1b + 1], p_smooth[i1a : i1b + 1]
    t2, p2 = time_s[i2a : i2b + 1], p_smooth[i2a : i2b + 1]
    _, _, _, _, pmax1, r2_1 = _fit_brimioulle_sine(t1, p1)
    _, _, _, _, pmax2, r2_2 = _fit_brimioulle_sine(t2, p2)
    raw_pmax = float(max(pmax1, pmax2))
    r2 = float(min(r2_1, r2_2))
    t_fit1 = list(map(float, t1))
    t_fit2 = list(map(float, t2))
    a1, b1, c1, d1, _, _ = _fit_brimioulle_sine(t1, p1)
    a2, b2, c2, d2, _, _ = _fit_brimioulle_sine(t2, p2)
    p_fit1 = [float(v) for v in _brimioulle_model(t1 - t1[0], a1, b1, c1, d1)]
    p_fit2 = [float(v) for v in _brimioulle_model(t2 - t2[0], a2, b2, c2, d2)]
    idx = pt_labels
    return {
        **empty_result(name, short, "sine", True, "ok", scale_factor),
        "raw_pmax": raw_pmax,
        "scaled_pmax": raw_pmax * scale_factor,
        "fit_r2": r2,
        "pt1": _pt(time_s[idx[0]], p_smooth[idx[0]]),
        "pt2": _pt(time_s[idx[1]], p_smooth[idx[1]]),
        "pt3": _pt(time_s[idx[2]], p_smooth[idx[2]]),
        "pt4": _pt(time_s[idx[3]], p_smooth[idx[3]]),
        "t_fit": t_fit1 + t_fit2,
        "p_fit_raw": p_fit1 + p_fit2,
        "p_fit_scaled": [v * scale_factor for v in p_fit1 + p_fit2],
        "t_used1": t_fit1,
        "p_used1": [float(v) for v in p1],
        "t_used2": t_fit2,
        "p_used2": [float(v) for v in p2],
        "peak_time": float(time_s[int(np.argmax(p_smooth))]),
    }


def brimioulle_sine(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp=None,
    edp=None,
    sv=None,
) -> dict[str, Any]:
    """Brimioulle-style sine on isovolumic windows from 2nd-derivative landmarks."""
    p_filt, p_smooth, dpdt, d2, d3, i_max, i_min = _prep_signals(time_s, pressure, fs)
    landmarks = find_second_derivative_landmarks(d2, i_max)
    if len(landmarks) < 2:
        return empty_result("Brimioulle Sine", "Brim", "sine", False, "Insufficient 2nd-derivative landmarks")
    left_2d = landmarks[0]
    right_2d = landmarks[-1]
    win1 = (left_2d, i_max)
    win2 = (i_min, right_2d)
    pts = (left_2d, i_max, i_min, right_2d)
    return _sine_method_result(
        "Brimioulle Sine", "Brim", time_s, p_smooth, win1, win2, scale_factor, pts
    )


def rv_isomax_sine(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp=None,
    edp=None,
    sv=None,
) -> dict[str, Any]:
    """RV IsoMax: 3rd-derivative inflection before max dP/dt through 2nd-deriv after min dP/dt."""
    p_filt, p_smooth, dpdt, d2, d3, i_max, i_min = _prep_signals(time_s, pressure, fs)
    inflect = find_third_derivative_inflection(d3, i_max)
    pt1 = inflect if inflect is not None else max(0, i_max - 20)
    landmarks = find_second_derivative_landmarks(d2, i_min)
    pt4 = max(landmarks) if landmarks else min(len(time_s) - 1, i_min + 40)
    win1 = (pt1, i_max)
    win2 = (i_min, pt4)
    pts = (pt1, i_max, i_min, pt4)
    return _sine_method_result(
        "RV IsoMax Sine", "IsoMax", time_s, p_smooth, win1, win2, scale_factor, pts
    )


def second_derivative_sine(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp=None,
    edp=None,
    sv=None,
) -> dict[str, Any]:
    """Windows from max/min 2nd-derivative landmarks around dP/dt extrema."""
    p_filt, p_smooth, dpdt, d2, d3, i_max, i_min = _prep_signals(time_s, pressure, fs)
    lm_max = find_second_derivative_landmarks(d2, i_max)
    lm_min = find_second_derivative_landmarks(d2, i_min)
    if len(lm_max) < 2 or len(lm_min) < 2:
        return empty_result(
            "Second Derivative Sine", "2ndSin", "sine", False, "Insufficient landmarks"
        )
    win1 = (min(lm_max), max(lm_max))
    win2 = (min(lm_min), max(lm_min))
    pts = (win1[0], i_max, i_min, win2[1])
    return _sine_method_result(
        "Second Derivative Sine", "2ndSin", time_s, p_smooth, win1, win2, scale_factor, pts
    )


def original_piecewise_sinusoid(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp: float | None = None,
    edp: float | None = None,
    sv: float | None = None,
) -> dict[str, Any]:
    """
    Default downstream method: piecewise sinusoid fit between dP/dt max and min.
    Ported conceptually from MATLAB SingleBeatAnalysis piecewise isovolumic model.
    """
    p_filt, p_smooth, dpdt, d2, d3, i_max, i_min = _prep_signals(time_s, pressure, fs)
    if i_min <= i_max:
        return empty_result(
            "Original Piecewise Sinusoid", "Orig", "piecewise", False, "dP/dt min before max"
        )
    t_seg = time_s[i_max:i_min + 1]
    p_seg = p_smooth[i_max:i_min + 1]
    if len(t_seg) < 8:
        return empty_result(
            "Original Piecewise Sinusoid", "Orig", "piecewise", False, "Segment too short"
        )
    # Half-sine rise model: P = P_base + A*sin(pi/2 * phase)
    phase = np.linspace(0, np.pi / 2, len(t_seg))
    p_base = float(p_smooth[i_max])
    amp = float(np.max(p_seg) - p_base)
    p_model = p_base + amp * np.sin(phase)
    ss_res = float(np.sum((p_seg - p_model) ** 2))
    ss_tot = float(np.sum((p_seg - np.mean(p_seg)) ** 2)) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    raw_pmax = float(p_base + amp)
    t_fit = [float(v) for v in t_seg]
    p_fit = [float(v) for v in p_model]
    return {
        **empty_result("Original Piecewise Sinusoid", "Orig", "piecewise", True, "ok", scale_factor),
        "raw_pmax": raw_pmax,
        "scaled_pmax": raw_pmax * scale_factor,
        "fit_r2": r2,
        "pt1": _pt(time_s[i_max], p_smooth[i_max]),
        "pt2": _pt(time_s[i_min], p_smooth[i_min]),
        "t_fit": t_fit,
        "p_fit_raw": p_fit,
        "p_fit_scaled": [v * scale_factor for v in p_fit],
        "t_used1": t_fit,
        "p_used1": [float(v) for v in p_seg],
        "peak_time": float(time_s[int(np.argmax(p_smooth))]),
    }


def kremer_shih_tangent(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp=None,
    edp=None,
    sv=None,
) -> dict[str, Any]:
    """
    Kremer/Shih tangent construction from dP/dt max and min.
    Tangent intersection extrapolates isovolumic Pmax.
    """
    p_filt, p_smooth, dpdt, d2, d3, i_max, i_min = _prep_signals(time_s, pressure, fs)
    t1, p1 = time_s[i_max], p_smooth[i_max]
    t2, p2 = time_s[i_min], p_smooth[i_min]
    m1 = float(dpdt[i_max])
    m2 = float(dpdt[i_min])
    if abs(m1 - m2) < 1e-9:
        return empty_result("Kremer/Shih Tangent", "K/S", "tangent", False, "Parallel tangents")
    # Line: p = p1 + m1*(t-t1), p = p2 + m2*(t-t2)
    t_cross = (p2 - p1 + m1 * t1 - m2 * t2) / (m1 - m2)
    p_cross = p1 + m1 * (t_cross - t1)
    raw_pmax = float(p_cross)
    pad = float(np.mean(p_smooth[max(0, i_max - 10) : i_max + 1]))
    return {
        **empty_result("Kremer/Shih Tangent", "K/S", "tangent", True, "ok", scale_factor),
        "raw_pmax": raw_pmax,
        "scaled_pmax": raw_pmax * scale_factor,
        "pt1": _pt(t1, p1),
        "pt2": _pt(t2, p2),
        "t_cross": float(t_cross),
        "p_cross": float(p_cross),
        "pad_mean": pad,
        "peak_time": float(time_s[int(np.argmax(p_smooth))]),
        "t_used1": [float(t1)],
        "p_used1": [float(p1)],
        "t_used2": [float(t2)],
        "p_used2": [float(p2)],
    }
