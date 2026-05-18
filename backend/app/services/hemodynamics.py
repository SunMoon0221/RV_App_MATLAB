"""Hemodynamic parameter computation from averaged single-beat waveform."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.derivatives import (
    DEFAULT_FS,
    butterworth_lowpass,
    compute_derivatives,
    find_dpdt_extrema,
    gaussian_smooth_pressure,
)
from app.services.pmax_methods import run_all_pmax_methods


def compute_hemodynamics(
    time_s: np.ndarray,
    pressure: np.ndarray,
    stroke_volume_ml: float,
    pmax_scale: float,
    selected_pmax_method: str,
    event_marker_peaks: list[int] | None = None,
    esp_idx: int | None = None,
    edp_idx: int | None = None,
    fs: float = DEFAULT_FS,
    cutoff_hz: float = 25.0,
    sigma_ms: float = 70.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Compute ESP, EDP, dP/dt extrema, tau, Ees, Ea, Ees/Ea, volumes, Eed, beta.
    """
    p_filt = butterworth_lowpass(pressure, fs, cutoff_hz)
    p_smooth = gaussian_smooth_pressure(p_filt, fs, sigma_ms)
    dpdt, d2, _ = compute_derivatives(time_s, p_smooth, fs)
    i_dpdt_max, i_dpdt_min = find_dpdt_extrema(dpdt)

    if esp_idx is None:
        esp_idx = int(np.argmax(p_smooth[i_dpdt_max:]) + i_dpdt_max)
    if edp_idx is None:
        edp_idx = int(np.argmin(p_smooth[: i_dpdt_max + 1]))

    esp = float(p_smooth[esp_idx])
    edp = float(p_smooth[edp_idx])
    dpdt_max = float(dpdt[i_dpdt_max])
    dpdt_min = float(dpdt[i_dpdt_min])

    # Tau: monoexponential decay time constant from dP/dt min region
    tau = _estimate_tau(time_s, p_smooth, i_dpdt_min)

    # Elastance Ees = (ESP - EDP) / (EDV - ESV); need volumes
    esv = stroke_volume_ml * 0.35  # placeholder split if EDV unknown
    edv = stroke_volume_ml + esv
    ees = (esp - edp) / max(edv - esv, 1e-6)
    ea = esp / max(stroke_volume_ml, 1e-6)
    ees_ea = ees / max(ea, 1e-6)

    # EDPVR: exponential Eed and beta from half-height EDP detection
    beta, eed_exp = _edpvr_fit(time_s, p_smooth, edp_idx)
    eed_linear = (esp - edp) / max(edv - esv, 1e-6)

    pmax_results = run_all_pmax_methods(
        time_s,
        p_smooth,
        pmax_scale,
        selected_pmax_method,
        fs,
        esp=esp,
        edp=edp,
        stroke_volume=stroke_volume_ml,
    )
    selected = next((m for m in pmax_results if m.get("is_selected")), pmax_results[0])
    pmax = selected.get("scaled_pmax") or selected.get("raw_pmax")

    # Enrich methods with shared hemodynamics
    for m in pmax_results:
        if m.get("success"):
            m["ees"] = ees
            m["ea"] = ea
            m["ees_ea"] = ees_ea
            m["esv"] = esv
            m["edv"] = edv

    hemo = {
        "esp": esp,
        "edp": edp,
        "pmax": pmax,
        "ees": ees,
        "ea": ea,
        "sv": stroke_volume_ml,
        "dpdt_max": dpdt_max,
        "dpdt_min": dpdt_min,
        "beta": beta,
        "tau": tau,
        "ees_ea": ees_ea,
        "eed": eed_exp,
        "eed_linear": eed_linear,
        "esv": esv,
        "edv": edv,
        "pmax_scale": pmax_scale,
        "esp_idx": esp_idx,
        "edp_idx": edp_idx,
        "dpdt_max_idx": i_dpdt_max,
        "dpdt_min_idx": i_dpdt_min,
    }
    return hemo, pmax_results


def _estimate_tau(time_s: np.ndarray, pressure: np.ndarray, idx_min: int) -> float:
    """Log-linear decay time constant after dP/dt minimum."""
    end = min(len(pressure), idx_min + int(0.3 * len(pressure)))
    seg_t = time_s[idx_min:end] - time_s[idx_min]
    seg_p = pressure[idx_min:end]
    if len(seg_p) < 5:
        return float("nan")
    p0 = seg_p[0]
    valid = seg_p > p0 * 0.2
    if np.sum(valid) < 3:
        return float("nan")
    y = np.log(seg_p[valid] / p0)
    x = seg_t[valid]
    slope = np.polyfit(x, y, 1)[0]
    return float(-1.0 / slope) if slope < 0 else float("nan")


def _edpvr_fit(
    time_s: np.ndarray, pressure: np.ndarray, edp_idx: int
) -> tuple[float, float]:
    """
    EDPVR exponential: P = a * exp(beta * V) + c — simplified pressure-domain beta.
    Uses half-height region before EDP for stiffness estimate.
    """
    half = (pressure[edp_idx] + np.max(pressure)) / 2
    crossings = np.where(np.diff(np.sign(pressure - half)))[0]
    if len(crossings) < 2:
        return float("nan"), float("nan")
    i0, i1 = crossings[0], crossings[-1]
    seg = pressure[i0:i1]
    if len(seg) < 3:
        return float("nan"), float("nan")
    t_seg = time_s[i0:i1] - time_s[i0]
    log_p = np.log(np.maximum(seg - pressure[edp_idx] + 1.0, 0.1))
    beta = float(np.polyfit(t_seg, log_p, 1)[0])
    eed = float(np.max(np.gradient(seg, t_seg[1] - t_seg[0] if len(t_seg) > 1 else 1.0)))
    return beta, eed
