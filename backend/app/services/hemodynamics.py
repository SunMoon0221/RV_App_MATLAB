"""Hemodynamic parameter computation — MATLAB singleBeatAnalysis parity."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize

from app.services.derivatives import (
    DEFAULT_FS,
    butterworth_lowpass,
    build_event_marker_signal,
    compute_derivatives,
    find_closest_index_half_height,
    find_dpdt_extrema,
    gaussian_smooth_pressure,
)
from app.services.pmax_methods import attach_downstream_metrics, run_all_pmax_methods


def _remap_sample_index(idx: int, time_in: np.ndarray, time_out: np.ndarray) -> int:
    """Map a sample index from pre-resample time base to post-resample (500 Hz) indices."""
    if len(time_in) < 1 or len(time_out) < 1:
        return 0
    idx = int(np.clip(idx, 0, len(time_in) - 1))
    t = float(time_in[idx])
    return int(np.clip(np.searchsorted(time_out, t), 0, len(time_out) - 1))


def compute_hemodynamics(
    time_s: np.ndarray,
    pressure: np.ndarray,
    stroke_volume_ml: float,
    pmax_scale: float,
    selected_pmax_method: str = "Original Piecewise Sinusoid",
    event_marker_peaks: list[int] | None = None,
    esp_idx: int | None = None,
    edp_idx: int | None = None,
    fs: float = DEFAULT_FS,
    cutoff_hz: float = 25.0,
    sigma_ms: float = 70.0,
    marker_sigma_ms: float = 400.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    MATLAB singleBeatAnalysis steps 0–8:
      normalize/smooth, Pmax methods, peak selection, Ees/Ea/EDPVR/tau, PV loop inputs.
    """
    if stroke_volume_ml <= 0 or not np.isfinite(stroke_volume_ml):
        raise ValueError("Invalid stroke volume")

    time_s = np.asarray(time_s, dtype=float).ravel()
    pressure = np.asarray(pressure, dtype=float).ravel()
    n = min(len(time_s), len(pressure))
    time_s, pressure = time_s[:n], pressure[:n]
    time_in = time_s.copy()

    # Uniform resample to 500 Hz (MATLAB normalizeToFixedRate)
    t_u, ia = np.unique(time_s, return_index=True)
    p_u = pressure[np.sort(ia)]
    if len(t_u) < 10:
        raise ValueError("Waveform too short")
    t_uniform = np.arange(t_u[0], t_u[-1] + 0.5 / fs, 1.0 / fs)
    if t_uniform[-1] > t_u[-1]:
        t_uniform = t_uniform[t_uniform <= t_u[-1]]
    p_uniform = np.interp(t_uniform, t_u, p_u)
    time_s, pressure = t_uniform, p_uniform
    n = len(time_s)

    if event_marker_peaks is not None:
        event_marker_peaks = [
            _remap_sample_index(int(p), time_in, time_s) for p in event_marker_peaks
        ]
    if esp_idx is not None:
        esp_idx = _remap_sample_index(int(esp_idx), time_in, time_s)
    if edp_idx is not None:
        edp_idx = _remap_sample_index(int(edp_idx), time_in, time_s)

    p_filt = butterworth_lowpass(pressure, fs, cutoff_hz)
    p_smooth = gaussian_smooth_pressure(p_filt, fs, sigma_ms)
    dpdt, d2, _ = compute_derivatives(time_s, p_smooth, fs)
    i_dpdt_max, i_dpdt_min = find_dpdt_extrema(dpdt)

    if i_dpdt_max < 2 or i_dpdt_min > n - 2 or i_dpdt_max >= i_dpdt_min:
        raise ValueError("Invalid dP/dt extrema for single-beat analysis")

    # Pmax methods (downstream default: Original Piecewise Sinusoid)
    pmax_results = run_all_pmax_methods(
        time_s,
        p_smooth,
        pmax_scale,
        selected_pmax_method,
        fs,
    )
    selected = _resolve_selected_pmax(pmax_results, selected_pmax_method, pmax_scale)
    pmax_raw = float(selected["raw_pmax"])
    pmax_scaled = float(selected["scaled_pmax"])

    # Peak selection for ESP / EDP
    event_marker = build_event_marker_signal(p_smooth, time_s, marker_sigma_ms, fs)
    peaks = event_marker_peaks
    if peaks is not None:
        peaks = sorted(int(np.clip(int(p), 0, n - 1)) for p in peaks)
        if len(peaks) < 4:
            raise ValueError("Select exactly 4 event-marker peaks")
        if esp_idx is None:
            esp_idx = peaks[2]  # MATLAB peaks(3) — third peak = ESP
        if edp_idx is None:
            edp_idx = find_closest_index_half_height(event_marker, peaks)
    if esp_idx is None:
        if peaks and len(peaks) >= 3:
            esp_idx = peaks[2]
        else:
            esp_idx = int(np.argmax(p_smooth[i_dpdt_max:]) + i_dpdt_max)
    if edp_idx is None:
        edp_idx = find_closest_index_half_height(
            event_marker, peaks if peaks else [int(np.argmax(event_marker))]
        )

    esp_idx = int(np.clip(esp_idx, 0, n - 1))
    edp_idx = int(np.clip(edp_idx, 0, n - 1))
    esp = float(p_smooth[esp_idx])
    edp = float(p_smooth[edp_idx])

    # Fitted Pmax can sit slightly below a user-picked systolic sample; keep Ees well-posed.
    if esp >= pmax_scaled:
        pmax_scaled = max(pmax_scaled, esp + 0.05)

    # Hemodynamics from selected scaled Pmax (MATLAB)
    ees = (pmax_scaled - esp) / stroke_volume_ml
    ea = esp / stroke_volume_ml
    if not np.isfinite(ees) or ees <= 0:
        raise ValueError("Computed Ees is non-positive. Check Pmax and ESP.")
    if not np.isfinite(ea) or ea <= 0:
        raise ValueError("Computed Ea is non-positive. Check ESP and SV.")

    esv = esp / ees
    edv = esv + stroke_volume_ml
    if not np.isfinite(esv) or not np.isfinite(edv) or edv <= esv:
        raise ValueError("Computed ESV/EDV are invalid.")

    ees_ea = ees / ea
    eed_linear = edp / edv

    a_edpvr, b_edpvr = fit_edpvr_anchored(esv, edv, edp)
    eed_exponential = a_edpvr * b_edpvr * np.exp(b_edpvr * edv)

    pmax_results = attach_downstream_metrics(pmax_results, esp, stroke_volume_ml)

    tau_logistic_ms = solve_numerically_tau(time_s, p_smooth, i_dpdt_min, edp_idx)

    dpdt_max = float(np.max(dpdt))
    dpdt_min = float(np.min(dpdt))

    hemo = {
        "esp": esp,
        "edp": edp,
        "pmax": pmax_scaled,
        "pmax_raw": pmax_raw,
        "ees": float(ees),
        "ea": float(ea),
        "sv": stroke_volume_ml,
        "dpdt_max": dpdt_max,
        "dpdt_min": dpdt_min,
        "beta": float(b_edpvr),
        "tau": float(tau_logistic_ms),
        "ees_ea": float(ees_ea),
        "eed": float(eed_exponential),
        "eed_linear": float(eed_linear),
        "esv": float(esv),
        "edv": float(edv),
        "edpvr_a": float(a_edpvr),
        "pmax_scale": pmax_scale,
        "esp_idx": esp_idx,
        "edp_idx": edp_idx,
        "dpdt_max_idx": i_dpdt_max,
        "dpdt_min_idx": i_dpdt_min,
        "event_marker_peak_indices": peaks or [],
        "selected_pmax_method": selected["name"],
    }
    return hemo, pmax_results


def _resolve_selected_pmax(
    method_results: list[dict[str, Any]],
    selected_name: str,
    scale_factor: float,
) -> dict[str, Any]:
    names = [m["name"] for m in method_results]
    idx = None
    for i, n in enumerate(names):
        if n.lower() == selected_name.lower() and method_results[i].get("success"):
            idx = i
            break
    if idx is None:
        for i, n in enumerate(names):
            if "original" in n.lower() and method_results[i].get("success"):
                idx = i
                break
    if idx is None:
        for i, m in enumerate(method_results):
            if m.get("success") and m.get("raw_pmax") is not None:
                idx = i
                break
    if idx is None:
        raise ValueError("Could not resolve a valid selected Pmax method")
    m = method_results[idx]
    m["is_selected"] = True
    if m.get("scaled_pmax") is None and m.get("raw_pmax") is not None:
        m["scaled_pmax"] = m["raw_pmax"] * scale_factor
    return m


def fit_edpvr_anchored(esv: float, edv: float, edp: float) -> tuple[float, float]:
    """MATLAB fitEDPVRAnchored: (exp(b*EDV)-1) = EDP*(exp(b*ESV)-1)."""
    if edp <= 0 or edv <= esv or esv <= 0:
        raise ValueError("EDPVR anchors invalid")

    def eqn(bb: float) -> float:
        return (np.exp(bb * edv) - 1.0) - edp * (np.exp(bb * esv) - 1.0)

    try:
        b = float(optimize.fsolve(eqn, 0.05)[0])
    except Exception:
        b = float(optimize.brentq(eqn, 1e-6, 0.5))
    if not np.isfinite(b) or b <= 0:
        raise ValueError("Failed to solve EDPVR beta from anchors")
    a = 1.0 / (np.exp(b * esv) - 1.0)
    return float(a), float(b)


def solve_numerically_tau(
    time_s: np.ndarray,
    pressure: np.ndarray,
    dpdt_min_index: int,
    edp_idx: int,
) -> float:
    """MATLAB solveNumerically — tau in ms."""
    d_pdt = np.gradient(pressure, time_s)
    sliced = pressure[dpdt_min_index:]
    target = pressure[edp_idx] + 2.0
    closest = int(np.argmin(np.abs(sliced - target)))
    end_index = closest + dpdt_min_index

    p0 = float(pressure[dpdt_min_index])
    p_b = float(np.min(pressure))
    p_a = 2.0 * (p0 - p_b)
    dpdt_min = float(np.min(d_pdt))
    if abs(dpdt_min) < 1e-9:
        return float("nan")
    tau_logistic = -p_a / (4.0 * dpdt_min)
    return float(1000.0 * tau_logistic)
