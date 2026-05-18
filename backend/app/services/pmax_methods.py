"""Pmax estimation methods — MATLAB-faithful ports with identical result schema."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize

from app.services.derivatives import (
    DEFAULT_FS,
    butterworth_lowpass,
    compute_derivatives,
    find_d2p_landmarks,
    find_dpdt_extrema,
    find_isomax_landmarks,
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


def _pt_idx(i: int) -> dict[str, int]:
    return {"index": int(i)}


def _finalize_sine_result(
    base: dict[str, Any],
    time_s: np.ndarray,
    p_smooth: np.ndarray,
    idx: tuple[int, int, int, int],
    pmax: float,
    r2: float,
    t_fit: np.ndarray,
    p_fit: np.ndarray,
    t_u1: np.ndarray,
    p_u1: np.ndarray,
    t_u2: np.ndarray,
    p_u2: np.ndarray,
    peak_time: float,
    scale_factor: float,
) -> dict[str, Any]:
    i1, i2, i3, i4 = idx
    base["raw_pmax"] = float(pmax)
    base["scaled_pmax"] = float(pmax) * scale_factor
    base["fit_r2"] = float(r2)
    base["pt1"] = _pt_idx(i1)
    base["pt2"] = _pt_idx(i2)
    base["pt3"] = _pt_idx(i3)
    base["pt4"] = _pt_idx(i4)
    base["t_fit"] = [float(v) for v in t_fit]
    base["p_fit_raw"] = [float(v) for v in p_fit]
    base["p_fit_scaled"] = [float(v) for v in p_fit]
    base["t_used1"] = [float(v) for v in t_u1]
    base["p_used1"] = [float(v) for v in p_u1]
    base["t_used2"] = [float(v) for v in t_u2]
    base["p_used2"] = [float(v) for v in p_u2]
    base["peak_time"] = float(peak_time)
    return base


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
    runners: list[tuple[str, str, str, Any]] = [
        ("Original Piecewise Sinusoid", "Orig", "legacy", original_piecewise_sinusoid),
        ("Brimioulle Sine", "Brim", "sine", brimioulle_sine),
        ("RV IsoMax Sine", "IsoMax", "sine", rv_isomax_sine),
        ("Second Derivative Sine", "2ndSin", "sine", second_derivative_sine),
        ("Kremer/Shih Tangent", "K/S", "tangent", kremer_shih_tangent),
    ]
    results: list[dict[str, Any]] = []
    for name, short, kind, runner in runners:
        try:
            r = runner(time_s, pressure, scale_factor, fs, esp, edp, stroke_volume)
        except Exception as exc:  # noqa: BLE001
            r = empty_result(name, short, kind, False, str(exc), scale_factor)
        r["is_selected"] = r["name"] == selected_method
        if r["success"] and scale_factor != 1.0 and r.get("raw_pmax") is not None:
            r["scaled_pmax"] = r["raw_pmax"] * scale_factor
            r["scale_factor"] = scale_factor
        results.append(r)
    return results


def attach_downstream_metrics(
    method_results: list[dict[str, Any]], esp: float, sv: float
) -> list[dict[str, Any]]:
    """MATLAB attachPmaxDownstreamMetrics."""
    if sv <= 0:
        return method_results
    ea = esp / sv
    for m in method_results:
        if m.get("success") and m.get("scaled_pmax") is not None:
            pmax = float(m["scaled_pmax"])
            ees = (pmax - esp) / sv
            m["ea"] = ea
            m["ees"] = ees
            if np.isfinite(ees) and ees > 0:
                m["ees_ea"] = ees / ea
                m["esv"] = esp / ees
                m["edv"] = m["esv"] + sv
    return method_results


def _prep_signals(
    time_s: np.ndarray, pressure: np.ndarray, fs: float, cutoff: float = 25.0, sigma_ms: float = 70.0
):
    p_filt = butterworth_lowpass(pressure, fs, cutoff)
    p_smooth = gaussian_smooth_pressure(p_filt, fs, sigma_ms)
    dpdt, d2, dt = compute_derivatives(time_s, p_smooth, fs)
    i_max, i_min = find_dpdt_extrema(dpdt)
    return p_filt, p_smooth, dpdt, d2, dt, i_max, i_min


# --- MATLAB fitNonlinearFunction ---


def _fit_nonlinear_function(
    time_s: np.ndarray,
    pressure: np.ndarray,
    dpdt: np.ndarray,
    dpdt_max_index: int,
    dpdt_min_index: int,
) -> tuple[float, float, float, float, float, float]:
    """Six-parameter piecewise sinusoid between dP/dt max and min."""
    t = time_s.ravel().astype(float)
    p = pressure.ravel().astype(float)
    d = dpdt.ravel().astype(float)
    n = len(t)
    if dpdt_max_index < 1 or dpdt_min_index > n or dpdt_max_index >= dpdt_min_index:
        raise ValueError("Invalid dP/dt landmark ordering")

    t_local = t - t[dpdt_max_index]
    p_local = p - p[dpdt_max_index]
    dpdt_max = float(d[dpdt_max_index])
    dpdt_min = float(d[dpdt_min_index])
    if dpdt_max <= 0 or dpdt_min >= 0:
        raise ValueError("dP/dt extrema signs invalid")

    dt_span = float(t_local[dpdt_min_index])
    if dt_span <= 0:
        raise ValueError("Non-positive time span between dP/dt extrema")

    p_min_local = float(p_local[dpdt_min_index])
    t1_guess = max(0.05, 0.45 * dt_span)
    t2_guess = max(0.05, dt_span - t1_guess)
    b1_guess = (np.pi / 2) / max(t1_guess, 1e-9)
    b2_guess = (np.pi / 2) / max(t2_guess, 1e-9)
    a1_guess = dpdt_max / max(b1_guess, 1e-9)
    a2_guess = abs(dpdt_min) / max(b2_guess, 1e-9)
    x0 = np.array([a1_guess, b1_guess, a2_guess, b2_guess, t1_guess, t2_guess])

    def equations(x):
        a1, b1, a2, b2, t1, t2 = x
        return np.array(
            [
                t1 + t2 - dt_span,
                a1 * b1 - dpdt_max,
                b1 * t1 - np.pi / 2,
                a2 * b2 - abs(dpdt_min),
                b2 * t2 - np.pi / 2,
                a1 - (a2 + p_min_local),
            ]
        )

    sol = optimize.root(equations, x0, method="hybr", options={"xtol": 1e-10})
    if not sol.success or not np.all(np.isfinite(sol.x)):
        raise RuntimeError("fitNonlinearFunction did not converge")
    params = sol.x
    if np.any(params <= 0):
        raise RuntimeError("fitNonlinearFunction returned non-positive parameters")
    return tuple(float(v) for v in params)  # type: ignore[return-value]


def original_piecewise_sinusoid(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp=None,
    edp=None,
    sv=None,
) -> dict[str, Any]:
    """MATLAB fitLegacyPiecewiseSinePrior — default downstream Pmax method."""
    name = "Original Piecewise Sinusoid"
    short = "Orig"
    try:
        _, p_smooth, dpdt, d2, dt, i_max, i_min = _prep_signals(time_s, pressure, fs)
        n = len(time_s)
        a1, b1, a2, b2, t1opt, t2opt = _fit_nonlinear_function(
            time_s, p_smooth, dpdt, i_max, i_min
        )

        t1_star = np.linspace(0, t1opt, 250)
        p1_plot = a1 * np.sin(b1 * t1_star) + p_smooth[i_max]
        t2_star = np.linspace(0, t2opt, 250)
        p2_plot = a2 * np.sin(b2 * t2_star) + p_smooth[i_min]

        t_fit_left = time_s[i_max] + t1_star
        t_fit_right = time_s[i_min] - t2_star[::-1]
        p_fit_left = p1_plot
        p_fit_right = p2_plot[::-1]
        t_fit = np.concatenate([t_fit_left, t_fit_right[1:]])
        p_fit = np.concatenate([p_fit_left, p_fit_right[1:]])
        raw_pmax = float(np.max(p_fit))

        dt_med = float(np.median(np.diff(time_s))) if n >= 2 else 1.0 / fs
        if not np.isfinite(dt_med) or dt_med <= 0:
            dt_med = 1.0 / fs
        n_pre = max(3, int(round(t1opt / dt_med)))
        n_post = max(3, int(round(t2opt / dt_med)))
        pt1 = max(0, i_max)
        pt2 = min(n - 1, i_max + n_pre)
        pt3 = max(0, i_min - n_post)
        pt4 = min(n - 1, i_min)

        idx_con = np.arange(pt1, pt2 + 1)
        idx_rel = np.arange(pt3, pt4 + 1)
        pfit_con = np.interp(time_s[idx_con], t_fit, p_fit)
        pfit_rel = np.interp(time_s[idx_rel], t_fit, p_fit)
        all_data = np.concatenate([p_smooth[idx_con], p_smooth[idx_rel]])
        all_fit = np.concatenate([pfit_con, pfit_rel])
        ss_res = float(np.sum((all_data - all_fit) ** 2))
        ss_tot = float(np.sum((all_data - np.mean(all_data)) ** 2)) + 1e-9
        r2 = max(0.0, 1.0 - ss_res / ss_tot)

        scaled = _scale_piecewise_display(p_fit, raw_pmax, raw_pmax * scale_factor)
        return {
            **empty_result(name, short, "legacy", True, "ok", scale_factor),
            "raw_pmax": raw_pmax,
            "scaled_pmax": raw_pmax * scale_factor,
            "fit_r2": r2,
            "pt1": _pt_idx(pt1),
            "pt2": _pt_idx(pt2),
            "pt3": _pt_idx(pt3),
            "pt4": _pt_idx(pt4),
            "t_fit": [float(v) for v in t_fit],
            "p_fit_raw": [float(v) for v in p_fit],
            "p_fit_scaled": [float(v) for v in scaled],
            "t_used1": [float(v) for v in time_s[idx_con]],
            "p_used1": [float(v) for v in p_smooth[idx_con]],
            "t_used2": [float(v) for v in time_s[idx_rel]],
            "p_used2": [float(v) for v in p_smooth[idx_rel]],
            "peak_time": float(time_s[int(np.argmax(p_smooth))]),
        }
    except Exception as exc:
        return empty_result(name, short, "legacy", False, str(exc), scale_factor)


def _scale_piecewise_display(
    p_fit: np.ndarray, raw_pmax: float, scaled_pmax: float
) -> np.ndarray:
    """MATLAB scaleDisplayedOriginalPiecewiseFit."""
    out = p_fit.copy().astype(float)
    good = np.isfinite(out)
    if not np.any(good) or not np.isfinite(raw_pmax) or not np.isfinite(scaled_pmax):
        return out
    idx = np.where(good)[0]
    i1, i2 = idx[0], idx[-1]
    imax = idx[int(np.argmax(out[good]))]
    p_left = out[i1]
    p_right = out[i2]
    denom_l = raw_pmax - p_left
    if np.isfinite(denom_l) and abs(denom_l) > 1e-9:
        scale_l = (scaled_pmax - p_left) / denom_l
        out[i1:imax + 1] = p_left + (out[i1:imax + 1] - p_left) * scale_l
    denom_r = raw_pmax - p_right
    if np.isfinite(denom_r) and abs(denom_r) > 1e-9:
        scale_r = (scaled_pmax - p_right) / denom_r
        out[imax:i2 + 1] = p_right + (out[imax:i2 + 1] - p_right) * scale_r
    out[imax] = scaled_pmax
    return out


# --- Brimioulle LM: P = a + exp(lb)*(1+sin(exp(lc)*t+d)), Pmax = a + 2*exp(lb) ---


def _brimioulle_model(theta: np.ndarray, tt: np.ndarray) -> np.ndarray:
    a, lb, lc, d = theta
    return a + np.exp(lb) * (1.0 + np.sin(np.exp(lc) * tt + d))


def _peak_time_local(theta: np.ndarray, t_peak_lb: float, t_peak_ub: float) -> float:
    c = np.exp(theta[2])
    d = theta[3]
    if c <= 0:
        return float("nan")
    t_mid = 0.5 * (t_peak_lb + t_peak_ub)
    k = round((c * t_mid + d - np.pi / 2) / (2 * np.pi))
    return float((np.pi / 2 - d + 2 * np.pi * k) / c)


def fit_brimioulle_sine_lm_from_windows(
    time_s: np.ndarray,
    pressure: np.ndarray,
    idx1_start: int,
    idx1_end: int,
    idx2_start: int,
    idx2_end: int,
) -> tuple[float, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB fitBrimioulleSineLMFromWindows."""
    t = time_s.ravel().astype(float)
    p = pressure.ravel().astype(float)
    n = len(t)
    idx = [int(idx1_start), int(idx1_end), int(idx2_start), int(idx2_end)]
    if any(i < 0 or i >= n for i in idx) or not (idx[0] < idx[1] < idx[2] < idx[3]):
        raise ValueError(f"Invalid window indices {idx}")

    t_u1 = t[idx[0] : idx[1] + 1]
    p_u1 = p[idx[0] : idx[1] + 1]
    t_u2 = t[idx[2] : idx[3] + 1]
    p_u2 = p[idx[2] : idx[3] + 1]
    if len(t_u1) < 5 or len(t_u2) < 5:
        raise ValueError("Fitting windows too short")
    p_used = np.concatenate([p_u1, p_u2])
    if float(np.ptp(p_used)) < 0.25:
        raise ValueError("Fitting samples too flat")

    t_ref = t[idx[0]]
    t_used_local = np.concatenate([t_u1, t_u2]) - t_ref
    p_used_arr = p_used
    p_obs_max = float(np.max(p))
    p_obs_min = float(np.min(p))
    p_used_min = float(np.min(p_used))
    p_used_max = float(np.max(p_used))
    t_span = t[idx[3]] - t_ref
    t_peak_lb = t[idx[1]] - t_ref
    t_peak_ub = t[idx[2]] - t_ref
    t_peak_mid = 0.5 * (t_peak_lb + t_peak_ub)

    pmax_guess = max(p_obs_max + max(2.0, 0.15 * float(np.ptp(p))), p_used_max + 1.0)
    a0 = max(0.0, p_used_min - 0.10 * float(np.ptp(p)))
    b0 = max((pmax_guess - a0) / 2.0, 0.5)
    c_base = 2 * np.pi / max(t_span, 1e-6)

    starts = []
    for cf in (0.55, 0.85, 1.0, 1.3, 1.8):
        for bf in (0.7, 1.0, 1.3):
            for pf in (0.4, 0.5, 0.6):
                c0 = max(c_base * cf, 1e-6)
                tpk0 = t_peak_lb + pf * (t_peak_ub - t_peak_lb)
                d0 = np.pi / 2 - c0 * tpk0
                starts.append([a0, np.log(max(b0 * bf, 1e-6)), np.log(c0), d0])

    best_theta = None
    best_ss = np.inf
    best_gated = False
    best_pmax = np.nan
    best_tpk = np.nan

    for s in starts:
        try:
            res = optimize.least_squares(
                lambda th: _brimioulle_model(th, t_used_local) - p_used_arr,
                s,
                method="lm",
                max_nfev=3000,
            )
            theta = res.x
            if not np.all(np.isfinite(theta)):
                continue
            yhat = _brimioulle_model(theta, t_used_local)
            ss = float(np.sum((p_used_arr - yhat) ** 2))
            pm = float(theta[0] + 2 * np.exp(theta[1]))
            tpk = _peak_time_local(theta, t_peak_lb, t_peak_ub)
            peak_ok = np.isfinite(tpk) and (t_peak_lb - 0.025) <= tpk <= (t_peak_ub + 0.025)
            pmax_ok = np.isfinite(pm) and pm > p_obs_max and pm < p_obs_max + max(220, 7 * float(np.ptp(p)))
            amin_ok = np.isfinite(theta[0]) and theta[0] > p_obs_min - max(70, 5 * float(np.ptp(p)))
            gated = peak_ok and pmax_ok and amin_ok
            if (gated and not best_gated) or (gated == best_gated and ss < best_ss):
                best_theta = theta
                best_ss = ss
                best_gated = gated
                best_pmax = pm
                best_tpk = tpk
        except Exception:
            continue

    if best_theta is None:
        raise RuntimeError("Brimioulle LM failed for all starts")

    theta = best_theta
    pmax = float(theta[0] + 2 * np.exp(theta[1]))
    yhat_used = _brimioulle_model(theta, t_used_local)
    ss_res = float(np.sum((p_used_arr - yhat_used) ** 2))
    ss_tot = float(np.sum((p_used_arr - np.mean(p_used_arr)) ** 2)) + 1e-9
    r2 = max(0.0, 1.0 - ss_res / ss_tot)

    c = np.exp(theta[2])
    half_period = np.pi / c
    t_peak_local = _peak_time_local(theta, t_peak_lb, t_peak_ub)
    win_lo = max(t[idx[0]] - t_ref, t_peak_local - half_period)
    win_hi = min(t[idx[3]] - t_ref, t_peak_local + half_period)
    if not (np.isfinite(win_lo) and np.isfinite(win_hi) and win_hi > win_lo):
        win_lo = t[idx[0]] - t_ref
        win_hi = t[idx[3]] - t_ref

    t_dense_local = np.linspace(win_lo, win_hi, 700)
    p_dense = _brimioulle_model(theta, t_dense_local)
    t_fit = t_ref + t_dense_local
    p_fit = p_dense
    peak_time = t_ref + t_peak_local

    return pmax, t_fit, p_fit, r2, peak_time, t_u1, p_u1, t_u2, p_u2


def brimioulle_sine(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp=None,
    edp=None,
    sv=None,
) -> dict[str, Any]:
    name, short = "Brimioulle Sine", "Brim"
    try:
        _, p_smooth, _, d2, _, i_max, i_min = _prep_signals(time_s, pressure, fs)
        pt1, _, _, pt4 = find_d2p_landmarks(d2, i_max, i_min, fs)
        idx1_start, idx1_end = pt1, i_max
        idx2_start, idx2_end = i_min, pt4
        pmax, t_fit, p_fit, r2, peak_time, t_u1, p_u1, t_u2, p_u2 = fit_brimioulle_sine_lm_from_windows(
            time_s, p_smooth, idx1_start, idx1_end, idx2_start, idx2_end
        )
        base = empty_result(name, short, "sine", True, "ok", scale_factor)
        return _finalize_sine_result(
            base,
            time_s,
            p_smooth,
            (idx1_start, idx1_end, idx2_start, idx2_end),
            pmax,
            r2,
            t_fit,
            p_fit,
            t_u1,
            p_u1,
            t_u2,
            p_u2,
            peak_time,
            scale_factor,
        )
    except Exception as exc:
        return empty_result(name, short, "sine", False, str(exc), scale_factor)


def rv_isomax_sine(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp=None,
    edp=None,
    sv=None,
) -> dict[str, Any]:
    name, short = "RV IsoMax Sine", "IsoMax"
    try:
        _, p_smooth, _, d2, _, i_max, i_min = _prep_signals(time_s, pressure, fs)
        pt1, pt2, pt3, pt4 = find_isomax_landmarks(time_s, p_smooth, i_max, i_min, fs)
        pmax, t_fit, p_fit, r2, peak_time, t_u1, p_u1, t_u2, p_u2 = fit_brimioulle_sine_lm_from_windows(
            time_s, p_smooth, pt1, pt2, pt3, pt4
        )
        base = empty_result(name, short, "sine", True, "ok", scale_factor)
        return _finalize_sine_result(
            base,
            time_s,
            p_smooth,
            (pt1, pt2, pt3, pt4),
            pmax,
            r2,
            t_fit,
            p_fit,
            t_u1,
            p_u1,
            t_u2,
            p_u2,
            peak_time,
            scale_factor,
        )
    except Exception as exc:
        return empty_result(name, short, "sine", False, str(exc), scale_factor)


def second_derivative_sine(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp=None,
    edp=None,
    sv=None,
) -> dict[str, Any]:
    name, short = "Second Derivative Sine", "2ndSin"
    try:
        _, p_smooth, _, d2, _, i_max, i_min = _prep_signals(time_s, pressure, fs)
        pt1, pt2, pt3, pt4 = find_d2p_landmarks(d2, i_max, i_min, fs)
        pmax, t_fit, p_fit, r2, peak_time, t_u1, p_u1, t_u2, p_u2 = fit_brimioulle_sine_lm_from_windows(
            time_s, p_smooth, pt1, pt2, pt3, pt4
        )
        base = empty_result(name, short, "sine", True, "ok", scale_factor)
        return _finalize_sine_result(
            base,
            time_s,
            p_smooth,
            (pt1, pt2, pt3, pt4),
            pmax,
            r2,
            t_fit,
            p_fit,
            t_u1,
            p_u1,
            t_u2,
            p_u2,
            peak_time,
            scale_factor,
        )
    except Exception as exc:
        return empty_result(name, short, "sine", False, str(exc), scale_factor)


def kremer_shih_tangent(
    time_s: np.ndarray,
    pressure: np.ndarray,
    scale_factor: float,
    fs: float,
    esp=None,
    edp=None,
    sv=None,
) -> dict[str, Any]:
    """MATLAB fitKremerShihTangentMethod."""
    name, short = "Kremer/Shih Tangent", "K/S"
    try:
        _, p_smooth, dpdt, _, _, i_max, i_min = _prep_signals(time_s, pressure, fs)
        t_dpmax = float(time_s[i_max])
        t_dpmin = float(time_s[i_min])
        p_dpmax = float(p_smooth[i_max])
        p_dpmin = float(p_smooth[i_min])
        d_pmax = float(dpdt[i_max])
        d_pmin = float(dpdt[i_min])
        ejt = t_dpmin - t_dpmax
        if ejt <= 0:
            raise ValueError("Invalid ejection time")
        if d_pmax <= 0 or d_pmin >= 0:
            raise ValueError("Invalid slopes for Kremer/Shih")
        denom = d_pmax - d_pmin
        if abs(denom) < 1e-9:
            raise ValueError("Parallel tangents")
        t1 = ((p_dpmin - p_dpmax) - d_pmin * ejt) / denom
        t_cross = t_dpmax + t1
        p_cross = p_dpmax + d_pmax * t1
        mean_pad = 0.5 * (p_dpmax + p_dpmin)
        raw_pmax = (2.0 / np.pi) * (p_cross - mean_pad) + mean_pad
        t_fit = [t_dpmax, t_cross, t_dpmin]
        p_fit = [p_dpmax, raw_pmax, p_dpmin]
        return {
            **empty_result(name, short, "tangent", True, "ok", scale_factor),
            "raw_pmax": float(raw_pmax),
            "scaled_pmax": float(raw_pmax) * scale_factor,
            "fit_r2": None,
            "pt1": _pt_idx(i_max),
            "pt2": _pt_idx(i_max),
            "pt3": _pt_idx(i_min),
            "pt4": _pt_idx(i_min),
            "t_fit": [float(v) for v in t_fit],
            "p_fit_raw": [float(v) for v in p_fit],
            "p_fit_scaled": [float(v) for v in p_fit],
            "t_used1": [t_dpmax],
            "p_used1": [p_dpmax],
            "t_used2": [t_dpmin],
            "p_used2": [p_dpmin],
            "t_cross": float(t_cross),
            "p_cross": float(p_cross),
            "pad_mean": float(mean_pad),
            "peak_time": float(t_cross),
        }
    except Exception as exc:
        return empty_result(name, short, "tangent", False, str(exc), scale_factor)
