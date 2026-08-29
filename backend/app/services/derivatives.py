"""Pressure derivatives, filtering, landmark detection, and event-marker signals."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, median_filter


DEFAULT_FS = 500.0


def butterworth_lowpass(
    pressure: np.ndarray, fs: float = DEFAULT_FS, cutoff_hz: float = 25.0, order: int = 4
) -> np.ndarray:
    """Butterworth low-pass (zero-phase), matching MATLAB filtfilt path."""
    if len(pressure) < 3 * (2 * order + 1):
        return pressure.copy()
    nyq = fs / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(order, wn, btype="low")
    return signal.filtfilt(b, a, pressure)


def gaussian_smooth_pressure(
    pressure: np.ndarray, fs: float = DEFAULT_FS, sigma_ms: float = 70.0
) -> np.ndarray:
    sigma_samples = max(1.0, sigma_ms * 1e-3 * fs)
    return gaussian_filter1d(pressure.astype(float), sigma=sigma_samples)


def ms_to_sigma_samples(sigma_ms: float, fs: float = DEFAULT_FS) -> int:
    return max(1, int(round(sigma_ms * fs / 1000.0)))


def compute_derivatives(
    time_s: np.ndarray, pressure: np.ndarray, fs: float = DEFAULT_FS
) -> tuple[np.ndarray, np.ndarray, float]:
    """dP/dt and d²P/dt² via gradient."""
    if len(time_s) >= 2:
        dt = float(np.median(np.diff(time_s)))
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0 / fs
    else:
        dt = 1.0 / fs
    dpdt = np.gradient(pressure, dt)
    d2pdt2 = np.gradient(dpdt, dt)
    return dpdt, d2pdt2, dt


def find_dpdt_extrema(dpdt: np.ndarray) -> tuple[int, int]:
    return int(np.argmax(dpdt)), int(np.argmin(dpdt))


def refine_index_to_local_extremum(
    sig: np.ndarray, idx0: int, mode: str, radius: int = 5
) -> int:
    lo = max(0, idx0 - radius)
    hi = min(len(sig) - 1, idx0 + radius)
    seg = sig[lo : hi + 1]
    if mode == "max":
        rel = int(np.argmax(seg))
    else:
        rel = int(np.argmin(seg))
    return lo + rel


def find_d2p_landmarks(
    d2p: np.ndarray, dpdt_max_idx: int, dpdt_min_idx: int, fs: float = DEFAULT_FS
) -> tuple[int, int, int, int]:
    """
    MATLAB findD2PLandmarks:
      pt1 last positive d2 peak before dP/dt max
      pt2 first negative d2 trough after dP/dt max
      pt3 last negative d2 trough before dP/dt min
      pt4 first positive d2 peak after dP/dt min
    """
    n = len(d2p)
    if dpdt_max_idx < 2 or dpdt_min_idx > n - 2 or dpdt_max_idx >= dpdt_min_idx:
        raise ValueError("Invalid dP/dt landmark ordering for d2 landmarks")

    beat_span = dpdt_min_idx - dpdt_max_idx
    prom = max(1e-8, 0.02 * float(np.percentile(np.abs(d2p), 95)))
    min_width = 10

    left1 = max(0, dpdt_max_idx - int(round(0.60 * beat_span)))
    right1 = max(left1 + 2, dpdt_max_idx - 2)
    seg1 = d2p[left1 : right1 + 1]
    pk1, _ = signal.find_peaks(seg1, prominence=prom)
    pt1 = left1 + pk1[-1] if len(pk1) else max(0, dpdt_max_idx - int(round(0.30 * beat_span)))

    left2 = min(n - 1, dpdt_max_idx + 1)
    right2 = min(n - 1, dpdt_max_idx + int(round(0.35 * beat_span)))
    seg2 = d2p[left2 : right2 + 1]
    pk2, _ = signal.find_peaks(-seg2, prominence=prom)
    pt2 = left2 + pk2[0] if len(pk2) else min(n - 1, dpdt_max_idx + int(round(0.20 * beat_span)))

    left3 = max(0, dpdt_min_idx - int(round(0.40 * beat_span)))
    right3 = max(left3 + 2, dpdt_min_idx - 2)
    seg3 = d2p[left3 : right3 + 1]
    pk3, _ = signal.find_peaks(-seg3, prominence=prom)
    pt3 = left3 + pk3[-1] if len(pk3) else max(0, dpdt_min_idx - int(round(0.20 * beat_span)))

    left4 = min(n - 1, dpdt_min_idx + 2)
    right4 = min(n - 1, dpdt_min_idx + int(round(0.90 * beat_span)))
    seg4 = d2p[left4 : right4 + 1]
    pk4, _ = signal.find_peaks(seg4, prominence=prom)
    pt4 = left4 + pk4[0] if len(pk4) else min(n - 1, dpdt_min_idx + int(round(0.35 * beat_span)))

    pt1 = refine_index_to_local_extremum(d2p, pt1, "max", 5)
    pt2 = refine_index_to_local_extremum(d2p, pt2, "min", 5)
    pt3 = refine_index_to_local_extremum(d2p, pt3, "min", 5)
    pt4 = refine_index_to_local_extremum(d2p, pt4, "max", 5)

    pt1 = max(0, min(pt1, dpdt_max_idx - 3))
    pt2 = max(pt1 + min_width, min(pt2, dpdt_min_idx - min_width - 2))
    pt3 = max(pt2 + min_width, min(pt3, dpdt_min_idx - 3))
    pt4 = max(pt3 + min_width, pt4)
    pt4 = max(pt4, min(n - 1, dpdt_min_idx + int(round(0.22 * beat_span))))
    pt4 = min(pt4, n - 1)

    if not (pt1 < pt2 < pt3 < pt4):
        pt1 = max(0, dpdt_max_idx - int(round(0.30 * beat_span)))
        pt2 = min(n - 1, dpdt_max_idx + int(round(0.20 * beat_span)))
        pt3 = max(0, dpdt_min_idx - int(round(0.20 * beat_span)))
        pt4 = min(n - 1, dpdt_min_idx + int(round(0.35 * beat_span)))
        pt2 = max(pt2, pt1 + min_width)
        pt3 = max(pt3, pt2 + min_width)
        pt4 = max(pt4, pt3 + min_width)

    return int(pt1), int(pt2), int(pt3), int(pt4)


def find_isomax_landmarks(
    time_s: np.ndarray,
    pressure: np.ndarray,
    dpdt_max_idx: int,
    dpdt_min_idx: int,
    fs: float = DEFAULT_FS,
) -> tuple[int, int, int, int]:
    """
    RV IsoMax windows (MATLAB findIsoMaxLandmarks):
      pt1 = last max d3 before max dP/dt
      pt2 = max dP/dt
      pt3 = min dP/dt
      pt4 = first max d2 after min dP/dt
    """
    n = len(time_s)
    if dpdt_max_idx < 4 or dpdt_min_idx > n - 3 or dpdt_max_idx >= dpdt_min_idx:
        raise ValueError("Invalid dP/dt ordering for IsoMax landmarks")

    sig_deriv = max(3, int(round(0.010 * fs)))
    p_sm = gaussian_smooth_pressure(pressure, fs, sig_deriv * 1000.0 / fs)
    dt = 1.0 / fs
    d1 = np.gradient(p_sm, dt)
    d2 = np.gradient(d1, dt)
    d3 = np.gradient(d2, dt)

    beat_span = dpdt_min_idx - dpdt_max_idx
    prom_d3 = max(1e-6, 0.02 * float(np.percentile(np.abs(d3), 95)))
    prom_d2 = max(1e-6, 0.02 * float(np.percentile(np.abs(d2), 95)))

    left1 = max(0, dpdt_max_idx - int(round(0.60 * beat_span)))
    right1 = max(left1 + 2, dpdt_max_idx - 1)
    seg1 = d3[left1 : right1 + 1]
    pk1, _ = signal.find_peaks(seg1, prominence=prom_d3)
    pt1 = left1 + pk1[-1] if len(pk1) else max(0, dpdt_max_idx - int(round(0.30 * beat_span)))

    pt2 = dpdt_max_idx
    pt3 = dpdt_min_idx

    left4 = min(n - 1, dpdt_min_idx + 2)
    right4 = min(n - 1, dpdt_min_idx + int(round(0.90 * beat_span)))
    seg4 = d2[left4 : right4 + 1]
    pk4, _ = signal.find_peaks(seg4, prominence=prom_d2)
    pt4 = left4 + pk4[0] if len(pk4) else min(n - 1, dpdt_min_idx + int(round(0.35 * beat_span)))

    pt1 = refine_index_to_local_extremum(d3, pt1, "max", 5)
    pt4 = refine_index_to_local_extremum(d2, pt4, "max", 5)

    min_w = 8
    pt1 = max(0, min(pt1, pt2 - min_w))
    pt4 = max(pt3 + min_w, min(pt4, n - 1))

    if not (pt1 < pt2 <= pt3 < pt4):
        pt1 = max(0, dpdt_max_idx - int(round(0.25 * beat_span)))
        pt4 = min(n - 1, dpdt_min_idx + int(round(0.30 * beat_span)))
        if pt1 >= pt2:
            pt1 = max(0, pt2 - min_w)
        if pt4 <= pt3:
            pt4 = min(n - 1, pt3 + min_w)

    return int(pt1), int(pt2), int(pt3), int(pt4)


def compute_robust_second_derivative_marker(
    time_s: np.ndarray,
    pressure: np.ndarray,
    sigma_ms: float = 400.0,
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    """MATLAB computeRobustSecondDerivativeMarker / buildExactEventMarkerSignal core."""
    time_s = np.asarray(time_s, dtype=float).ravel()
    pressure = np.asarray(pressure, dtype=float).ravel()
    n = min(len(time_s), len(pressure))
    time_s = time_s[:n]
    pressure = pressure[:n]
    if n < 7:
        raise ValueError("Signal too short for event-marker computation")

    sigma_samp = ms_to_sigma_samples(sigma_ms, fs)
    if sigma_samp > 1:
        p_smooth = gaussian_smooth_pressure(pressure, fs, sigma_ms)
    else:
        p_smooth = pressure.copy()

    dt = float(np.median(np.diff(time_s))) if n >= 2 else 1.0 / fs
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0 / fs

    d2 = np.gradient(np.gradient(p_smooth, dt), dt)
    med_win = max(3, int(round(0.012 * fs)))
    d2_med = median_filter(d2, size=med_win, mode="nearest")
    d2sq = d2_med ** 2

    x = d2sq[np.isfinite(d2sq)]
    if x.size == 0:
        return np.zeros_like(d2sq)

    xmed = float(np.median(x))
    xmad = 1.4826 * float(np.median(np.abs(x - xmed)))
    if not np.isfinite(xmad) or xmad <= 0:
        xmad = float(np.std(x))
    if not np.isfinite(xmad) or xmad <= 0:
        xmad = max(float(np.percentile(x, 90)) / 10.0, 1e-6)

    clip_hi = max(float(np.percentile(x, 99.7)), xmed + 12.0 * xmad)
    d2sq = np.minimum(d2sq, clip_hi)

    sig_final = max(3, int(round(0.020 * fs)))
    out = gaussian_filter1d(d2sq, sigma=sig_final)
    out[~np.isfinite(out)] = 0.0
    return out


def build_event_marker_signal(
    pressure: np.ndarray,
    time_s: np.ndarray | None = None,
    sigma_ms: float = 400.0,
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    """Alias for robust squared-second-derivative marker used in peak selection."""
    if time_s is None:
        time_s = np.arange(len(pressure)) / fs
    return compute_robust_second_derivative_marker(time_s, pressure, sigma_ms, fs)


def find_closest_index_half_height(event_marker: np.ndarray, peaks: list[int]) -> int:
    """MATLAB findClosestIndex: last index at/below half of first event-marker peak."""
    if not peaks:
        raise ValueError("No event-marker peaks provided")
    peaks = sorted(int(p) for p in peaks)
    p1 = peaks[0]
    if p1 < 1 or p1 >= len(event_marker):
        raise ValueError("First peak index out of range")
    first_val = float(event_marker[p1])
    if not np.isfinite(first_val) or first_val <= 0:
        raise ValueError("First event-marker peak has invalid amplitude")
    half_val = 0.5 * first_val
    vals = event_marker[: p1 + 1]
    candidates = np.where(vals <= half_val)[0]
    if candidates.size:
        return int(candidates[-1])
    return int(np.argmin(np.abs(vals - half_val)))


def find_candidate_peaks(signal_1d: np.ndarray, min_distance: int = 20) -> np.ndarray:
    idx, _ = signal.find_peaks(signal_1d, distance=min_distance)
    return idx


# Legacy helpers (kept for compatibility)
def find_second_derivative_landmarks(d2pdt2: np.ndarray, center: int, window: int = 80) -> list[int]:
    lo = max(0, center - window)
    hi = min(len(d2pdt2), center + window)
    seg = d2pdt2[lo:hi]
    peaks_max, _ = signal.find_peaks(seg)
    peaks_min, _ = signal.find_peaks(-seg)
    landmarks = [lo + int(i) for i in peaks_max] + [lo + int(i) for i in peaks_min]
    landmarks.sort()
    return landmarks


def find_third_derivative_inflection(d3pdt3: np.ndarray, before_idx: int) -> int | None:
    if before_idx < 5:
        return None
    seg = d3pdt3[:before_idx]
    zc = np.where(np.diff(np.signbit(seg)))[0]
    return int(zc[-1]) if len(zc) else None
