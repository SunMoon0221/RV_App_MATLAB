"""Pressure derivatives, filtering, and landmark detection."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


DEFAULT_FS = 500.0


def butterworth_lowpass(
    pressure: np.ndarray, fs: float = DEFAULT_FS, cutoff_hz: float = 25.0, order: int = 4
) -> np.ndarray:
    """Butterworth low-pass filter for pressure trace."""
    nyq = fs / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(order, wn, btype="low")
    return signal.filtfilt(b, a, pressure)


def gaussian_smooth_pressure(
    pressure: np.ndarray, fs: float = DEFAULT_FS, sigma_ms: float = 70.0
) -> np.ndarray:
    sigma_samples = sigma_ms * 1e-3 * fs
    return gaussian_filter1d(pressure, sigma=sigma_samples)


def compute_derivatives(
    time_s: np.ndarray, pressure: np.ndarray, fs: float = DEFAULT_FS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dP/dt and d²P/dt² via gradient on uniform time grid."""
    dt = 1.0 / fs if len(time_s) < 2 else np.median(np.diff(time_s))
    dpdt = np.gradient(pressure, dt)
    d2pdt2 = np.gradient(dpdt, dt)
    return dpdt, d2pdt2, np.array([dt])


def find_dpdt_extrema(dpdt: np.ndarray) -> tuple[int, int]:
    """Indices of max and min dP/dt."""
    return int(np.argmax(dpdt)), int(np.argmin(dpdt))


def find_second_derivative_landmarks(d2pdt2: np.ndarray, center: int, window: int = 80) -> list[int]:
    """Local max/min of second derivative around center."""
    lo = max(0, center - window)
    hi = min(len(d2pdt2), center + window)
    seg = d2pdt2[lo:hi]
    peaks_max, _ = signal.find_peaks(seg)
    peaks_min, _ = signal.find_peaks(-seg)
    landmarks = [lo + int(i) for i in peaks_max] + [lo + int(i) for i in peaks_min]
    landmarks.sort()
    return landmarks


def find_third_derivative_inflection(d3pdt3: np.ndarray, before_idx: int) -> int | None:
    """Third-derivative zero crossing / inflection before index."""
    if before_idx < 5:
        return None
    seg = d3pdt3[:before_idx]
    zc = np.where(np.diff(np.signbit(seg)))[0]
    return int(zc[-1]) if len(zc) else None


def build_event_marker_signal(
    pressure: np.ndarray,
    dpdt: np.ndarray,
    d2pdt2: np.ndarray,
    sigma_ms: float = 70.0,
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    """
    Combined event-marker signal for peak picking (MATLAB-style composite).
    Normalized blend of smoothed pressure derivative features.
    """
    p_s = gaussian_filter1d(pressure, sigma=sigma_ms * 1e-3 * fs)
    em = np.abs(dpdt) + 0.5 * np.abs(d2pdt2) + 0.25 * np.abs(np.gradient(d2pdt2))
    em = em / (np.max(np.abs(em)) + 1e-9)
    return em * (p_s / (np.max(p_s) + 1e-9))


def find_candidate_peaks(signal_1d: np.ndarray, min_distance: int = 20) -> np.ndarray:
    idx, _ = signal.find_peaks(signal_1d, distance=min_distance)
    return idx
