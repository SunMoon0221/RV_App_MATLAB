"""Beat detection, QC, manual beat handling, and waveform averaging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


DEFAULT_FS = 500.0


@dataclass
class BeatInfo:
    start_idx: int
    end_idx: int
    peak_idx: int
    start_time: float
    end_time: float
    peak_time: float
    peak_pressure: float
    keep: bool = True
    qc_passed: bool = True
    qc_message: str = ""


def clean_trace(time_s: np.ndarray, pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove non-finite values."""
    mask = np.isfinite(time_s) & np.isfinite(pressure)
    return time_s[mask], pressure[mask]


def resample_trace(
    time_s: np.ndarray, pressure: np.ndarray, fs: float = DEFAULT_FS
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform resample to fixed rate."""
    t0, t1 = time_s[0], time_s[-1]
    if t1 <= t0:
        raise ValueError("Invalid time range")
    n = max(int((t1 - t0) * fs) + 1, 2)
    t_new = np.linspace(t0, t1, n)
    f = interp1d(time_s, pressure, kind="linear", fill_value="extrapolate")
    return t_new, f(t_new)


def detect_beats(
    time_s: np.ndarray,
    pressure: np.ndarray,
    fs: float = DEFAULT_FS,
) -> list[BeatInfo]:
    """
    Detect systolic peaks with progressive fallback find_peaks settings.
    Boundaries from adjacent peak midpoints, refined to local minima.
    """
    t, p = clean_trace(time_s, pressure)
    t, p = resample_trace(t, p, fs)
    p_smooth = signal.savgol_filter(p, window_length=min(51, len(p) // 2 * 2 + 1), polyorder=3)

    peaks = _find_peaks_progressive(p_smooth)
    if len(peaks) < 1:
        return []

    beats: list[BeatInfo] = []
    for i, pk in enumerate(peaks):
        if i == 0:
            start = 0
        else:
            start = int((peaks[i - 1] + pk) / 2)
        if i == len(peaks) - 1:
            end = len(p) - 1
        else:
            end = int((pk + peaks[i + 1]) / 2)
        start = _refine_to_minimum(p_smooth, start, direction=-1)
        end = _refine_to_minimum(p_smooth, end, direction=1)
        bi = BeatInfo(
            start_idx=start,
            end_idx=end,
            peak_idx=pk,
            start_time=float(t[start]),
            end_time=float(t[end]),
            peak_time=float(t[pk]),
            peak_pressure=float(p[pk]),
        )
        bi.qc_passed, bi.qc_message = _qc_beat(t, p, bi)
        bi.keep = bi.qc_passed
        beats.append(bi)
    return beats


def _find_peaks_progressive(signal_1d: np.ndarray) -> list[int]:
    settings = [
        {"distance": 50, "prominence": 0.5},
        {"distance": 30, "prominence": 0.3},
        {"distance": 20, "prominence": 0.15},
        {"distance": 10, "prominence": 0.05},
    ]
    for kw in settings:
        idx, _ = signal.find_peaks(signal_1d, **kw)
        if len(idx) >= 1:
            return list(idx)
    return []


def _refine_to_minimum(p: np.ndarray, idx: int, direction: int, window: int = 30) -> int:
    lo = max(0, idx - window)
    hi = min(len(p) - 1, idx + window)
    seg = p[lo : hi + 1]
    return lo + int(np.argmin(seg))


def _qc_beat(t: np.ndarray, p: np.ndarray, beat: BeatInfo) -> tuple[bool, str]:
    """Physiologic QC checks."""
    duration = beat.end_time - beat.start_time
    if duration < 0.15 or duration > 2.0:
        return False, f"duration {duration:.3f}s out of range"
    seg = p[beat.start_idx : beat.end_idx + 1]
    excursion = float(np.max(seg) - np.min(seg))
    if excursion < 1.0:
        return False, f"excursion {excursion:.2f} mmHg too small"
    rel_peak = (beat.peak_idx - beat.start_idx) / max(beat.end_idx - beat.start_idx, 1)
    if rel_peak < 0.15 or rel_peak > 0.85:
        return False, "peak timing too early/late"
    baseline_start = float(np.median(seg[: max(5, len(seg) // 10)]))
    baseline_end = float(np.median(seg[-max(5, len(seg) // 10) :]))
    if abs(baseline_start - baseline_end) > 0.5 * excursion:
        return False, "baseline closure mismatch"
    return True, "ok"


def beats_to_dicts(beats: list[BeatInfo]) -> list[dict]:
    return [
        {
            "start_idx": b.start_idx,
            "end_idx": b.end_idx,
            "peak_idx": b.peak_idx,
            "start_time": b.start_time,
            "end_time": b.end_time,
            "peak_time": b.peak_time,
            "peak_pressure": b.peak_pressure,
            "keep": b.keep,
            "qc_passed": b.qc_passed,
            "qc_message": b.qc_message,
        }
        for b in beats
    ]


def dicts_to_beats(data: list[dict]) -> list[BeatInfo]:
    return [BeatInfo(**{k: d[k] for k in BeatInfo.__dataclass_fields__ if k in d}) for d in data]


def average_beats(
    time_s: np.ndarray,
    pressure: np.ndarray,
    beats: list[BeatInfo],
    fs: float = DEFAULT_FS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Resample beats, align by upstroke foot (fallback max dP/dt), average pressure.
    Returns (t_avg, p_avg, mean_correlation).
    """
    t, p = resample_trace(*clean_trace(time_s, pressure), fs)
    kept = [b for b in beats if b.keep]
    if not kept:
        raise ValueError("No beats selected for averaging")

    segments: list[np.ndarray] = []
    corrs: list[float] = []
    ref_seg: np.ndarray | None = None

    for beat in kept:
        seg_t = t[beat.start_idx : beat.end_idx + 1]
        seg_p = p[beat.start_idx : beat.end_idx + 1]
        if len(seg_p) < 10:
            continue
        foot_idx = _find_foot(seg_p, fs)
        shift = foot_idx
        aligned = seg_p[shift:]
        if len(aligned) < 10:
            continue
        # Normalize length
        target_len = 500
        x_old = np.linspace(0, 1, len(aligned))
        x_new = np.linspace(0, 1, target_len)
        aligned_rs = np.interp(x_new, x_old, aligned)
        segments.append(aligned_rs)
        if ref_seg is not None:
            c = np.corrcoef(ref_seg, aligned_rs)[0, 1]
            if np.isfinite(c):
                corrs.append(float(c))
        else:
            ref_seg = aligned_rs

    if not segments:
        raise ValueError("Could not extract valid beat segments")

    stack = np.vstack(segments)
    p_avg = np.mean(stack, axis=0)
    t_avg = np.linspace(0, 1, len(p_avg))  # normalized beat phase
    mean_corr = float(np.mean(corrs)) if corrs else 1.0
    return t_avg, p_avg, mean_corr


def _find_foot(p: np.ndarray, fs: float) -> int:
    """Upstroke foot via max dP/dt before peak."""
    dpdt = np.gradient(p, 1.0 / fs)
    peak = int(np.argmax(p))
    if peak > 5:
        return int(np.argmax(dpdt[:peak]))
    return 0


def sync_keep_flags(beats: list[BeatInfo]) -> None:
    """Ensure keep flags array matches beats length."""
    for b in beats:
        if not hasattr(b, "keep"):
            b.keep = True
