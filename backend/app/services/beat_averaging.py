"""Beat detection, QC, alignment, and ensemble averaging (MATLAB-faithful)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from app.services.derivatives import DEFAULT_FS, gaussian_smooth_pressure


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
    mask = np.isfinite(time_s) & np.isfinite(pressure)
    return time_s[mask], pressure[mask]


def resample_trace(
    time_s: np.ndarray, pressure: np.ndarray, fs: float = DEFAULT_FS
) -> tuple[np.ndarray, np.ndarray]:
    t0, t1 = time_s[0], time_s[-1]
    if t1 <= t0:
        raise ValueError("Invalid time range")
    t_new = np.arange(t0, t1 + 0.5 / fs, 1.0 / fs)
    if t_new[-1] > t1:
        t_new = t_new[t_new <= t1]
    if t_new[-1] < t1:
        t_new = np.append(t_new, t1)
    f = interp1d(time_s, pressure, kind="linear", fill_value="extrapolate")
    return t_new, f(t_new)


def detect_beats(
    time_s: np.ndarray,
    pressure: np.ndarray,
    fs: float = DEFAULT_FS,
) -> list[BeatInfo]:
    """Robust peak-midpoint segmentation with physiologic QC."""
    t, p = clean_trace(time_s, pressure)
    t, p = resample_trace(t, p, fs)
    p_det = gaussian_smooth_pressure(p, fs, max(8.0, 0.010 * 1000))

    trace_range = float(np.ptp(p_det))
    if trace_range <= 0:
        return []

    min_peak_dist = max(int(round(0.35 * fs)), 1)
    min_prom = max(0.75, 0.08 * trace_range)
    pk_locs, pk_props = signal.find_peaks(
        p_det, distance=min_peak_dist, prominence=min_prom
    )
    pk_vals = pk_props.get("prominences", np.ones(len(pk_locs)))
    if len(pk_locs) < 2:
        pk_locs, pk_props = signal.find_peaks(
            p_det,
            distance=max(int(round(0.25 * fs)), 1),
            prominence=max(0.40, 0.05 * trace_range),
        )
        pk_vals = pk_props.get("prominences", np.ones(len(pk_locs)))
    if len(pk_locs) < 2:
        pk_locs, pk_props = signal.find_peaks(
            p_det,
            distance=max(int(round(0.20 * fs)), 1),
            prominence=max(0.20, 0.03 * trace_range),
        )
        pk_vals = pk_props.get("prominences", np.ones(len(pk_locs)))
    if len(pk_locs) < 1:
        return []

    pk_locs = np.atleast_1d(np.asarray(pk_locs, dtype=int))
    pk_vals = np.atleast_1d(np.asarray(pk_vals, dtype=float))
    med_pk = float(np.median(pk_vals))
    keep_pk = pk_vals >= (med_pk - 0.35 * trace_range)
    if np.sum(keep_pk) >= 1:
        pk_locs = pk_locs[keep_pk]
        pk_vals = pk_vals[keep_pk]
    if len(pk_locs) < 2:
        return []

    valley_locs, _ = signal.find_peaks(
        -p_det,
        distance=max(int(round(0.10 * fs)), 1),
        prominence=max(0.05, 0.01 * trace_range),
    )
    valley_locs = np.asarray(valley_locs, dtype=int)

    beats_raw: list[tuple[int, int, int]] = []
    n_peaks = len(pk_locs)
    for ii, pk in enumerate(pk_locs):
        if ii == 0:
            left0 = max(0, pk - int(round(0.55 * (pk_locs[ii + 1] - pk)))) if n_peaks >= 2 else max(0, pk - int(round(0.40 * fs)))
        else:
            left0 = int(round((pk_locs[ii - 1] + pk) / 2))
        if ii == n_peaks - 1:
            right0 = min(len(p_det) - 1, pk + int(round(0.55 * (pk - pk_locs[ii - 1])))) if n_peaks >= 2 else min(len(p_det) - 1, pk + int(round(0.40 * fs)))
        else:
            right0 = int(round((pk + pk_locs[ii + 1]) / 2))
        left0 = max(0, min(left0, pk - 2))
        right0 = min(len(p_det) - 1, max(right0, pk + 2))
        lv = _refine_boundary_to_minimum(p_det, left0, pk, valley_locs, "left")
        rv = _refine_boundary_to_minimum(p_det, pk, right0, valley_locs, "right")
        if lv is None or rv is None or not (lv < pk < rv):
            continue
        dur = (rv - lv + 1) / fs
        p_left, p_right, p_peak = p_det[lv], p_det[rv], p_det[pk]
        excursion = p_peak - min(p_left, p_right)
        if excursion <= 0 or dur < 0.25 or dur > 2.0:
            continue
        close_frac = abs(p_right - p_left) / max(excursion, 1e-9)
        if close_frac > 0.40:
            continue
        pk_frac = (pk - lv) / max(rv - lv, 1)
        if pk_frac < 0.12 or pk_frac > 0.82:
            continue
        beats_raw.append((lv, rv, pk))

    beats: list[BeatInfo] = []
    for lv, rv, pk in beats_raw:
        bi = BeatInfo(
            start_idx=lv,
            end_idx=rv,
            peak_idx=pk,
            start_time=float(t[lv]),
            end_time=float(t[rv]),
            peak_time=float(t[pk]),
            peak_pressure=float(p[pk]),
        )
        bi.qc_passed, bi.qc_message = _qc_beat(t, p, bi)
        bi.keep = bi.qc_passed
        beats.append(bi)
    return beats


def _refine_boundary_to_minimum(
    p_det: np.ndarray,
    i_start: int,
    i_end: int,
    valley_locs: np.ndarray,
    side: str,
) -> int | None:
    if i_end <= i_start:
        return None
    in_window = valley_locs[(valley_locs >= i_start) & (valley_locs <= i_end)]
    if in_window.size:
        if side == "left":
            return int(in_window[np.argmin(np.abs(in_window - i_start))])
        return int(in_window[np.argmin(np.abs(in_window - i_end))])
    seg = p_det[i_start : i_end + 1]
    return i_start + int(np.argmin(seg))


def _qc_beat(t: np.ndarray, p: np.ndarray, beat: BeatInfo) -> tuple[bool, str]:
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
    """JSON-safe beat dicts (plain int/float, no numpy scalars)."""

    def _num(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    return [
        {
            "start_idx": int(b.start_idx),
            "end_idx": int(b.end_idx),
            "peak_idx": int(b.peak_idx),
            "start_time": _num(b.start_time),
            "end_time": _num(b.end_time),
            "peak_time": _num(b.peak_time),
            "peak_pressure": _num(b.peak_pressure),
            "keep": bool(b.keep),
            "qc_passed": bool(b.qc_passed),
            "qc_message": str(b.qc_message),
        }
        for b in beats
    ]


def dicts_to_beats(data: list[dict]) -> list[BeatInfo]:
    return [BeatInfo(**{k: d[k] for k in BeatInfo.__dataclass_fields__ if k in d}) for d in data]


def align_and_normalize_beats_for_average(
    beats_raw: list[np.ndarray],
    keep_flags: list[bool] | np.ndarray,
    fs: float = DEFAULT_FS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB alignAndNormalizeBeatsForAverage.
    Each beat: Nx2 [time, pressure].
    Returns all_beats_norm, avg_time, avg_pressure, beat_quality.
    """
    n_total = len(beats_raw)
    if n_total < 1:
        return np.array([]), np.array([]), np.array([]), np.array([])

    keep = np.asarray(keep_flags, dtype=bool).ravel()
    if keep.size != n_total:
        raise ValueError("keep_flags length does not match beats_raw")
    if not np.any(keep):
        raise ValueError("At least one beat must be selected for averaging")

    beats_rs: list[np.ndarray | None] = [None] * n_total
    beat_len = np.full(n_total, np.nan)
    align_idx = np.full(n_total, np.nan)

    for ii in range(n_total):
        b = beats_raw[ii]
        if b is None or b.size == 0 or b.shape[1] < 2 or b.shape[0] < 6:
            continue
        tb = b[:, 0]
        pb = b[:, 1]
        _, ia = np.unique(tb, return_index=True)
        tb = tb[np.sort(ia)]
        pb = pb[np.sort(ia)]
        if tb.size < 6 or tb[-1] <= tb[0]:
            continue
        t_uniform = np.arange(tb[0], tb[-1] + 0.5 / fs, 1.0 / fs)
        if t_uniform[-1] > tb[-1]:
            t_uniform = t_uniform[t_uniform <= tb[-1]]
        p_uniform = np.interp(t_uniform, tb, pb)
        beats_rs[ii] = np.column_stack([t_uniform, p_uniform])
        beat_len[ii] = len(p_uniform)

        sig = max(3, int(round(0.008 * fs)))
        p_det = gaussian_smooth_pressure(p_uniform, fs, sig * 1000 / fs)
        dpdt = np.gradient(p_det) * fs
        i_peak = int(np.argmax(p_det))
        if i_peak < 4:
            align_idx[ii] = int(np.argmax(dpdt))
            continue
        pre_end = max(3, min(i_peak - 2, int(round(0.35 * i_peak))))
        p_base = float(np.median(p_det[:pre_end]))
        amp = float(np.max(p_det) - p_base)
        if not np.isfinite(amp) or amp <= 0.25:
            align_idx[ii] = int(np.argmax(dpdt))
            continue
        dp_thresh = max(0.10 * float(np.max(dpdt)), 0.5)
        p_thresh = p_base + 0.10 * amp
        cand = np.where(
            (np.arange(len(p_det)) < i_peak) & (p_det <= p_thresh) & (dpdt <= dp_thresh)
        )[0]
        align_idx[ii] = int(cand[-1]) if cand.size else int(np.argmax(dpdt))

    valid_keep = keep & np.array([b is not None for b in beats_rs]) & np.isfinite(align_idx)
    if not np.any(valid_keep):
        raise ValueError("No valid beats remained for alignment")

    ref_idx = int(round(float(np.median(align_idx[valid_keep]))))
    pre_counts = align_idx[valid_keep] - 1
    post_counts = beat_len[valid_keep] - align_idx[valid_keep]
    target_pre = int(np.max(pre_counts))
    target_post = int(np.max(post_counts))
    target_len = target_pre + target_post + 1

    aligned = np.full((target_len, n_total), np.nan)
    for ii in range(n_total):
        if not valid_keep[ii]:
            continue
        p = beats_rs[ii][:, 1]
        a = int(align_idx[ii])
        insert_start = target_pre - (a - 1) + 1
        insert_end = insert_start + len(p) - 1
        src_start, src_end = 1, len(p)
        if insert_start < 1:
            src_start = 2 - insert_start
            insert_start = 1
        if insert_end > target_len:
            overflow = insert_end - target_len
            src_end = len(p) - overflow
            insert_end = target_len
        if insert_start <= insert_end and src_start <= src_end:
            aligned[insert_start - 1 : insert_end, ii] = p[src_start - 1 : src_end]

    prelim_avg = np.nanmean(aligned[:, valid_keep], axis=1)
    refine_max = max(1, int(round(0.03 * fs)))
    aligned_refined = aligned.copy()
    for ii in range(n_total):
        if not valid_keep[ii]:
            continue
        beat_col = aligned[:, ii]
        v = np.isfinite(beat_col) & np.isfinite(prelim_avg)
        if np.sum(v) < max(20, int(round(0.15 * fs))):
            continue
        x = beat_col[v] - np.nanmean(beat_col[v])
        y = prelim_avg[v] - np.nanmean(prelim_avg[v])
        if len(x) < 10:
            continue
        corr = np.correlate(x, y, mode="full")
        lags = np.arange(-len(x) + 1, len(x))
        best_lag = int(lags[int(np.argmax(corr))])
        if abs(best_lag) > refine_max:
            best_lag = int(np.sign(best_lag) * refine_max)
        if best_lag != 0:
            shifted = np.full_like(beat_col, np.nan)
            if best_lag > 0:
                shifted[:-best_lag] = beat_col[best_lag:]
            else:
                L = abs(best_lag)
                shifted[L:] = beat_col[:-L]
            aligned_refined[:, ii] = shifted

    support_count = np.sum(np.isfinite(aligned_refined[:, valid_keep]), axis=1)
    min_support = max(1, int(np.ceil(0.60 * np.sum(valid_keep))))
    supported = np.where(support_count >= min_support)[0]
    if supported.size == 0:
        supported = np.where(np.any(np.isfinite(aligned_refined[:, valid_keep]), axis=1))[0]
    if supported.size == 0:
        raise ValueError("No supported aligned region remained after refinement")

    i1, i2 = int(supported[0]), int(supported[-1])
    aligned_refined = aligned_refined[i1 : i2 + 1, :]
    target_norm_len = int(
        round(float(np.median(np.sum(np.isfinite(aligned_refined[:, valid_keep]), axis=0))))
    )
    target_norm_len = max(target_norm_len, 20)

    all_beats_norm = np.full((target_norm_len, n_total), np.nan)
    for ii in range(n_total):
        col = aligned_refined[:, ii]
        good = np.isfinite(col)
        if np.sum(good) < 8:
            continue
        pseg = col[good]
        told = np.linspace(0, 1, len(pseg))
        tnew = np.linspace(0, 1, target_norm_len)
        all_beats_norm[:, ii] = np.interp(tnew, told, pseg)

    avg_pressure = np.nanmean(all_beats_norm[:, valid_keep], axis=1)
    avg_time = np.arange(target_norm_len) / fs

    beat_quality = np.full(n_total, np.nan)
    for ii in range(n_total):
        col = all_beats_norm[:, ii]
        if np.any(~np.isfinite(col)):
            continue
        c = np.corrcoef(avg_pressure, col)[0, 1]
        if np.isfinite(c):
            beat_quality[ii] = c

    return all_beats_norm, avg_time, avg_pressure, beat_quality


def average_beats(
    time_s: np.ndarray,
    pressure: np.ndarray,
    beats: list[BeatInfo],
    fs: float = DEFAULT_FS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Average using MATLAB alignment pipeline."""
    t, p = resample_trace(*clean_trace(time_s, pressure), fs)
    kept = [b for b in beats if b.keep]
    if not kept:
        raise ValueError("No beats selected for averaging")

    beats_raw = [np.column_stack([t[b.start_idx : b.end_idx + 1], p[b.start_idx : b.end_idx + 1]]) for b in kept]
    keep_flags = [True] * len(beats_raw)
    _, avg_time, avg_pressure, beat_quality = align_and_normalize_beats_for_average(
        beats_raw, keep_flags, fs
    )
    mean_corr = float(np.nanmean(beat_quality)) if beat_quality.size else 1.0
    if not np.isfinite(mean_corr):
        mean_corr = 1.0
    return avg_time, avg_pressure, mean_corr


def average_pre_segmented_beats(
    beats_raw: list[np.ndarray],
    keep_flags: list[bool],
    fs: float = DEFAULT_FS,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    _, avg_time, avg_pressure, beat_quality = align_and_normalize_beats_for_average(
        beats_raw, keep_flags, fs
    )
    mean_corr = float(np.nanmean(beat_quality)) if beat_quality.size else 1.0
    return avg_time, avg_pressure, mean_corr, beat_quality


def sync_keep_flags(beats: list[BeatInfo]) -> None:
    for b in beats:
        if not hasattr(b, "keep"):
            b.keep = True
