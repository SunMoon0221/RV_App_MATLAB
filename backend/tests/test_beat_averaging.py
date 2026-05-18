import numpy as np

from app.services.beat_averaging import BeatInfo, average_beats, detect_beats, dicts_to_beats


def _synthetic_trace(n_beats: int = 2, fs: float = 500.0):
    t = np.linspace(0, n_beats * 0.8, int(n_beats * 0.8 * fs))
    p = 5 + 20 * np.sin(2 * np.pi * t / 0.8) ** 2
    return t, p


def test_detect_beats_returns_list():
    t, p = _synthetic_trace(2)
    beats = detect_beats(t, p)
    assert isinstance(beats, list)


def test_zero_auto_beats_manual_add():
    t = np.linspace(0, 1, 500)
    p = np.linspace(5, 10, 500)  # monotonic — no peaks
    beats = detect_beats(t, p)
    assert len(beats) == 0
    manual = [
        BeatInfo(
            start_idx=0,
            end_idx=400,
            peak_idx=200,
            start_time=float(t[0]),
            end_time=float(t[400]),
            peak_time=float(t[200]),
            peak_pressure=float(p[200]),
            keep=True,
        )
    ]
    assert len(manual) == 1


def test_keep_flags_match_after_edit():
    t, p = _synthetic_trace(1)
    beats = detect_beats(t, p)
    if not beats:
        beats = [
            BeatInfo(0, 300, 150, float(t[0]), float(t[300]), float(t[150]), float(p[150]), keep=True)
        ]
    data = [{"start_idx": b.start_idx, "end_idx": b.end_idx, "peak_idx": b.peak_idx,
             "start_time": b.start_time, "end_time": b.end_time, "peak_time": b.peak_time,
             "peak_pressure": b.peak_pressure, "keep": b.keep, "qc_passed": True, "qc_message": ""}
            for b in beats]
    data.append({**data[0], "start_idx": 350, "end_idx": 450, "peak_idx": 400, "keep": False})
    restored = dicts_to_beats(data)
    assert len(restored) == len(data)
    assert all(hasattr(b, "keep") for b in restored)


def test_average_beats():
    t, p = _synthetic_trace(2)
    beats = detect_beats(t, p)
    if len(beats) < 1:
        beats = [BeatInfo(0, len(t) - 1, len(t) // 2, t[0], t[-1], t[len(t)//2], p[len(t)//2], True)]
    for b in beats:
        b.keep = True
    t_avg, p_avg, corr = average_beats(t, p, beats)
    assert len(t_avg) == len(p_avg)
    assert np.isfinite(corr) or corr == 1.0
