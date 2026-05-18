"""Hemodynamics parity checks against MATLAB singleBeatAnalysis formulas."""

import numpy as np
import pytest

from app.services.derivatives import find_closest_index_half_height
from app.services.hemodynamics import compute_hemodynamics, fit_edpvr_anchored


def _single_beat_wave():
    fs = 500.0
    t = np.linspace(0, 0.9, int(0.9 * fs))
    p = 5 + 25 * np.sin(np.pi * np.clip(t / 0.35, 0, 1)) ** 2
    p += 3 * np.sin(2 * np.pi * t / 0.9)
    return t, p


def test_ees_uses_pmax_minus_esp_over_sv():
    t, p = _single_beat_wave()
    sv = 60.0
    peaks = [50, 120, 200, 280]
    hemo, methods = compute_hemodynamics(
        t,
        p,
        sv,
        1.0,
        "Original Piecewise Sinusoid",
        event_marker_peaks=peaks,
    )
    selected = next(m for m in methods if m.get("is_selected"))
    pmax = selected["scaled_pmax"]
    esp = hemo["esp"]
    assert abs(hemo["ees"] - (pmax - esp) / sv) < 1e-6


def test_find_closest_index_half_height():
    em = np.array([0.1, 0.5, 1.0, 0.8, 0.4, 0.2])
    idx = find_closest_index_half_height(em, [2])
    assert idx == 1


def test_fit_edpvr_anchored_positive_beta():
    a, b = fit_edpvr_anchored(esv=40.0, edv=100.0, edp=8.0)
    assert b > 0
    assert a > 0
