"""Synthetic session for demo/testing."""

from __future__ import annotations

import numpy as np

from app.session_store import CalibrationParams, SessionStore
from app.services.exports import save_calibrated_trace_csv


def build_mock_session(store: SessionStore) -> dict:
    state = store.create()
    sid = state.session_id
    fs = 500.0
    t = np.linspace(0, 1.2, int(1.2 * fs))
    p = 5 + 25 * np.sin(np.pi * t / 0.35) ** 2 * np.exp(-2 * np.maximum(t - 0.35, 0))
    p = np.clip(p, 3, 35)

    store.save_array(sid, "calibrated_trace", np.column_stack([t, p]))
    params = CalibrationParams(origin_x=0, origin_y=0, x_ratio=1 / fs, y_ratio=1.0)
    state.calibration = params
    state.width = 800
    state.height = 400
    store.save_state(state)
    save_calibrated_trace_csv(store.export_path(sid, "calibrated_trace.csv"), t, p)

    return {
        "session_id": sid,
        "message": "Mock session with synthetic calibrated trace",
        "calibrated_trace_url": f"/api/session/{sid}/calibrated_trace.json",
    }
