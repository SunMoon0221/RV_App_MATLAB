"""CSV/JSON/PNG export helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.session_store import SessionState, SessionStore


def save_calibrated_trace_csv(path: Path, time_s: np.ndarray, pressure: np.ndarray) -> None:
    df = pd.DataFrame({"time_s": time_s, "pressure_mmhg": pressure})
    df.to_csv(path, index=False)


def save_averaged_waveform_csv(path: Path, time_s: np.ndarray, pressure: np.ndarray) -> None:
    df = pd.DataFrame({"time_s": time_s, "pressure_mmhg": pressure})
    df.to_csv(path, index=False)


def save_beat_selection_csv(path: Path, beats: list[dict[str, Any]]) -> None:
    pd.DataFrame(beats).to_csv(path, index=False)


def save_pmax_summary_csv(path: Path, methods: list[dict[str, Any]]) -> None:
    rows = []
    for m in methods:
        rows.append(
            {
                "name": m.get("name"),
                "success": m.get("success"),
                "message": m.get("message"),
                "raw_pmax": m.get("raw_pmax"),
                "scaled_pmax": m.get("scaled_pmax"),
                "fit_r2": m.get("fit_r2"),
                "is_selected": m.get("is_selected"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_pmax_curve_csv(path: Path, method: dict[str, Any]) -> None:
    t = method.get("t_fit") or []
    p = method.get("p_fit_raw") or []
    pd.DataFrame({"time_s": t, "pressure_mmhg": p}).to_csv(path, index=False)


def save_patient_data_csv(path: Path, hemo: dict[str, Any], sv: float) -> None:
    row = {**hemo, "stroke_volume_ml": sv}
    pd.DataFrame([row]).to_csv(path, index=False)


def save_analysis_summary_txt(path: Path, hemo: dict[str, Any], methods: list[dict[str, Any]]) -> None:
    lines = ["RV Single-Beat Analysis Summary", "=" * 40]
    for k, v in hemo.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Pmax Methods:")
    for m in methods:
        status = "OK" if m.get("success") else f"FAIL: {m.get('message')}"
        lines.append(f"  {m.get('name')}: {status} raw={m.get('raw_pmax')} scaled={m.get('scaled_pmax')}")
    path.write_text("\n".join(lines))


def export_session_bundle(store: SessionStore, state: SessionState) -> Path:
    """Write all export CSVs/JSON and return path to zip."""
    sid = state.session_id
    exp = store._path(sid) / "exports"
    exp.mkdir(parents=True, exist_ok=True)

    cal = store.load_array(sid, "calibrated_trace")
    if cal is not None and cal.ndim == 2 and cal.shape[1] >= 2:
        save_calibrated_trace_csv(exp / "calibrated_trace.csv", cal[:, 0], cal[:, 1])

    avg = store.load_array(sid, "averaged_waveform")
    if avg is not None and avg.ndim == 2:
        save_averaged_waveform_csv(exp / "averaged_waveform.csv", avg[:, 0], avg[:, 1])

    beats = state.manual_beats or state.detected_beats
    if beats:
        save_beat_selection_csv(exp / "beat_selection.csv", beats)

    if state.hemodynamic_results:
        save_patient_data_csv(exp / "patient_data.csv", state.hemodynamic_results, state.hemodynamic_results.get("sv", 0))

    if state.pmax_method_results:
        save_pmax_summary_csv(exp / "pmax_method_summary.csv", state.pmax_method_results)
        for m in state.pmax_method_results:
            safe = (m.get("short_name") or "method").replace("/", "_")
            save_pmax_curve_csv(exp / f"pmax_curve_{safe}.csv", m)

    (exp / "session_state.json").write_text(json.dumps(state.to_dict(), indent=2))
    if state.hemodynamic_results:
        save_analysis_summary_txt(
            exp / "analysis_summary.txt",
            state.hemodynamic_results,
            state.pmax_method_results,
        )

    zip_path = store._path(sid) / "exports_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in exp.iterdir():
            if f.is_file() and f.suffix != ".zip":
                zf.write(f, arcname=f.name)
    return zip_path
