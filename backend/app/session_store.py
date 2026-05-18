"""Local session storage — each session is isolated under data/sessions/{id}."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Project root: backend/../
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "sessions"


@dataclass
class CalibrationParams:
    origin_x: float = 0.0
    origin_y: float = 0.0
    x_ratio: float = 1.0  # seconds per pixel
    y_ratio: float = 1.0  # mmHg per pixel
    time_span_seconds: float = 1.0
    pressure_span_mmhg: float = 1.0


@dataclass
class SessionState:
    """In-memory view of persisted session; reload from disk on access."""

    session_id: str
    width: int = 0
    height: int = 0

    # Processing params
    threshold: int = 128
    apply_notch_filter: bool = False

    calibration: CalibrationParams = field(default_factory=CalibrationParams)

    # Beat / analysis metadata (JSON-serializable)
    detected_beats: list[dict[str, Any]] = field(default_factory=list)
    manual_beats: list[dict[str, Any]] = field(default_factory=list)
    averaged_waveform: list[dict[str, float]] | None = None
    selected_peak_markers: dict[str, Any] | None = None
    pmax_method_results: list[dict[str, Any]] = field(default_factory=list)
    hemodynamic_results: dict[str, Any] | None = None
    selected_pmax_method: str = "Original Piecewise Sinusoid"
    output_folder: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["calibration"] = asdict(self.calibration)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        cal = data.pop("calibration", {})
        state = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if cal:
            state.calibration = CalibrationParams(**cal)
        return state


class SessionStore:
    """Filesystem-backed session manager."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or DATA_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(self) -> SessionState:
        session_id = str(uuid.uuid4())
        path = self._path(session_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "exports").mkdir(exist_ok=True)
        state = SessionState(session_id=session_id)
        self.save_state(state)
        return state

    def _path(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def get(self, session_id: str) -> SessionState:
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        return SessionState.from_dict(json.loads((path / "session_state.json").read_text()))

    def save_state(self, state: SessionState) -> None:
        path = self._path(state.session_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "session_state.json").write_text(json.dumps(state.to_dict(), indent=2))

    def reset_derived(self, session_id: str) -> None:
        """Clear all derived artifacts when a new image is loaded."""
        state = self.get(session_id)
        path = self._path(session_id)
        for name in [
            "base_binary_mask.npy",
            "manual_erase_keep_mask.npy",
            "trace_keep_mask.npy",
            "filtered_mask.npy",
            "edges_mask.npy",
            "median_rows.npy",
            "calibrated_trace.npy",
            "averaged_waveform.npy",
            "detected_beats.json",
            "manual_beats.json",
            "pmax_results.json",
            "hemodynamics.json",
        ]:
            f = path / name
            if f.exists():
                f.unlink()
        state.detected_beats = []
        state.manual_beats = []
        state.averaged_waveform = None
        state.selected_peak_markers = None
        state.pmax_method_results = []
        state.hemodynamic_results = None
        state.calibration = CalibrationParams()
        self.save_state(state)

    def save_mask(self, session_id: str, name: str, mask: np.ndarray) -> None:
        np.save(self._path(session_id) / f"{name}.npy", mask.astype(np.bool_))

    def load_mask(self, session_id: str, name: str) -> np.ndarray | None:
        p = self._path(session_id) / f"{name}.npy"
        if not p.exists():
            return None
        return np.load(p)

    def save_array(self, session_id: str, name: str, arr: np.ndarray) -> None:
        np.save(self._path(session_id) / f"{name}.npy", arr)

    def load_array(self, session_id: str, name: str) -> np.ndarray | None:
        p = self._path(session_id) / f"{name}.npy"
        if not p.exists():
            return None
        return np.load(p)

    def save_image(self, session_id: str, name: str, data: bytes) -> Path:
        path = self._path(session_id) / name
        path.write_bytes(data)
        return path

    def image_path(self, session_id: str, name: str) -> Path:
        return self._path(session_id) / name

    def export_path(self, session_id: str, filename: str) -> Path:
        p = self._path(session_id) / "exports" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            shutil.rmtree(path)


store = SessionStore()
