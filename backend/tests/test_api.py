"""API integration tests with FastAPI TestClient."""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _png_bytes(w: int = 200, h: int = 100) -> bytes:
    """Synthetic waveform-like image: dark curve on white."""
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    for x in range(w):
        y = int(h * 0.5 + 0.3 * h * np.sin(2 * np.pi * x / w))
        y = max(0, min(h - 1, y))
        arr[y, x] = [20, 20, 20]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_upload_process_masks_flow():
    r = client.post(
        "/api/session/upload",
        files={"file": ("wave.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]

    r2 = client.post(f"/api/session/{sid}/process", json={"threshold": 128})
    assert r2.status_code == 200
    assert "mask_preview" in r2.json()["mask_preview_url"]

    r3 = client.get(f"/api/session/{sid}/masks")
    assert r3.status_code == 200
    assert r3.json()["width"] == 200


def test_invalid_session_404():
    r = client.get("/api/session/00000000-0000-0000-0000-000000000000/status")
    assert r.status_code == 404


def test_file_path_traversal_blocked():
    r = client.post(
        "/api/session/upload",
        files={"file": ("wave.png", _png_bytes(), "image/png")},
    )
    sid = r.json()["session_id"]
    r2 = client.get(f"/api/session/{sid}/file/../../etc/passwd")
    assert r2.status_code in (400, 404)


def test_average_beats_requires_keep():
    r = client.post("/api/mock-session")
    sid = r.json()["session_id"]
    beats = [
        {
            "start_idx": 0,
            "end_idx": 100,
            "peak_idx": 50,
            "start_time": 0.0,
            "end_time": 0.2,
            "peak_time": 0.1,
            "peak_pressure": 20.0,
            "keep": False,
        }
    ]
    r2 = client.post(f"/api/session/{sid}/average-beats", json={"beats": beats})
    assert r2.status_code == 400


def test_reprocess_clears_calibration():
    r = client.post(
        "/api/session/upload",
        files={"file": ("wave.png", _png_bytes(), "image/png")},
    )
    sid = r.json()["session_id"]
    client.post(f"/api/session/{sid}/process", json={"threshold": 128})
    client.post(
        f"/api/session/{sid}/calibrate",
        json={
            "horizontal_line": [0, 50, 100, 50],
            "vertical_line": [0, 0, 0, 50],
            "origin": [0, 50],
            "time_span_seconds": 1.0,
            "pressure_span_mmhg": 30.0,
        },
    )
    client.post(f"/api/session/{sid}/process", json={"threshold": 120})
    st = client.get(f"/api/session/{sid}/status").json()
    assert st["has_calibration"] is False
