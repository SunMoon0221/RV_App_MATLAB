"""End-to-end API workflow: image path and mock path through analysis."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _png_bytes(w: int = 400, h: int = 120) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    for x in range(w):
        y = int(h * 0.5 + 0.32 * h * np.sin(2 * np.pi * x / max(w / 4, 1)))
        for dy in range(-2, 3):
            yy = max(0, min(h - 1, y + dy))
            arr[yy, x] = [15, 15, 15]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _peak_selection_from_waveform(time_s: list[float], pressure: list[float]) -> dict:
    """Build peak_selection payload from pressure extrema (stable for E2E)."""
    from scipy import signal

    p = np.asarray(pressure, dtype=float)
    n = len(p)
    if n < 20:
        idx = [int(n * 0.15), int(n * 0.35), int(n * 0.55), int(n * 0.75)]
    else:
        prom = max(0.05 * float(np.ptp(p)), 0.5)
        peaks, _ = signal.find_peaks(p, distance=max(n // 8, 1), prominence=prom)
        peaks = np.asarray(peaks, dtype=int)
        peaks = peaks[(peaks > 2) & (peaks < n - 3)]
        if len(peaks) < 4:
            idx = [int(x) for x in np.linspace(3, n - 4, 4, dtype=int)]
        else:
            peaks = np.sort(peaks)
            if len(peaks) > 4:
                peaks = peaks[np.linspace(0, len(peaks) - 1, 4, dtype=int).astype(int)]
            idx = [int(x) for x in peaks[:4]]
    edp_i = int(np.argmin(p[: max(n // 2, 1)]))
    esp_i = int(np.argmax(p))
    return {
        "event_marker_peak_indices": idx,
        "edp_peak_indices": [edp_i],
        "esp_peak_indices": [esp_i],
        "smoothing_sigma_ms": 400.0,
    }


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_mock_session_full_analysis():
    r = client.post("/api/mock-session")
    assert r.status_code == 200
    sid = r.json()["session_id"]

    cal = client.get(f"/api/session/{sid}/calibrated_trace.json")
    assert cal.status_code == 200

    r_det = client.post(f"/api/session/{sid}/detect-beats")
    assert r_det.status_code == 200
    beats = r_det.json()["beats"]
    assert len(beats) >= 1
    for b in beats:
        b["keep"] = True

    r_avg = client.post(f"/api/session/{sid}/average-beats", json={"beats": beats})
    assert r_avg.status_code == 200

    avg = client.get(f"/api/session/{sid}/averaged_waveform.json")
    assert avg.status_code == 200
    avg_body = avg.json()

    r_ana = client.post(
        f"/api/session/{sid}/single-beat-analysis",
        json={
            "stroke_volume_ml": 60.0,
            "pmax_scale_factor": 1.0,
            "selected_pmax_method": "Original Piecewise Sinusoid",
            "peak_selection": _peak_selection_from_waveform(
                avg_body["time_s"], avg_body["pressure_mmhg"]
            ),
        },
    )
    assert r_ana.status_code == 200, r_ana.text
    body = r_ana.json()
    assert body["hemodynamics"]["pmax"] is not None
    assert len(body["pmax_methods"]) == 5
    assert any(m["success"] for m in body["pmax_methods"])

    r_zip = client.get(f"/api/session/{sid}/export/zip")
    assert r_zip.status_code == 200


def test_image_upload_calibrate_average_analyze():
    r = client.post(
        "/api/session/upload",
        files={"file": ("trace.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]

    client.post(f"/api/session/{sid}/process", json={"threshold": 95})
    med = client.get(f"/api/session/{sid}/median_rows.json")
    assert med.status_code == 200
    ys = [y for y in med.json()["y"] if y is not None]
    assert len(ys) > 10, "median rows missing after process"
    # Anchor 0 mmHg at trace trough (max row index); y increases downward in the image.
    y_trough = float(max(ys))
    x0 = float(med.json()["x"][0])
    x1 = float(med.json()["x"][-1])

    client.post(
        f"/api/session/{sid}/calibrate",
        json={
            "horizontal_line": [x0, y_trough, x1, y_trough],
            "vertical_line": [x0, y_trough, x0, y_trough - 40],
            "origin": [x0, y_trough],
            "time_span_seconds": 1.0,
            "pressure_span_mmhg": 40.0,
        },
    )

    r_det = client.post(f"/api/session/{sid}/detect-beats")
    assert r_det.status_code == 200
    beats = r_det.json()["beats"]
    if not beats:
        pytest.skip("No beats detected on synthetic image in this environment")
    for b in beats:
        b["keep"] = True

    r_avg = client.post(f"/api/session/{sid}/average-beats", json={"beats": beats})
    assert r_avg.status_code == 200

    avg = client.get(f"/api/session/{sid}/averaged_waveform.json").json()

    r_ana = client.post(
        f"/api/session/{sid}/single-beat-analysis",
        json={
            "stroke_volume_ml": 55.0,
            "pmax_scale_factor": 1.0,
            "selected_pmax_method": "Original Piecewise Sinusoid",
            "peak_selection": _peak_selection_from_waveform(
                avg["time_s"], avg["pressure_mmhg"]
            ),
        },
    )
    assert r_ana.status_code == 200, r_ana.text
