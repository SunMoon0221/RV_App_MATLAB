"""FastAPI application for RV single-beat pressure-volume analysis."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    AverageBeatsRequest,
    AverageBeatsResponse,
    BeatSegment,
    CalibrationRequest,
    CalibrationResponse,
    DetectBeatsResponse,
    MaskFinalizeResponse,
    MaskUpdateRequest,
    ProcessImageRequest,
    ProcessImageResponse,
    ReplaceLineRequest,
    SessionStatusResponse,
    SingleBeatAnalysisRequest,
    SingleBeatAnalysisResponse,
    UploadResponse,
    HemodynamicResults,
    PmaxMethodResultSchema,
)
from app.session_store import CalibrationParams, SessionState, store
from app.services import beat_averaging, calibration, exports, hemodynamics, image_processing
from app.services.beat_averaging import BeatInfo, average_beats, beats_to_dicts, detect_beats, dicts_to_beats
from app.services.calibration import apply_calibration, compute_ratios, replace_line_segment
from app.services.image_processing import (
    compose_filtered_mask,
    extract_median_rows,
    load_image_bgr,
    rebuild_edges_from_mask,
    rethreshold_preserving_masks,
    to_grayscale,
    apply_notch_filter,
    threshold_mask,
    clean_mask,
)

app = FastAPI(title="RV Single-Beat Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _session_url(session_id: str, filename: str) -> str:
    return f"/api/session/{session_id}/file/{filename}"


def _ensure_median_rows(session_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild median rows from current mask if missing."""
    med = store.load_array(session_id, "median_rows")
    if med is not None and med.ndim == 2 and med.shape[1] >= 2:
        return med[:, 0], med[:, 1]
    filtered = store.load_mask(session_id, "filtered_mask")
    if filtered is None:
        raise HTTPException(400, "No mask available; process image first")
    x_px, y_px = extract_median_rows(filtered)
    store.save_array(session_id, "median_rows", np.column_stack([x_px, y_px]))
    edges = rebuild_edges_from_mask(filtered)
    store.save_mask(session_id, "edges_mask", edges)
    return x_px, y_px


def _mask_from_b64(b64: str, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == shape[0] * shape[1]:
        return arr.reshape(shape).astype(bool)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid mask encoding")
    return img > 127


@app.post("/api/session/upload", response_model=UploadResponse)
async def upload_session(file: UploadFile = File(...)) -> UploadResponse:
    data = await file.read()
    state = store.create()
    store.reset_derived(state.session_id)
    store.save_image(state.session_id, "original.png", data)
    store.save_image(state.session_id, "displayed.png", data)

    bgr = load_image_bgr(data)
    state.width, state.height = bgr.shape[1], bgr.shape[0]
    store.save_state(state)

    return UploadResponse(
        session_id=state.session_id,
        width=state.width,
        height=state.height,
        preview_url=_session_url(state.session_id, "displayed.png"),
    )


@app.get("/api/session/{session_id}/status", response_model=SessionStatusResponse)
def session_status(session_id: str) -> SessionStatusResponse:
    try:
        state = store.get(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found") from None

    has_mask = store.load_mask(session_id, "filtered_mask") is not None
    has_cal = store.load_array(session_id, "calibrated_trace") is not None
    has_beats = bool(state.detected_beats or state.manual_beats)
    has_avg = store.load_array(session_id, "averaged_waveform") is not None
    has_analysis = state.hemodynamic_results is not None

    step_status = {
        "upload": "ready" if state.width else "not_ready",
        "process": "processed" if has_mask else "not_ready",
        "edit_mask": "ready" if has_mask else "not_ready",
        "calibrate": "ready" if has_cal else ("needs_calibration" if has_mask else "not_ready"),
        "average_beats": "averaged" if has_avg else "not_ready",
        "single_beat": "analysis_complete" if has_analysis else "not_ready",
        "export": "ready" if has_analysis else "not_ready",
    }

    return SessionStatusResponse(
        session_id=session_id,
        step_status=step_status,
        has_image=state.width > 0,
        has_mask=has_mask,
        has_calibration=has_cal,
        has_beats=has_beats,
        has_average=has_avg,
        has_analysis=has_analysis,
    )


@app.post("/api/session/{session_id}/process", response_model=ProcessImageResponse)
def process_image(session_id: str, body: ProcessImageRequest) -> ProcessImageResponse:
    try:
        state = store.get(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found") from None

    orig_path = store.image_path(session_id, "original.png")
    if not orig_path.exists():
        raise HTTPException(400, "No image uploaded")

    bgr = load_image_bgr(orig_path.read_bytes())
    gray = to_grayscale(bgr)
    if body.apply_notch_filter:
        gray = apply_notch_filter(gray, body.notch_freq, body.sample_rate_hz)

    threshold = body.threshold if body.threshold is not None else state.threshold
    base = clean_mask(threshold_mask(gray, threshold))
    manual = np.ones_like(base, dtype=bool)
    trace = np.ones_like(base, dtype=bool)
    filtered = compose_filtered_mask(base, manual, trace)
    edges = rebuild_edges_from_mask(filtered)
    x_px, y_px = extract_median_rows(filtered)

    store.save_mask(session_id, "base_binary_mask", base)
    store.save_mask(session_id, "manual_erase_keep_mask", manual)
    store.save_mask(session_id, "trace_keep_mask", trace)
    store.save_mask(session_id, "filtered_mask", filtered)
    store.save_mask(session_id, "edges_mask", edges)
    store.save_array(session_id, "median_rows", np.column_stack([x_px, y_px]))
    np.save(store._path(session_id) / "grayscale.npy", gray)

    # Mask preview PNG
    overlay = bgr.copy()
    overlay[filtered] = [0, 255, 0]
    cv2.imwrite(str(store.image_path(session_id, "mask_preview.png")), overlay)

    state.threshold = threshold
    state.apply_notch_filter = body.apply_notch_filter
    store.save_state(state)

    return ProcessImageResponse(
        session_id=session_id,
        width=state.width,
        height=state.height,
        mask_preview_url=_session_url(session_id, "mask_preview.png"),
        median_rows_url=_session_url(session_id, "median_rows.json"),
        status="processed",
    )


@app.get("/api/session/{session_id}/masks")
def get_masks(session_id: str) -> dict:
    """Return base64 PNG masks for editor."""
    try:
        state = store.get(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found") from None
    shape = (state.height, state.width)
    base = store.load_mask(session_id, "base_binary_mask")
    manual = store.load_mask(session_id, "manual_erase_keep_mask")
    trace = store.load_mask(session_id, "trace_keep_mask")
    filtered = store.load_mask(session_id, "filtered_mask")
    if base is None:
        raise HTTPException(400, "Process image first")

    def enc(m: np.ndarray) -> str:
        img = (m.astype(np.uint8) * 255)
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode()

    x_px, y_px = _ensure_median_rows(session_id)
    return {
        "width": state.width,
        "height": state.height,
        "base_mask_b64": enc(base),
        "manual_erase_b64": enc(manual) if manual is not None else enc(np.ones(shape, bool)),
        "trace_keep_b64": enc(trace) if trace is not None else enc(np.ones(shape, bool)),
        "filtered_mask_b64": enc(filtered) if filtered is not None else enc(base),
        "median_rows": {"x": x_px.tolist(), "y": [float(v) if np.isfinite(v) else None for v in y_px]},
        "image_url": _session_url(session_id, "displayed.png"),
    }


@app.post("/api/session/{session_id}/masks/update", response_model=MaskFinalizeResponse)
def update_masks(session_id: str, body: MaskUpdateRequest) -> MaskFinalizeResponse:
    try:
        state = store.get(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found") from None

    shape = (state.height, state.width)
    base = store.load_mask(session_id, "base_binary_mask")
    manual = store.load_mask(session_id, "manual_erase_keep_mask")
    trace = store.load_mask(session_id, "trace_keep_mask")
    if base is None:
        raise HTTPException(400, "Process image first")
    manual = manual if manual is not None else np.ones(shape, bool)
    trace = trace if trace is not None else np.ones(shape, bool)

    if body.manual_erase_keep_mask_b64:
        manual = _mask_from_b64(body.manual_erase_keep_mask_b64, shape)
    if body.trace_keep_mask_b64:
        trace = _mask_from_b64(body.trace_keep_mask_b64, shape)

    gray_path = store._path(session_id) / "grayscale.npy"
    if body.threshold is not None and gray_path.exists():
        gray = np.load(gray_path)
        base, filtered = rethreshold_preserving_masks(gray, body.threshold, manual, trace)
        state.threshold = body.threshold
        store.save_mask(session_id, "base_binary_mask", base)
    else:
        filtered = compose_filtered_mask(base, manual, trace)

    store.save_mask(session_id, "manual_erase_keep_mask", manual)
    store.save_mask(session_id, "trace_keep_mask", trace)
    store.save_mask(session_id, "filtered_mask", filtered)
    edges = rebuild_edges_from_mask(filtered)
    store.save_mask(session_id, "edges_mask", edges)
    x_px, y_px = extract_median_rows(filtered)
    store.save_array(session_id, "median_rows", np.column_stack([x_px, y_px]))

    bgr = load_image_bgr(store.image_path(session_id, "displayed.png").read_bytes())
    overlay = bgr.copy()
    overlay[filtered] = [0, 255, 0]
    cv2.imwrite(str(store.image_path(session_id, "mask_preview.png")), overlay)
    store.save_state(state)

    return MaskFinalizeResponse(
        session_id=session_id,
        median_rows_url=_session_url(session_id, "median_rows.json"),
        mask_preview_url=_session_url(session_id, "mask_preview.png"),
        status="processed",
    )


@app.post("/api/session/{session_id}/masks/finalize", response_model=MaskFinalizeResponse)
def finalize_masks(session_id: str) -> MaskFinalizeResponse:
    _ensure_median_rows(session_id)
    return update_masks(session_id, MaskUpdateRequest())


@app.get("/api/session/{session_id}/median_rows.json")
def median_rows_json(session_id: str) -> dict:
    x_px, y_px = _ensure_median_rows(session_id)
    return {"x": x_px.tolist(), "y": [float(v) if np.isfinite(v) else None for v in y_px]}


@app.post("/api/session/{session_id}/calibrate", response_model=CalibrationResponse)
def calibrate_session(session_id: str, body: CalibrationRequest) -> CalibrationResponse:
    try:
        state = store.get(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found") from None

    x_px, y_px = _ensure_median_rows(session_id)
    x_ratio, y_ratio = compute_ratios(
        body.horizontal_line,
        body.vertical_line,
        body.time_span_seconds,
        body.pressure_span_mmhg,
    )
    ox, oy = body.origin
    params = CalibrationParams(
        origin_x=ox,
        origin_y=oy,
        x_ratio=x_ratio,
        y_ratio=y_ratio,
        time_span_seconds=body.time_span_seconds,
        pressure_span_mmhg=body.pressure_span_mmhg,
    )
    state.calibration = params
    t, p = apply_calibration(x_px, y_px, params)
    valid = np.isfinite(t) & np.isfinite(p)
    store.save_array(session_id, "calibrated_trace", np.column_stack([t[valid], p[valid]]))
    exports.save_calibrated_trace_csv(
        store.export_path(session_id, "calibrated_trace.csv"), t[valid], p[valid]
    )
    store.save_state(state)

    return CalibrationResponse(
        session_id=session_id,
        x_ratio=x_ratio,
        y_ratio=y_ratio,
        calibrated_trace_url=_session_url(session_id, "calibrated_trace.json"),
        status="calibrated",
    )


@app.get("/api/session/{session_id}/calibrated_trace.json")
def calibrated_trace_json(session_id: str) -> dict:
    cal = store.load_array(session_id, "calibrated_trace")
    if cal is None:
        raise HTTPException(400, "Not calibrated")
    return {"time_s": cal[:, 0].tolist(), "pressure_mmhg": cal[:, 1].tolist()}


@app.post("/api/session/{session_id}/replace-line")
def replace_line(session_id: str, body: ReplaceLineRequest) -> dict:
    x_px, y_px = _ensure_median_rows(session_id)
    y_new = replace_line_segment(x_px, y_px.astype(float), body.point1, body.point2)
    store.save_array(session_id, "median_rows", np.column_stack([x_px, y_new]))

    state = store.get(session_id)
    if state.calibration.x_ratio != 1.0 or store.load_array(session_id, "calibrated_trace") is not None:
        t, p = apply_calibration(x_px, y_new, state.calibration)
        valid = np.isfinite(t) & np.isfinite(p)
        store.save_array(session_id, "calibrated_trace", np.column_stack([t[valid], p[valid]]))
        exports.save_calibrated_trace_csv(
            store.export_path(session_id, "calibrated_trace.csv"), t[valid], p[valid]
        )
    return {"status": "ok", "median_rows_url": _session_url(session_id, "median_rows.json")}


@app.post("/api/session/{session_id}/detect-beats", response_model=DetectBeatsResponse)
def detect_beats_endpoint(session_id: str) -> DetectBeatsResponse:
    cal = store.load_array(session_id, "calibrated_trace")
    if cal is None:
        raise HTTPException(400, "Calibrate first")
    beats = detect_beats(cal[:, 0], cal[:, 1])
    state = store.get(session_id)
    state.detected_beats = beats_to_dicts(beats)
    state.manual_beats = list(state.detected_beats)
    store.save_state(state)
    return DetectBeatsResponse(
        session_id=session_id,
        beats=[BeatSegment(**b) for b in state.detected_beats],
        trace_url=_session_url(session_id, "calibrated_trace.json"),
        status="ok",
    )


@app.post("/api/session/{session_id}/average-beats", response_model=AverageBeatsResponse)
def average_beats_endpoint(session_id: str, body: AverageBeatsRequest) -> AverageBeatsResponse:
    cal = store.load_array(session_id, "calibrated_trace")
    if cal is None:
        raise HTTPException(400, "Calibrate first")
    beats = dicts_to_beats([b.model_dump() for b in body.beats])
    if len(beats) != len(body.beats):
        raise HTTPException(400, "Beat count mismatch")
    for b, seg in zip(beats, body.beats):
        b.keep = seg.keep
    t_avg, p_avg, mean_corr = average_beats(cal[:, 0], cal[:, 1], beats)
    # Scale normalized time to physiological duration (mean beat length)
    durations = [b.end_time - b.start_time for b in beats if b.keep]
    dur = float(np.mean(durations)) if durations else 1.0
    t_phys = t_avg * dur
    store.save_array(session_id, "averaged_waveform", np.column_stack([t_phys, p_avg]))

    state = store.get(session_id)
    state.manual_beats = [b.model_dump() for b in body.beats]
    state.averaged_waveform = {"time_s": t_phys.tolist(), "pressure_mmhg": p_avg.tolist()}
    store.save_state(state)
    exports.save_averaged_waveform_csv(
        store.export_path(session_id, "averaged_waveform.csv"), t_phys, p_avg
    )
    exports.save_beat_selection_csv(
        store.export_path(session_id, "beat_selection.csv"),
        state.manual_beats,
    )

    return AverageBeatsResponse(
        session_id=session_id,
        averaged_waveform_url=_session_url(session_id, "averaged_waveform.json"),
        beat_selection_url=_session_url(session_id, "beat_selection.csv"),
        mean_correlation=mean_corr,
        status="averaged",
    )


@app.get("/api/session/{session_id}/averaged_waveform.json")
def averaged_waveform_json(session_id: str) -> dict:
    avg = store.load_array(session_id, "averaged_waveform")
    if avg is None:
        raise HTTPException(400, "No averaged waveform")
    return {"time_s": avg[:, 0].tolist(), "pressure_mmhg": avg[:, 1].tolist()}


@app.post("/api/session/{session_id}/single-beat-analysis", response_model=SingleBeatAnalysisResponse)
def single_beat_analysis(session_id: str, body: SingleBeatAnalysisRequest) -> SingleBeatAnalysisResponse:
    avg = store.load_array(session_id, "averaged_waveform")
    if avg is None:
        raise HTTPException(400, "Average beats first")

    esp_idx = edp_idx = None
    if body.peak_selection:
        peaks = body.peak_selection.event_marker_peak_indices
        if len(peaks) != 4:
            raise HTTPException(400, "Select exactly 4 event-marker peaks")
        esp_idx = body.peak_selection.esp_peak_indices[0]
        edp_idx = body.peak_selection.edp_peak_indices[0]

    hemo, methods = hemodynamics.compute_hemodynamics(
        avg[:, 0],
        avg[:, 1],
        body.stroke_volume_ml,
        body.pmax_scale_factor,
        body.selected_pmax_method,
        body.peak_selection.event_marker_peak_indices if body.peak_selection else None,
        esp_idx,
        edp_idx,
        cutoff_hz=body.filter_cutoff_hz,
        sigma_ms=body.gaussian_sigma_ms,
    )

    state = store.get(session_id)
    state.hemodynamic_results = hemo
    state.pmax_method_results = methods
    state.selected_pmax_method = body.selected_pmax_method
    store.save_state(state)
    exports.save_pmax_summary_csv(store.export_path(session_id, "pmax_method_summary.csv"), methods)
    exports.save_patient_data_csv(
        store.export_path(session_id, "patient_data.csv"), hemo, body.stroke_volume_ml
    )
    exports.save_analysis_summary_txt(
        store.export_path(session_id, "analysis_summary.txt"), hemo, methods
    )

    return SingleBeatAnalysisResponse(
        session_id=session_id,
        hemodynamics=HemodynamicResults(**{k: v for k, v in hemo.items() if k in HemodynamicResults.model_fields}),
        pmax_methods=[PmaxMethodResultSchema(**m) for m in methods],
        selected_method=body.selected_pmax_method,
        status="analysis_complete",
    )


@app.get("/api/session/{session_id}/analysis")
def get_analysis(session_id: str) -> dict:
    state = store.get(session_id)
    return {
        "hemodynamics": state.hemodynamic_results,
        "pmax_methods": state.pmax_method_results,
        "selected_method": state.selected_pmax_method,
    }


@app.get("/api/session/{session_id}/export/zip")
def export_zip(session_id: str) -> FileResponse:
    state = store.get(session_id)
    zip_path = exports.export_session_bundle(store, state)
    return FileResponse(zip_path, media_type="application/zip", filename=f"{session_id}_exports.zip")


@app.get("/api/session/{session_id}/file/{filename}")
def get_file(session_id: str, filename: str) -> FileResponse:
    path = store.image_path(session_id, filename)
    if not path.exists():
        path = store._path(session_id) / filename
    if not path.exists():
        path = store.export_path(session_id, filename)
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path)


@app.post("/api/mock-session")
def create_mock_session() -> dict:
    """Synthetic pressure waveform for testing without image upload."""
    from app.services.mock_session import build_mock_session

    return build_mock_session(store)


# Mount data for static previews
sessions_dir = store.base_dir
sessions_dir.mkdir(parents=True, exist_ok=True)
