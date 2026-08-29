"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    session_id: str
    width: int
    height: int
    preview_url: str


class ProcessImageRequest(BaseModel):
    threshold: int | None = Field(default=None, ge=0, le=255)
    apply_notch_filter: bool = False
    notch_freq: float = 50.0
    sample_rate_hz: float = 500.0


class ProcessImageResponse(BaseModel):
    session_id: str
    width: int
    height: int
    mask_preview_url: str
    median_rows_url: str | None = None
    status: str = "processed"


class MaskUpdateRequest(BaseModel):
    """Serialized boolean mask as flat list or base64; frontend sends RLE or PNG."""
    manual_erase_keep_mask_b64: str | None = None
    trace_keep_mask_b64: str | None = None
    threshold: int | None = None


class MaskFinalizeResponse(BaseModel):
    session_id: str
    median_rows_url: str
    mask_preview_url: str
    status: str


class CalibrationRequest(BaseModel):
    horizontal_line: tuple[float, float, float, float]  # x1,y1,x2,y2 pixels
    vertical_line: tuple[float, float, float, float]
    origin: tuple[float, float]
    time_span_seconds: float
    pressure_span_mmhg: float


class CalibrationResponse(BaseModel):
    session_id: str
    x_ratio: float
    y_ratio: float
    calibrated_trace_url: str
    status: str = "calibrated"


class ReplaceLineRequest(BaseModel):
    point1: tuple[float, float]
    point2: tuple[float, float]


class BeatSegment(BaseModel):
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


class DetectBeatsResponse(BaseModel):
    session_id: str
    beats: list[BeatSegment]
    trace_url: str
    status: str


class AverageBeatsRequest(BaseModel):
    beats: list[BeatSegment]


class AverageBeatsResponse(BaseModel):
    session_id: str
    averaged_waveform_url: str
    beat_selection_url: str
    mean_correlation: float
    status: str


class PeakSelectionInput(BaseModel):
    edp_peak_indices: list[int] = Field(..., min_length=1)
    esp_peak_indices: list[int] = Field(..., min_length=1)
    event_marker_peak_indices: list[int] = Field(..., min_length=4, max_length=4)
    smoothing_sigma_ms: float = 70.0


class SingleBeatAnalysisRequest(BaseModel):
    stroke_volume_ml: float = Field(..., gt=0)
    pmax_scale_factor: float = 1.0
    selected_pmax_method: str = "Original Piecewise Sinusoid"
    peak_selection: PeakSelectionInput | None = None
    filter_cutoff_hz: float = 25.0
    gaussian_sigma_ms: float = 70.0


class PmaxMethodResultSchema(BaseModel):
    name: str
    short_name: str
    kind: str
    success: bool
    message: str
    raw_pmax: float | None = None
    scaled_pmax: float | None = None
    scale_factor: float = 1.0
    fit_r2: float | None = None
    pt1: dict[str, float] | None = None
    pt2: dict[str, float] | None = None
    pt3: dict[str, float] | None = None
    pt4: dict[str, float] | None = None
    t_fit: list[float] | None = None
    p_fit_raw: list[float] | None = None
    p_fit_scaled: list[float] | None = None
    t_used1: list[float] | None = None
    p_used1: list[float] | None = None
    t_used2: list[float] | None = None
    p_used2: list[float] | None = None
    peak_time: float | None = None
    t_cross: float | None = None
    p_cross: float | None = None
    pad_mean: float | None = None
    ees: float | None = None
    ea: float | None = None
    ees_ea: float | None = None
    esv: float | None = None
    edv: float | None = None
    is_selected: bool = False


class HemodynamicResults(BaseModel):
    esp: float | None = None
    edp: float | None = None
    pmax: float | None = None
    ees: float | None = None
    ea: float | None = None
    sv: float | None = None
    dpdt_max: float | None = None
    dpdt_min: float | None = None
    beta: float | None = None
    tau: float | None = None
    ees_ea: float | None = None
    eed: float | None = None
    eed_linear: float | None = None
    esv: float | None = None
    edv: float | None = None
    pmax_scale: float = 1.0


class SingleBeatAnalysisResponse(BaseModel):
    session_id: str
    hemodynamics: HemodynamicResults
    pmax_methods: list[PmaxMethodResultSchema]
    selected_method: str
    event_marker_url: str | None = None
    pressure_plot_url: str | None = None
    status: str


class SessionStatusResponse(BaseModel):
    session_id: str
    step_status: dict[str, str]
    has_image: bool
    has_mask: bool
    has_calibration: bool
    has_beats: bool
    has_average: bool
    has_analysis: bool
