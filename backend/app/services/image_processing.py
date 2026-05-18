"""Image loading, thresholding, mask composition, and median-row extraction."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import morphology

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def load_image_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to BGR uint8 array."""
    if cv2 is None:
        raise RuntimeError("OpenCV required for image decoding")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """Robust grayscale conversion (luminance)."""
    if cv2 is None:
        return np.mean(bgr, axis=2).astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def apply_notch_filter(
    gray: np.ndarray, notch_freq: float = 50.0, sample_rate: float = 500.0
) -> np.ndarray:
    """
    Optional 1D notch along rows for periodic screen artifacts.
    Applied per-row in frequency domain when enabled.
    """
    out = gray.astype(np.float64).copy()
    n = gray.shape[1]
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    for i in range(gray.shape[0]):
        row = out[i, :]
        spec = np.fft.fft(row)
        idx = np.argmin(np.abs(freqs - notch_freq))
        spec[idx] = 0
        if idx > 0:
            spec[-idx] = 0
        out[i, :] = np.real(np.fft.ifft(spec))
    return np.clip(out, 0, 255).astype(np.uint8)


def threshold_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Binary mask: True where trace ink is present (dark pixels below threshold)."""
    return gray < threshold


def clean_mask(mask: np.ndarray, min_size: int = 30) -> np.ndarray:
    """Morphological open/close and remove small objects."""
    cleaned = morphology.remove_small_objects(mask, min_size=min_size)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(2))
    cleaned = morphology.binary_opening(cleaned, morphology.disk(1))
    return cleaned


def compose_filtered_mask(
    base: np.ndarray, manual_erase: np.ndarray, trace_keep: np.ndarray
) -> np.ndarray:
    """
    filtered = base & manual_erase & trace_keep
  All masks are boolean same shape.
    """
    return base & manual_erase & trace_keep


def extract_median_rows(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each column x, median row index of mask pixels.
    Returns (x_pixels int array, y_pixels float array); NaN where empty column.
    """
    h, w = mask.shape
    x_pixels = np.arange(w, dtype=np.int32)
    y_pixels = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        rows = np.where(mask[:, x])[0]
        if rows.size:
            y_pixels[x] = float(np.median(rows))
    return x_pixels, y_pixels


def rebuild_edges_from_mask(mask: np.ndarray) -> np.ndarray:
    """Edge/cleaned mask via morphological gradient."""
    return morphology.binary_dilation(mask) ^ morphology.binary_erosion(mask)


def mask_to_overlay_rgba(mask: np.ndarray, color: tuple[int, int, int, int] = (0, 255, 0, 120)) -> np.ndarray:
    """RGBA overlay for preview."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask] = color
    return rgba


def rethreshold_preserving_masks(
    gray: np.ndarray,
    threshold: int,
    manual_erase: np.ndarray,
    trace_keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute base mask from gray; preserve user erase/trace masks."""
    base = clean_mask(threshold_mask(gray, threshold))
    filtered = compose_filtered_mask(base, manual_erase, trace_keep)
    return base, filtered
