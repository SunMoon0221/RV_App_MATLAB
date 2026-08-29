import numpy as np

from app.services.image_processing import compose_filtered_mask, extract_median_rows, threshold_mask


def test_mask_composition():
    base = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
    erase = np.ones_like(base, dtype=bool)
    erase[0, 0] = False
    trace = np.ones_like(base, dtype=bool)
    trace[2, 2] = False
    filtered = compose_filtered_mask(base, erase, trace)
    assert filtered[0, 0] is np.False_
    assert filtered[2, 2] is np.False_
    assert filtered[0, 1]


def test_median_row_extraction():
    mask = np.zeros((10, 5), dtype=bool)
    mask[3:7, 2] = True
    x, y = extract_median_rows(mask)
    assert y[2] == 4.5  # median of rows 3,4,5,6
    assert np.isnan(y[0])


def test_threshold_dark_trace():
    gray = np.full((5, 5), 200, dtype=np.uint8)
    gray[2, :] = 30
    m = threshold_mask(gray, 128)
    assert m[2, :].all()
