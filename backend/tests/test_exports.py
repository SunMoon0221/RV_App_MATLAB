import pandas as pd

from app.services.exports import save_pmax_summary_csv, save_calibrated_trace_csv
from app.services.pmax_methods import empty_result


def test_calibrated_trace_csv_columns(tmp_path):
    p = tmp_path / "calibrated_trace.csv"
    save_calibrated_trace_csv(p, [0.0, 0.1], [5.0, 10.0])
    df = pd.read_csv(p)
    assert list(df.columns) == ["time_s", "pressure_mmhg"]


def test_pmax_summary_columns(tmp_path):
    p = tmp_path / "pmax_method_summary.csv"
    methods = [empty_result("A", "a", "sine", True, "ok")]
    save_pmax_summary_csv(p, methods)
    df = pd.read_csv(p)
    assert "name" in df.columns and "raw_pmax" in df.columns
