import numpy as np

from app.services.pmax_methods import (
    METHOD_NAMES,
    empty_result,
    run_all_pmax_methods,
)


def _wave():
    fs = 500.0
    t = np.linspace(0, 0.8, int(0.8 * fs))
    p = 8 + 22 * np.sin(np.pi * t / 0.25) ** 2 * (t < 0.5)
    p += 8
    return t, p


def test_result_schema_identical_success_failure():
    ok = empty_result("A", "a", "sine", True, "ok")
    bad = empty_result("B", "b", "sine", False, "fail")
    assert set(ok.keys()) == set(bad.keys())
    assert ok["success"] and not bad["success"]


def test_failed_method_does_not_crash_analysis():
    t, p = _wave()
    results = run_all_pmax_methods(t, p, 1.0, "Original Piecewise Sinusoid")
    assert len(results) == 5
    assert all(set(r.keys()) == set(empty_result("", "", "", False, "").keys()) for r in results)


def test_sine_methods_use_window_points():
    t, p = _wave()
    results = run_all_pmax_methods(t, p)
    for r in results:
        if r.get("kind") == "sine" and r.get("success"):
            assert r.get("p_used1") is not None
            assert len(r["p_used1"]) >= 5
            assert r.get("t_used1") is not None


def test_all_method_names_present():
    t, p = _wave()
    results = run_all_pmax_methods(t, p)
    names = {r["name"] for r in results}
    for n in METHOD_NAMES:
        assert n in names
