"""Official e-VRO testdata.json Solver parity. Skips if frvcpy is missing."""

import json

import pytest

from baselines.frvcpy_reference import (
    EXPECTED_REFERENCE_ROUTES,
    UPSTREAM_ABS_TOL,
    load_testdata,
    official_reference_specs,
    parity_against_testdata,
)
from baselines.frvcpy_solver import frvcpy_available, solve_native
from data.paths import EXTERNAL_DIR


def test_official_testdata_is_vendored():
    root = EXTERNAL_DIR / "frvcpy"
    assert (root / "testdata.json").is_file()
    assert (root / "frvcpy-instance.json").is_file()
    payload = load_testdata()
    assert len(payload) == EXPECTED_REFERENCE_ROUTES
    specs = official_reference_specs()
    assert len(specs) == EXPECTED_REFERENCE_ROUTES
    assert all(spec["official_published_tour"] for spec in specs)
    assert all("known_obj" in spec for spec in specs)
    tiny = json.loads((root / "routes.json").read_text(encoding="utf-8"))
    assert tiny["official_published_tours"] is False
    bench = json.loads((root / "benchmark" / "routes.json").read_text(encoding="utf-8"))
    assert bench["official_published_tours"] is False
    assert "not" in bench["note"].lower()


def test_frvcpy_reference_parity():
    if not frvcpy_available():
        pytest.skip("frvcpy not installed")
    report = parity_against_testdata()
    assert report["available"]
    assert report["n_routes"] == EXPECTED_REFERENCE_ROUTES
    assert report["n_ok"] == EXPECTED_REFERENCE_ROUTES
    assert report["max_abs_error"] is not None
    assert report["max_abs_error"] <= UPSTREAM_ABS_TOL
    assert report["errors"] == []


def test_tiny_native_frvcp_optional_solver_smoke():
    path = EXTERNAL_DIR / "frvcpy" / "tiny-instance.json"
    instance = json.loads(path.read_text(encoding="utf-8"))
    solved = solve_native(instance, [0, 1, 0], 16.0)
    if not frvcpy_available():
        assert solved.status == "frvcpy_not_installed"
    else:
        assert solved.status in {"optimal", "infeasible", "frvcpy_not_installed"} or solved.status.startswith("solver_error")
