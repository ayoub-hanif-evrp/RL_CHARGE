"""Official e-VRO/frvcpy reference routes (testdata.json). No torch.

Primary reviewer-facing exact FRVCP benchmark. Sequential node-ID tours are
not official published routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from data.paths import EXTERNAL_DIR

from .frvcpy_solver import frvcpy_available, load_instance_json, solve_native

# unittest.TestCase.assertAlmostEqual(..., places=3) ⇒ |a-b| < 5e-4
UPSTREAM_ABS_TOL = 5e-4
OFFICIAL_INSTANCE = "frvcpy-instance.json"
OFFICIAL_TESTDATA = "testdata.json"
EXPECTED_REFERENCE_ROUTES = 133


def reference_dir() -> Path:
    return EXTERNAL_DIR / "frvcpy"


def load_testdata(root: Optional[Path] = None) -> Dict[str, dict]:
    path = (root or reference_dir()) / OFFICIAL_TESTDATA
    return json.loads(path.read_text(encoding="utf-8"))


def load_official_instance(root: Optional[Path] = None) -> dict:
    path = (root or reference_dir()) / OFFICIAL_INSTANCE
    return load_instance_json(path)


def official_reference_specs(root: Optional[Path] = None) -> List[dict]:
    root = root or reference_dir()
    testdata = load_testdata(root)
    instance = load_official_instance(root)
    q_init = float(instance["max_q"])
    specs = []
    for route_id, info in testdata.items():
        specs.append(
            {
                "route_id": route_id,
                "instance": OFFICIAL_INSTANCE,
                "nodes": list(info["route"]),
                "q_init": q_init,
                "known_obj": float(info["obj"]),
                "source": "e-VRO/frvcpy testdata.json",
                "official_published_tour": True,
                "equivalent_to_evrptwgr": "native_frvcp",
            }
        )
    return specs


def parity_against_testdata(root: Optional[Path] = None) -> dict:
    """Solve every official reference route with real frvcpy.Solver."""
    root = root or reference_dir()
    if not frvcpy_available():
        return {
            "available": False,
            "n_routes": 0,
            "n_ok": 0,
            "max_abs_error": None,
            "errors": ["frvcpy_not_installed"],
        }
    instance = load_official_instance(root)
    specs = official_reference_specs(root)
    max_abs = 0.0
    n_ok = 0
    errors = []
    for spec in specs:
        solved = solve_native(instance, spec["nodes"], spec["q_init"])
        known = spec["known_obj"]
        if solved.status != "optimal" or solved.duration is None:
            errors.append(f"{spec['route_id']}: {solved.status}")
            continue
        abs_err = abs(float(solved.duration) - known)
        if abs_err > max_abs:
            max_abs = abs_err
        if abs_err > UPSTREAM_ABS_TOL:
            errors.append(f"{spec['route_id']}: |{solved.duration}-{known}|={abs_err}")
        else:
            n_ok += 1
    return {
        "available": True,
        "n_routes": len(specs),
        "n_ok": n_ok,
        "max_abs_error": max_abs,
        "errors": errors,
        "tolerance": UPSTREAM_ABS_TOL,
    }
