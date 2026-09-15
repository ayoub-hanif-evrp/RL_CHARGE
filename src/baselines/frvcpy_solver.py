"""Call frvcpy's Froger labeling solver on native FRVCP instances.

Exact for FRVCP with a given energy matrix and published piecewise charging.
Not exact for EVRPTW-GR.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class FRVCPSolveResult:
    status: str
    duration: Optional[float]
    feasible_route: Any
    runtime_s: float
    equivalent_to_evrptwgr: str = "native_frvcp"


def frvcpy_available() -> bool:
    try:
        import frvcpy  # noqa: F401

        return True
    except ImportError:
        return False


def solve_native(instance_json, route: List[int], q_init: float) -> FRVCPSolveResult:
    import time

    started = time.perf_counter()
    try:
        from frvcpy.solver import Solver
    except ImportError:
        return FRVCPSolveResult(
            status="frvcpy_not_installed",
            duration=None,
            feasible_route=None,
            runtime_s=time.perf_counter() - started,
        )
    try:
        solver = Solver(instance_json, route, q_init)
        duration, feas_route = solver.solve()
        status = "optimal" if duration is not None else "infeasible"
        return FRVCPSolveResult(
            status=status,
            duration=None if duration is None else float(duration),
            feasible_route=feas_route,
            runtime_s=time.perf_counter() - started,
        )
    except Exception as exc:
        return FRVCPSolveResult(
            status=f"solver_error:{type(exc).__name__}",
            duration=None,
            feasible_route=None,
            runtime_s=time.perf_counter() - started,
        )


def load_instance_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def optimality_gap_percent(method: float, optimum: float) -> Optional[float]:
    if optimum is None or optimum <= 0 or method is None:
        return None
    return 100.0 * (method - optimum) / optimum
