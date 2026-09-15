"""Named new-simulator baselines. Imports are lazy so native FRVCP code can
run without torch.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DISCRETE_U_GRID",
    "FRVCPFixture",
    "FRVCPSolveResult",
    "GreedyFullCharge",
    "GreedyMinimumSufficientCharge",
    "LegacyTwoStageDDQN",
    "OneStepLookahead",
    "SOC_LEVELS",
    "DiscretePPO",
    "evrptwgr_to_frvcp_surrogate",
    "optimality_gap_percent",
    "round_trip_fixture",
    "snap_u",
    "solve_native",
]


def __getattr__(name: str) -> Any:
    if name in ("DISCRETE_U_GRID", "DiscretePPO", "snap_u"):
        from .discrete_ppo import DISCRETE_U_GRID, DiscretePPO, snap_u

        return {"DISCRETE_U_GRID": DISCRETE_U_GRID, "DiscretePPO": DiscretePPO, "snap_u": snap_u}[name]
    if name in ("SOC_LEVELS", "LegacyTwoStageDDQN"):
        from .legacy_ddqn import SOC_LEVELS, LegacyTwoStageDDQN

        return {"SOC_LEVELS": SOC_LEVELS, "LegacyTwoStageDDQN": LegacyTwoStageDDQN}[name]
    if name in ("GreedyFullCharge", "GreedyMinimumSufficientCharge"):
        from .greedy import GreedyFullCharge, GreedyMinimumSufficientCharge

        return {
            "GreedyFullCharge": GreedyFullCharge,
            "GreedyMinimumSufficientCharge": GreedyMinimumSufficientCharge,
        }[name]
    if name == "OneStepLookahead":
        from .lookahead import OneStepLookahead

        return OneStepLookahead
    if name in ("FRVCPFixture", "evrptwgr_to_frvcp_surrogate", "round_trip_fixture"):
        from .frvcpy_adapter import FRVCPFixture, evrptwgr_to_frvcp_surrogate, round_trip_fixture

        return {
            "FRVCPFixture": FRVCPFixture,
            "evrptwgr_to_frvcp_surrogate": evrptwgr_to_frvcp_surrogate,
            "round_trip_fixture": round_trip_fixture,
        }[name]
    if name in ("FRVCPSolveResult", "optimality_gap_percent", "solve_native"):
        from .frvcpy_solver import FRVCPSolveResult, optimality_gap_percent, solve_native

        return {
            "FRVCPSolveResult": FRVCPSolveResult,
            "optimality_gap_percent": optimality_gap_percent,
            "solve_native": solve_native,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
