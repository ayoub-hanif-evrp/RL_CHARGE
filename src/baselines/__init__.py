"""Named new-simulator baselines. None of these use the deleted Training/ stack."""

from .discrete_ppo import DISCRETE_U_GRID, DiscretePPO, snap_u
from .frvcpy_adapter import FRVCPFixture, evrptwgr_to_frvcp_surrogate, round_trip_fixture
from .frvcpy_solver import FRVCPSolveResult, optimality_gap_percent, solve_native
from .greedy import GreedyFullCharge, GreedyMinimumSufficientCharge
from .legacy_ddqn import SOC_LEVELS, LegacyTwoStageDDQN
from .lookahead import OneStepLookahead

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
