"""Fixed-route charging simulator. No Gym, no PyTorch, no learned policy."""

from .actions import Action, ChargeAction, ContinueAction
from .events import ChargeEvent, CustomerServiceEvent, DepotEvent, SimulatorEvent, TravelEvent
from .feasibility import FeasibilityService, InfeasibilityReason, ReachabilityCheck
from .metrics import (
    CompletionTimeObjective,
    EnergyObjective,
    MetricsAccumulator,
    TrajectoryMetrics,
)
from .shield import ShieldDecision, SocInterval, evaluate_shield, map_u_to_target_soc
from .simulator import FixedRouteSimulator, TransitionResult, run_continue_only
from .state import SimulatorState

__all__ = [
    "Action",
    "ChargeAction",
    "ChargeEvent",
    "CompletionTimeObjective",
    "ContinueAction",
    "CustomerServiceEvent",
    "DepotEvent",
    "EnergyObjective",
    "FeasibilityService",
    "FixedRouteSimulator",
    "InfeasibilityReason",
    "MetricsAccumulator",
    "ReachabilityCheck",
    "ShieldDecision",
    "SocInterval",
    "SimulatorEvent",
    "SimulatorState",
    "TrajectoryMetrics",
    "TransitionResult",
    "TravelEvent",
    "evaluate_shield",
    "map_u_to_target_soc",
    "run_continue_only",
]
