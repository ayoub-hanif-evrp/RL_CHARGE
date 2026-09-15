"""Fixed-route charging simulator. No Gym, no PyTorch, no learned policy."""

from .actions import Action, ChargeAction, ContinueAction
from .events import ChargeEvent, CustomerServiceEvent, DepotEvent, SimulatorEvent, TravelEvent
from .feasibility import FeasibilityService, InfeasibilityReason, ReachabilityCheck
from .metrics import EnergyObjective, MetricsAccumulator, TrajectoryMetrics
from .simulator import FixedRouteSimulator, TransitionResult, run_continue_only
from .state import SimulatorState

__all__ = [
    "Action",
    "ChargeAction",
    "ChargeEvent",
    "ContinueAction",
    "CustomerServiceEvent",
    "DepotEvent",
    "EnergyObjective",
    "FeasibilityService",
    "FixedRouteSimulator",
    "InfeasibilityReason",
    "MetricsAccumulator",
    "ReachabilityCheck",
    "SimulatorEvent",
    "SimulatorState",
    "TrajectoryMetrics",
    "TransitionResult",
    "TravelEvent",
    "run_continue_only",
]
