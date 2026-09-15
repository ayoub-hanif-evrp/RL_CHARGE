"""Method-independent feasibility shield.

The shield masks illegal simulator transitions. It does not choose when to
charge, and it does not claim global charging-schedule existence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from data.models import NodeType
from simulation.actions import ChargeAction, ContinueAction
from simulation.feasibility import InfeasibilityReason
from simulation.simulator import FixedRouteSimulator


CONTINUE_INDEX = 0


@dataclass(frozen=True)
class SocInterval:
    soc_lower: float
    soc_upper: float

    def map_u(self, u: float) -> float:
        u = min(1.0, max(0.0, float(u)))
        return self.soc_lower + u * (self.soc_upper - self.soc_lower)


@dataclass(frozen=True)
class ShieldDecision:
    mask: Tuple[bool, ...]
    station_ids: Tuple[str, ...]
    continue_legal: bool
    continue_reason: Optional[InfeasibilityReason]
    station_reasons: Tuple[Optional[InfeasibilityReason], ...]

    @property
    def any_legal(self) -> bool:
        return any(self.mask)

    def station_index(self, station_id: str) -> int:
        return 1 + self.station_ids.index(station_id)


def station_ids_of(simulator: FixedRouteSimulator) -> Tuple[str, ...]:
    return tuple(node.string_id for node in simulator.instance.stations)


def evaluate_shield(simulator: FixedRouteSimulator) -> ShieldDecision:
    stations = station_ids_of(simulator)
    continue_legal, continue_reason = _continue_legal(simulator)
    station_mask: List[bool] = []
    station_reasons: List[Optional[InfeasibilityReason]] = []
    for station_id in stations:
        legal, reason = _station_legal(simulator, station_id)
        station_mask.append(legal)
        station_reasons.append(reason)
    mask = (continue_legal, *station_mask)
    return ShieldDecision(
        mask=mask,
        station_ids=stations,
        continue_legal=continue_legal,
        continue_reason=continue_reason,
        station_reasons=tuple(station_reasons),
    )


def soc_interval_for_station(
    simulator: FixedRouteSimulator,
    station_id: str,
    mode: str = "arrival_to_max",
) -> SocInterval:
    """Transition-level interval. Not claimed globally exact.

    ``mode='arrival_to_max'`` is the shield interval (cannot discharge).
    ``mode='min_to_max'`` is ablation A2: drop the arrival-SOC lower bound
    while still clipping to the battery's physical [min_soc, max_soc].
    """
    arrival = _arrival_soc(simulator, station_id)
    battery = simulator.battery_model.state_from_soc(arrival)
    soc_upper = float(battery.max_soc_fraction)
    if mode == "min_to_max":
        soc_lower = float(battery.min_soc_fraction)
    else:
        soc_lower = float(arrival)
    if soc_lower > soc_upper:
        soc_lower = soc_upper
    return SocInterval(soc_lower=soc_lower, soc_upper=soc_upper)


def map_u_to_target_soc(
    simulator: FixedRouteSimulator,
    station_id: str,
    u: float,
    mode: str = "arrival_to_max",
) -> float:
    target = soc_interval_for_station(simulator, station_id, mode=mode).map_u(u)
    # Physical validity: never request a discharge.
    current = _arrival_soc(simulator, station_id)
    return max(target, current)


def action_from_discrete(
    simulator: FixedRouteSimulator,
    discrete_index: int,
    u: float = 0.0,
    soc_mode: str = "arrival_to_max",
) -> ContinueAction | ChargeAction:
    decision = evaluate_shield(simulator)
    if discrete_index == CONTINUE_INDEX:
        return ContinueAction()
    station_id = decision.station_ids[discrete_index - 1]
    target = map_u_to_target_soc(simulator, station_id, u, mode=soc_mode)
    return ChargeAction(station_id, target)


def _continue_legal(
    simulator: FixedRouteSimulator,
) -> Tuple[bool, Optional[InfeasibilityReason]]:
    probe = simulator.clone()
    result = probe.step(ContinueAction())
    if result.feasible:
        return True, None
    return False, result.reason


def _station_legal(
    simulator: FixedRouteSimulator, station_id: str
) -> Tuple[bool, Optional[InfeasibilityReason]]:
    if simulator.state.n_station_visits_since_last_customer >= (
        simulator.profile.loop_guard_station_visits
    ):
        return False, InfeasibilityReason.LOOP_GUARD
    try:
        node = simulator.network.node(station_id)
    except KeyError:
        return False, InfeasibilityReason.UNKNOWN_STATION
    if node.node_type is not NodeType.STATION:
        return False, InfeasibilityReason.UNKNOWN_STATION
    if simulator.state.current_node_id == station_id:
        return True, None
    check = simulator.feasibility.can_reach(
        simulator.state.current_node_id,
        station_id,
        simulator.state.payload,
        simulator.battery_model.state_from_soc(simulator.state.soc.value),
    )
    if not check.battery_feasible:
        return False, InfeasibilityReason.INSUFFICIENT_ENERGY
    return True, None


def _arrival_soc(simulator: FixedRouteSimulator, station_id: str) -> float:
    if simulator.state.current_node_id == station_id:
        return float(simulator.state.soc.value)
    check = simulator.feasibility.can_reach(
        simulator.state.current_node_id,
        station_id,
        simulator.state.payload,
        simulator.battery_model.state_from_soc(simulator.state.soc.value),
    )
    applied = simulator.battery_model.apply_arc_energy(
        simulator.battery_model.state_from_soc(simulator.state.soc.value),
        check.energy.net_energy,
    )
    return float(applied.after.soc.value)


def masked_indices(mask: Sequence[bool]) -> Tuple[int, ...]:
    return tuple(i for i, legal in enumerate(mask) if legal)
