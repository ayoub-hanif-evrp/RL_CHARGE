"""Method-independent feasibility shield.

The shield masks illegal simulator transitions. It does not choose when to
charge, and it does not claim global charging-schedule existence.

Station SOC intervals use an *energy-continuation* lower bound: the minimum
departure SOC that can take a first hop onto a payload-dependent station graph
that can still reach the next frozen customer/depot with intermediate
recharges allowed. Time windows and charging duration are **not** included;
the bound is not globally exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from data.models import NodeType
from simulation.actions import ChargeAction, ContinueAction
from simulation.feasibility import InfeasibilityReason, ZERO_CHARGE_EPS
from simulation.simulator import FixedRouteSimulator


CONTINUE_INDEX = 0
CONTINUATION_TO_MAX = "continuation_to_max"
ARRIVAL_TO_MAX = "arrival_to_max"


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
    continuable = energy_continuable_stations(simulator)
    station_mask: List[bool] = []
    station_reasons: List[Optional[InfeasibilityReason]] = []
    for station_id in stations:
        legal, reason = _station_legal(simulator, station_id, continuable)
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
    mode: str = CONTINUATION_TO_MAX,
) -> SocInterval:
    """Transition-level interval. Not claimed globally exact.

    ``continuation_to_max`` (FULL): lower bound is max(arrival, energy-continuation
    minimum departure SOC).
    ``arrival_to_max`` (A2): physical validity only — [arrival, max].
    """
    arrival = _arrival_soc(simulator, station_id)
    battery = simulator.battery_model.state_from_soc(arrival)
    soc_upper = float(battery.max_soc_fraction)
    if mode == ARRIVAL_TO_MAX:
        soc_lower = float(arrival)
    else:
        bound = continuation_departure_soc(simulator, station_id)
        if bound is None:
            soc_lower = float(arrival)
        else:
            soc_lower = max(float(arrival), float(bound))
    if soc_lower > soc_upper:
        soc_lower = soc_upper
    return SocInterval(soc_lower=soc_lower, soc_upper=soc_upper)


def map_u_to_target_soc(
    simulator: FixedRouteSimulator,
    station_id: str,
    u: float,
    mode: str = CONTINUATION_TO_MAX,
) -> float:
    target = soc_interval_for_station(simulator, station_id, mode=mode).map_u(u)
    current = _arrival_soc(simulator, station_id)
    return max(target, current)


def action_from_discrete(
    simulator: FixedRouteSimulator,
    discrete_index: int,
    u: float = 0.0,
    soc_mode: str = CONTINUATION_TO_MAX,
) -> ContinueAction | ChargeAction:
    decision = evaluate_shield(simulator)
    if discrete_index == CONTINUE_INDEX:
        return ContinueAction()
    station_id = decision.station_ids[discrete_index - 1]
    target = map_u_to_target_soc(simulator, station_id, u, mode=soc_mode)
    return ChargeAction(station_id, target)


def min_departure_soc_for_arc(
    simulator: FixedRouteSimulator, from_id: str, to_id: str
) -> Optional[float]:
    """Minimum start SOC so the directed arc stays at or above the SOC floor.

    Uses the same EnergyModel and BatteryModel as the simulator. Not a TW bound.
    """
    payload = simulator.state.payload
    energy = simulator.energy_model.energy_for_arc(from_id, to_id, payload)
    battery = simulator.battery_model.state_from_soc(simulator.profile.max_soc_fraction)
    min_e = battery.min_energy.value
    q = battery.capacity.value
    required_energy = min_e + energy.net_energy.value
    required_soc = required_energy / q
    min_soc = float(battery.min_soc_fraction)
    max_soc = float(battery.max_soc_fraction)
    soc = max(min_soc, required_soc)
    if soc > max_soc + 1e-12:
        return None
    check = simulator.battery_model.apply_arc_energy(
        simulator.battery_model.state_from_soc(soc), energy.net_energy
    )
    if not check.feasible:
        return None
    return float(soc)


def continuation_hop_ranks(simulator: FixedRouteSimulator) -> Dict[str, int]:
    """Backward hop ranks from the next frozen customer/depot.

    Rank 0 is the next frozen node (not a station key). Rank 1 stations can
    reach it at full battery. Rank ``k+1`` stations can reach at least one
    station of rank ``k`` or lower. Stations missing from the result have no
    finite rank and are not energy-continuable.

    Energy only: time windows and charging duration are ignored.
    """
    nxt = simulator.next_frozen_node_id()
    stations = list(station_ids_of(simulator))
    ranks: Dict[str, int] = {}
    for station_id in stations:
        if _full_battery_reach(simulator, station_id, nxt):
            ranks[station_id] = 1
    k = 1
    while True:
        added = []
        for station_id in stations:
            if station_id in ranks:
                continue
            if any(_full_battery_reach(simulator, station_id, other) for other in ranks):
                added.append(station_id)
        if not added:
            break
        k += 1
        for station_id in added:
            ranks[station_id] = k
    return ranks


def energy_continuable_stations(simulator: FixedRouteSimulator) -> Set[str]:
    """Stations with a finite continuation hop rank. Time windows are ignored."""
    return set(continuation_hop_ranks(simulator))


def continuation_departure_soc(simulator: FixedRouteSimulator, station_id: str) -> Optional[float]:
    """Min departure SOC at ``station_id`` for a rank-decreasing first hop.

    A station action is continuation-valid only with a finite hop rank. For
    rank ``r``, the bound is the cheapest feasible first hop to the next frozen
    node or to a station of strictly lower rank. Station cycles are excluded
    without assuming Euclidean progress.

    Energy only: not a globally exact time-window feasibility guarantee.
    """
    nxt = simulator.next_frozen_node_id()
    ranks = continuation_hop_ranks(simulator)
    rank = ranks.get(station_id)
    if rank is None:
        return None
    candidates: List[float] = []
    direct = min_departure_soc_for_arc(simulator, station_id, nxt)
    if direct is not None:
        candidates.append(direct)
    for other, other_rank in ranks.items():
        if other == station_id or other_rank >= rank:
            continue
        hop = min_departure_soc_for_arc(simulator, station_id, other)
        if hop is not None:
            candidates.append(hop)
    if not candidates:
        return None
    return float(min(candidates))


def _full_battery_reach(simulator: FixedRouteSimulator, from_id: str, to_id: str) -> bool:
    battery = simulator.battery_model.state_from_soc(simulator.profile.max_soc_fraction)
    return simulator.feasibility.can_reach(
        from_id, to_id, simulator.state.payload, battery
    ).battery_feasible


def _continue_legal(
    simulator: FixedRouteSimulator,
) -> Tuple[bool, Optional[InfeasibilityReason]]:
    probe = simulator.clone()
    result = probe.step(ContinueAction())
    if result.feasible:
        return True, None
    return False, result.reason


def _station_legal(
    simulator: FixedRouteSimulator,
    station_id: str,
    continuable: Set[str],
) -> Tuple[bool, Optional[InfeasibilityReason]]:
    if simulator.state.n_station_visits_since_last_customer >= (
        simulator.profile.loop_guard_station_visits
    ):
        return False, InfeasibilityReason.LOOP_GUARD
    if station_id in simulator.state.stations_visited_since_progress:
        return False, InfeasibilityReason.STATION_REVISIT
    try:
        node = simulator.network.node(station_id)
    except KeyError:
        return False, InfeasibilityReason.UNKNOWN_STATION
    if node.node_type is not NodeType.STATION:
        return False, InfeasibilityReason.UNKNOWN_STATION
    at_station = simulator.state.current_node_id == station_id
    if at_station:
        room = simulator.profile.max_soc_fraction - simulator.state.soc.value
        if room < ZERO_CHARGE_EPS:
            return False, InfeasibilityReason.ZERO_CHARGE_NOOP
    else:
        check = simulator.feasibility.can_reach(
            simulator.state.current_node_id,
            station_id,
            simulator.state.payload,
            simulator.battery_model.state_from_soc(simulator.state.soc.value),
        )
        if not check.battery_feasible:
            return False, InfeasibilityReason.INSUFFICIENT_ENERGY
    if station_id not in continuable:
        return False, InfeasibilityReason.NO_ENERGY_CONTINUATION
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
