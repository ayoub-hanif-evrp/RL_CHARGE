"""Simulator-valid charging certificates for a frozen customer sequence.

A positive result is one feasible charging trajectory. A timeout or an
exhausted search is unverified. This search is not claimed to be globally exact,
and search failure is never reported as infeasibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from heapq import heappop, heappush
from time import perf_counter
from typing import Optional, Sequence

from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from routing.serialize import canonical_dumps
from simulation.actions import ChargeAction, ContinueAction
from simulation.events import ChargeEvent
from simulation.feasibility import ZERO_CHARGE_EPS
from simulation.shield import continuation_departure_soc, min_departure_soc_for_arc
from simulation.simulator import FixedRouteSimulator

STATUS_CERTIFIED = "certified_feasible"
STATUS_TIMEOUT = "unverified_timeout"
STATUS_EXHAUSTED = "unverified_search_exhausted"


@dataclass
class ChargingCertificate:
    status: str
    feasible: bool
    actions: tuple = ()
    trace: list = field(default_factory=list)
    completion_time: Optional[float] = None
    n_station_visits: int = 0
    n_expansions: int = 0
    runtime_s: float = 0.0
    sha256: str = ""
    exact: bool = False
    customer_ids: tuple = ()

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "feasible": self.feasible,
            "actions": [list(step) for step in self.actions],
            "trace": self.trace,
            "completion_time": self.completion_time,
            "n_station_visits": self.n_station_visits,
            "n_expansions": self.n_expansions,
            "runtime_s": self.runtime_s,
            "sha256": self.sha256,
            "exact": self.exact,
        }


def _priority(sim: FixedRouteSimulator) -> tuple:
    """Expand labels closer to the depot first, then earlier time, then higher SOC."""
    if sim.state.completed:
        remaining = -1
    else:
        remaining = len(sim.customer_ids) - int(sim.state.next_customer_index)
    # Higher SOC before lower time: charge-to-max witnesses are found before tiny charges.
    return (remaining, -float(sim.state.soc.value), float(sim.state.time.value))


def _discrete_key(sim: FixedRouteSimulator) -> tuple:
    state = sim.state
    visited = tuple(sorted(state.stations_visited_since_progress))
    return (state.next_customer_index, state.current_node_id, visited)


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[0] + 1e-9 and a[1] >= b[1] - 1e-9


def _pareto_accept(frontier: list, resources: tuple[float, float]) -> bool:
    for prev in frontier:
        if _dominates(prev, resources):
            return False
    kept = [prev for prev in frontier if not _dominates(resources, prev)]
    kept.append(resources)
    frontier.clear()
    frontier.extend(kept)
    return True


def _arrival_soc(sim: FixedRouteSimulator, station_id: str) -> Optional[float]:
    if sim.state.current_node_id == station_id:
        return float(sim.state.soc.value)
    battery = sim.battery_model.state_from_soc(sim.state.soc.value)
    check = sim.feasibility.can_reach(
        sim.state.current_node_id, station_id, sim.state.payload, battery
    )
    if not check.battery_feasible:
        return None
    applied = sim.battery_model.apply_arc_energy(battery, check.energy.net_energy)
    return float(applied.after.soc.value)


def target_soc_candidates(sim: FixedRouteSimulator, station_id: str) -> list[float]:
    """Physically meaningful target SOCs inside the reachable charging interval."""
    arrival = _arrival_soc(sim, station_id)
    if arrival is None:
        return []
    battery = sim.battery_model.state_from_soc(arrival)
    lower = float(arrival)
    upper = float(battery.max_soc_fraction)
    raw = [lower, upper]
    nxt = sim.next_frozen_node_id()
    direct = min_departure_soc_for_arc(sim, station_id, nxt)
    if direct is not None:
        raw.append(float(direct))
    visited = sim.state.stations_visited_since_progress
    for other in sim.instance.stations:
        other_id = other.string_id
        if other_id == station_id or other_id in visited:
            continue
        hop = min_departure_soc_for_arc(sim, station_id, other_id)
        if hop is not None:
            raw.append(float(hop))
    continuation = continuation_departure_soc(sim, station_id)
    if continuation is not None:
        raw.append(float(continuation))
    deduped: list[float] = []
    for value in raw:
        clamped = min(upper, max(lower, float(value)))
        if clamped < lower + ZERO_CHARGE_EPS:
            continue
        if any(abs(clamped - prev) <= 1e-8 for prev in deduped):
            continue
        deduped.append(clamped)
    return deduped


def _station_reachable(sim: FixedRouteSimulator, station_id: str) -> bool:
    if station_id in sim.state.stations_visited_since_progress:
        return False
    if sim.state.n_station_visits_since_last_customer >= sim.profile.loop_guard_station_visits:
        return False
    return bool(target_soc_candidates(sim, station_id))


def trace_from_replay(sim: FixedRouteSimulator, n_events_before: int, kind: str, station_id: str, target_soc: float) -> dict:
    new_events = sim.state.events[n_events_before:]
    charge = next((event for event in new_events if isinstance(event, ChargeEvent)), None)
    if charge is None:
        return {
            "kind": kind,
            "station_id": None if kind == "CONTINUE" else station_id,
            "target_soc": None if kind == "CONTINUE" else float(target_soc),
            "arrival_soc": float(sim.state.soc.value),
            "departure_soc": float(sim.state.soc.value),
            "charge_amount": 0.0,
            "charge_time": 0.0,
            "time_after": float(sim.state.time.value),
            "soc_after": float(sim.state.soc.value),
        }
    return {
        "kind": "CHARGE",
        "station_id": charge.station_id,
        "target_soc": float(charge.soc_target.value),
        "arrival_soc": float(charge.soc_before.value),
        "departure_soc": float(charge.soc_after.value),
        "charge_amount": float(charge.energy_added.value),
        "charge_time": float(charge.charging_duration.value),
        "time_after": float(sim.state.time.value),
        "soc_after": float(sim.state.soc.value),
    }


def replay_action_trace(instance, customer_ids: Sequence[str], actions: Sequence[tuple], profile=None) -> Optional[list]:
    """Replay actions on a fresh simulator. Return the trace only if the route completes."""
    profile = profile or PhysicsProfile.from_instance(instance)
    sim = FixedRouteSimulator(
        instance, tuple(customer_ids), profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    trace = []
    for index, action in enumerate(actions):
        before = len(sim.state.events)
        if action[0] == "CONTINUE":
            result = sim.step(ContinueAction())
            step = trace_from_replay(sim, before, "CONTINUE", "", 0.0)
        else:
            result = sim.step(ChargeAction(str(action[1]), float(action[2])))
            step = trace_from_replay(sim, before, "CHARGE", str(action[1]), float(action[2]))
        if not result.feasible:
            return None
        step["decision_index"] = index
        trace.append(step)
    if not sim.state.completed:
        return None
    return trace


def certificate_sha256(trace: list) -> str:
    payload = canonical_dumps(trace)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def solve_charging_certificate(
    instance,
    customer_ids: Sequence[str],
    *,
    max_expansions: int = 8_000,
    max_seconds: float = 2.0,
    profile=None,
) -> ChargingCertificate:
    """Best-first search for one simulator-valid charging trajectory."""
    customers = tuple(customer_ids)
    profile = profile or PhysicsProfile.from_instance(instance)
    root = FixedRouteSimulator(
        instance, customers, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    heap: list = []
    counter = 0
    heappush(heap, (*_priority(root), counter, root, tuple()))
    frontiers: dict[tuple, list] = {}
    expansions = 0
    started = perf_counter()

    def _finish(status: str, actions: tuple = ()) -> ChargingCertificate:
        runtime = perf_counter() - started
        if status == STATUS_CERTIFIED:
            trace = replay_action_trace(instance, customers, actions, profile=profile)
            if trace is None:
                raise AssertionError("certificate replay failed after search reported a completed trajectory")
            digest = certificate_sha256(trace)
            visits = sum(1 for step in trace if step["kind"] == "CHARGE")
            return ChargingCertificate(
                status=STATUS_CERTIFIED,
                feasible=True,
                actions=tuple(actions),
                trace=trace,
                completion_time=float(trace[-1]["time_after"]) if trace else None,
                n_station_visits=visits,
                n_expansions=expansions,
                runtime_s=runtime,
                sha256=digest,
                exact=False,
                customer_ids=customers,
            )
        return ChargingCertificate(
            status=status,
            feasible=False,
            n_expansions=expansions,
            runtime_s=runtime,
            exact=False,
        )

    while heap:
        if (perf_counter() - started) >= float(max_seconds):
            return _finish(STATUS_TIMEOUT)
        _remaining, _time_val, _neg_soc, _tie, sim, actions = heappop(heap)
        if sim.state.completed:
            return _finish(STATUS_CERTIFIED, actions)
        resources = (float(sim.state.time.value), float(sim.state.soc.value))
        frontier = frontiers.setdefault(_discrete_key(sim), [])
        if not _pareto_accept(frontier, resources):
            continue
        expansions += 1
        if expansions > max_expansions:
            return _finish(STATUS_TIMEOUT)
        probe = sim.clone()
        continued = probe.step(ContinueAction())
        if continued.feasible:
            counter += 1
            heappush(heap, (*_priority(probe), counter, probe, actions + (("CONTINUE",),)))
        for station in sim.instance.stations:
            if (perf_counter() - started) >= float(max_seconds):
                return _finish(STATUS_TIMEOUT)
            station_id = station.string_id
            if not _station_reachable(sim, station_id):
                continue
            for target in target_soc_candidates(sim, station_id):
                nxt = sim.clone()
                result = nxt.step(ChargeAction(station_id, target))
                if not result.feasible:
                    continue
                counter += 1
                heappush(
                    heap,
                    (
                        *_priority(nxt),
                        counter,
                        nxt,
                        actions + (("CHARGE", station_id, float(target)),),
                    ),
                )
    return _finish(STATUS_EXHAUSTED)
