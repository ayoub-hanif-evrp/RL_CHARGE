"""Evaluate a policy on frozen routes. Reports completion time, energy, and distance separately."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional

from domain.load_convention import LoadConvention, require_load_convention
from physics.parameters import PhysicsProfile
from routing.fixed_route import FrozenRoute
from simulation.metrics import CompletionTimeObjective
from simulation.simulator import FixedRouteSimulator

from .metadata import experiment_metadata


@dataclass
class EpisodeResult:
    route_id: str
    feasible: bool
    completed: bool
    route_completion_time: float
    total_net_energy: float
    total_distance: float
    n_station_visits: int
    return_value: float
    extra: Dict[str, Any] = field(default_factory=dict)
    total_travel_time: float = 0.0
    total_waiting_at_customers: float = 0.0
    total_charging_time: float = 0.0
    total_energy_consumed: float = 0.0
    total_energy_regenerated: float = 0.0
    total_energy_charged: float = 0.0
    terminal_soc: float = 0.0
    reason: Optional[str] = None
    runtime_s: float = 0.0
    horizon: float = 0.0

    def completion_time_all_routes(self) -> float:
        if self.feasible and self.completed:
            return self.route_completion_time
        return float(self.horizon)

    def to_record(self, **context: Any) -> dict[str, Any]:
        payload = {
            "route_id": self.route_id,
            "feasible": self.feasible,
            "completed": self.completed,
            "reason": self.reason,
            "route_completion_time": self.route_completion_time,
            "completion_time_all_routes": self.completion_time_all_routes(),
            "total_distance": self.total_distance,
            "total_travel_time": self.total_travel_time,
            "total_waiting_at_customers": self.total_waiting_at_customers,
            "total_charging_time": self.total_charging_time,
            "total_net_energy": self.total_net_energy,
            "total_energy_consumed": self.total_energy_consumed,
            "total_energy_regenerated": self.total_energy_regenerated,
            "total_energy_charged": self.total_energy_charged,
            "n_station_visits": self.n_station_visits,
            "terminal_soc": self.terminal_soc,
            "return_value": self.return_value,
            "runtime_s": self.runtime_s,
            "horizon": self.horizon,
        }
        payload.update(context)
        return payload


def evaluate_policy(
    *,
    instance,
    route: FrozenRoute,
    policy: Callable,
    load_convention=LoadConvention.OFFICIAL_REFERENCE_PICKUP,
    profile_name: str = "official_evrptwgr",
    eval_mode: bool = True,
    charging_model=None,
    min_soc_fraction: Optional[float] = None,
) -> EpisodeResult:
    load_convention = require_load_convention(load_convention)
    profile = PhysicsProfile.from_instance(instance, name=profile_name)
    if min_soc_fraction is not None:
        profile = replace(profile, min_soc_fraction=float(min_soc_fraction))
    simulator = FixedRouteSimulator(
        instance,
        route.customer_ids,
        profile,
        load_convention,
        charging_model=charging_model,
    )
    started = time.perf_counter()
    result = policy(simulator, eval_mode=eval_mode)
    runtime = time.perf_counter() - started
    metrics = simulator.metrics(feasible=simulator.state.completed)
    reason = getattr(result, "extra", {}) or {}
    reason_s = None
    if not getattr(result, "feasible", simulator.state.completed):
        inner = reason.get("reason") if isinstance(reason, dict) else None
        fail = getattr(result, "reason", None)
        if fail is not None:
            reason_s = getattr(fail, "value", str(fail))
        elif inner is not None:
            reason_s = getattr(inner, "value", str(inner))
        elif metrics.infeasibility_reason is not None:
            reason_s = metrics.infeasibility_reason.value
    return EpisodeResult(
        route_id=route.route_id,
        feasible=bool(getattr(result, "feasible", simulator.state.completed)),
        completed=simulator.state.completed,
        route_completion_time=CompletionTimeObjective.value(metrics).value,
        total_net_energy=metrics.total_net_energy.value,
        total_distance=metrics.total_distance.value,
        n_station_visits=metrics.number_of_station_visits,
        return_value=float(getattr(result, "return_value", -metrics.route_completion_time.value)),
        extra=dict(getattr(result, "extra", {}) or {}),
        total_travel_time=metrics.total_travel_time.value,
        total_waiting_at_customers=metrics.total_waiting_at_customers.value,
        total_charging_time=metrics.total_charging_time.value,
        total_energy_consumed=metrics.total_energy_consumed.value,
        total_energy_regenerated=metrics.total_energy_regenerated.value,
        total_energy_charged=metrics.total_energy_charged.value,
        terminal_soc=metrics.terminal_soc.value,
        reason=reason_s,
        runtime_s=runtime,
        horizon=float(instance.depot.due_date),
    )
