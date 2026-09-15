"""Canonical evaluation metrics. Unlike units are never added together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from domain.quantities import Distance, Energy, SocFraction, TravelTime

from .feasibility import InfeasibilityReason


@dataclass(frozen=True)
class TrajectoryMetrics:
    total_distance: Distance
    total_travel_time: TravelTime
    total_waiting_at_customers: TravelTime
    total_service_time: TravelTime
    total_charging_time: TravelTime
    total_energy_consumed: Energy
    total_energy_regenerated: Energy
    total_net_energy: Energy
    total_energy_charged: Energy
    number_of_station_visits: int
    terminal_soc: SocFraction
    route_completion_time: TravelTime
    time_window_violations: int
    feasible: bool
    infeasibility_reason: Optional[InfeasibilityReason]
    energy_objective: Energy

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_distance": self.total_distance.value,
            "total_travel_time": self.total_travel_time.value,
            "total_waiting_at_customers": self.total_waiting_at_customers.value,
            "total_service_time": self.total_service_time.value,
            "total_charging_time": self.total_charging_time.value,
            "total_energy_consumed": self.total_energy_consumed.value,
            "total_energy_regenerated": self.total_energy_regenerated.value,
            "total_net_energy": self.total_net_energy.value,
            "total_energy_charged": self.total_energy_charged.value,
            "number_of_station_visits": self.number_of_station_visits,
            "terminal_soc": self.terminal_soc.value,
            "route_completion_time": self.route_completion_time.value,
            "time_window_violations": self.time_window_violations,
            "feasible": self.feasible,
            "infeasibility_reason": None
            if self.infeasibility_reason is None
            else self.infeasibility_reason.value,
            "energy_objective": self.energy_objective.value,
        }


class EnergyObjective:
    """EVRPTW-GR Model 2 objective: sum of net arc energies (normalized Schneider units)."""

    name = "EnergyObjective"

    @staticmethod
    def value(metrics: TrajectoryMetrics) -> Energy:
        return metrics.energy_objective


@dataclass
class MetricsAccumulator:
    total_distance: float = 0.0
    total_travel_time: float = 0.0
    total_waiting_at_customers: float = 0.0
    total_service_time: float = 0.0
    total_charging_time: float = 0.0
    total_energy_consumed: float = 0.0
    total_energy_regenerated: float = 0.0
    total_net_energy: float = 0.0
    total_energy_charged: float = 0.0
    number_of_station_visits: int = 0
    time_window_violations: int = 0

    def snapshot(
        self,
        terminal_soc: SocFraction,
        completion_time: TravelTime,
        feasible: bool,
        reason: Optional[InfeasibilityReason],
    ) -> TrajectoryMetrics:
        net = Energy(self.total_net_energy)
        return TrajectoryMetrics(
            total_distance=Distance(self.total_distance),
            total_travel_time=TravelTime(self.total_travel_time),
            total_waiting_at_customers=TravelTime(self.total_waiting_at_customers),
            total_service_time=TravelTime(self.total_service_time),
            total_charging_time=TravelTime(self.total_charging_time),
            total_energy_consumed=Energy(self.total_energy_consumed),
            total_energy_regenerated=Energy(self.total_energy_regenerated),
            total_net_energy=net,
            total_energy_charged=Energy(self.total_energy_charged),
            number_of_station_visits=self.number_of_station_visits,
            terminal_soc=terminal_soc,
            route_completion_time=completion_time,
            time_window_violations=self.time_window_violations,
            feasible=feasible,
            infeasibility_reason=reason,
            energy_objective=net,
        )
