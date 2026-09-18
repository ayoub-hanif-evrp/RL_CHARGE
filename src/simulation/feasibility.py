"""Infeasibility reasons and method-independent transition checks.

A station is never declared feasible merely because "some station is reachable".
Global charging-schedule existence is **not** claimed here; that is Part 3
(frvcpy / exact solvers). Part 2 marks generated routes ``unverified``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from data.models import NodeType
from domain.quantities import PayloadMass
from physics.battery import BatteryModel, BatteryState
from physics.energy import EnergyModel, EnergyResult


class InfeasibilityReason(Enum):
    INSUFFICIENT_ENERGY = "INSUFFICIENT_ENERGY"
    TIME_WINDOW_VIOLATION = "TIME_WINDOW_VIOLATION"
    CAPACITY_VIOLATION = "CAPACITY_VIOLATION"
    INVALID_TARGET_SOC = "INVALID_TARGET_SOC"
    UNKNOWN_STATION = "UNKNOWN_STATION"
    INVALID_STATE = "INVALID_STATE"
    LOOP_GUARD = "LOOP_GUARD"
    NO_FEASIBLE_ACTION = "NO_FEASIBLE_ACTION"
    ZERO_CHARGE_NOOP = "ZERO_CHARGE_NOOP"
    NO_ENERGY_CONTINUATION = "NO_ENERGY_CONTINUATION"
    STATION_REVISIT = "STATION_REVISIT"


# Same-station charges below this SOC increment are no-ops and must be rejected.
ZERO_CHARGE_EPS = 1e-9


@dataclass(frozen=True)
class ReachabilityCheck:
    from_id: str
    to_id: str
    payload: PayloadMass
    energy: EnergyResult
    battery_feasible: bool
    reason: Optional[InfeasibilityReason]


class FeasibilityService:
    def __init__(self, energy_model: EnergyModel, battery_model: BatteryModel):
        self.energy_model = energy_model
        self.battery_model = battery_model
        self.network = energy_model.network

    def can_reach(
        self,
        from_id: str,
        to_id: str,
        payload: PayloadMass,
        battery: BatteryState,
    ) -> ReachabilityCheck:
        energy = self.energy_model.energy_for_arc(from_id, to_id, payload)
        transition = self.battery_model.apply_arc_energy(battery, energy.net_energy)
        return ReachabilityCheck(
            from_id=from_id,
            to_id=to_id,
            payload=payload,
            energy=energy,
            battery_feasible=transition.feasible,
            reason=None if transition.feasible else InfeasibilityReason.INSUFFICIENT_ENERGY,
        )

    def is_station(self, string_id: str) -> bool:
        return self.network.node(string_id).node_type is NodeType.STATION
