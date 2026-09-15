"""Mutable simulator state. The frozen customer sequence is never stored here as a list that can be reordered; it lives on the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from domain.quantities import BatteryEnergy, PayloadMass, SocFraction, TravelTime

from .events import SimulatorEvent
from .metrics import MetricsAccumulator


@dataclass
class SimulatorState:
    current_node_id: str
    time: TravelTime
    battery_energy: BatteryEnergy
    soc: SocFraction
    payload: PayloadMass
    next_customer_index: int
    n_station_visits_since_last_customer: int
    completed: bool
    events: List[SimulatorEvent] = field(default_factory=list)
    metrics: MetricsAccumulator = field(default_factory=MetricsAccumulator)
    served_customers: List[str] = field(default_factory=list)
