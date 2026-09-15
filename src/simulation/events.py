"""Auditable event records for a simulated fixed-route trajectory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Union

from domain.quantities import (
    Altitude,
    BatteryEnergy,
    Distance,
    Energy,
    PayloadMass,
    SocFraction,
    TravelTime,
)


def _qty(value) -> float:
    return value.value if hasattr(value, "value") else value


@dataclass(frozen=True)
class TravelEvent:
    kind: str
    from_node: str
    to_node: str
    distance: Distance
    travel_time: TravelTime
    altitude_difference: Altitude
    gradient_percent: float
    payload_before: PayloadMass
    energy_net: Energy
    energy_consumed: Energy
    energy_recovered: Energy
    battery_before: BatteryEnergy
    battery_after: BatteryEnergy
    departure_time: TravelTime
    arrival_time: TravelTime
    ceiling_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "distance": _qty(self.distance),
            "travel_time": _qty(self.travel_time),
            "altitude_difference": _qty(self.altitude_difference),
            "gradient_percent": self.gradient_percent,
            "payload_before": _qty(self.payload_before),
            "energy_net": _qty(self.energy_net),
            "energy_consumed": _qty(self.energy_consumed),
            "energy_recovered": _qty(self.energy_recovered),
            "battery_before": _qty(self.battery_before),
            "battery_after": _qty(self.battery_after),
            "departure_time": _qty(self.departure_time),
            "arrival_time": _qty(self.arrival_time),
            "ceiling_hit": self.ceiling_hit,
        }


@dataclass(frozen=True)
class CustomerServiceEvent:
    kind: str
    customer_id: str
    ready_time: float
    due_date: float
    waiting_time: TravelTime
    service_start: TravelTime
    service_end: TravelTime
    service_time: TravelTime
    demand: float
    payload_before: PayloadMass
    payload_after: PayloadMass

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["waiting_time"] = _qty(self.waiting_time)
        data["service_start"] = _qty(self.service_start)
        data["service_end"] = _qty(self.service_end)
        data["service_time"] = _qty(self.service_time)
        data["payload_before"] = _qty(self.payload_before)
        data["payload_after"] = _qty(self.payload_after)
        return data


@dataclass(frozen=True)
class ChargeEvent:
    kind: str
    station_id: str
    soc_before: SocFraction
    soc_target: SocFraction
    soc_after: SocFraction
    energy_added: Energy
    charging_duration: TravelTime
    arrival_time: TravelTime
    departure_time: TravelTime
    payload: PayloadMass

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "station_id": self.station_id,
            "soc_before": _qty(self.soc_before),
            "soc_target": _qty(self.soc_target),
            "soc_after": _qty(self.soc_after),
            "energy_added": _qty(self.energy_added),
            "charging_duration": _qty(self.charging_duration),
            "arrival_time": _qty(self.arrival_time),
            "departure_time": _qty(self.departure_time),
            "payload": _qty(self.payload),
        }


@dataclass(frozen=True)
class DepotEvent:
    kind: str
    node_id: str
    time: TravelTime
    battery: BatteryEnergy
    payload: PayloadMass

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "node_id": self.node_id,
            "time": _qty(self.time),
            "battery": _qty(self.battery),
            "payload": _qty(self.payload),
        }


SimulatorEvent = Union[TravelEvent, CustomerServiceEvent, ChargeEvent, DepotEvent]
