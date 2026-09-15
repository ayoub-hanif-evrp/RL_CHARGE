"""Shared scientific types that are not dataset parsing and not simulation."""

from .load_convention import (
    LoadConvention,
    initial_payload_kg,
    payload_after_service_kg,
    require_load_convention,
)
from .quantities import (
    Altitude,
    BatteryEnergy,
    Distance,
    Energy,
    Gradient,
    PayloadMass,
    SocFraction,
    TravelTime,
    energy_from_normalized_rate,
)

__all__ = [
    "Altitude",
    "BatteryEnergy",
    "Distance",
    "Energy",
    "Gradient",
    "LoadConvention",
    "initial_payload_kg",
    "payload_after_service_kg",
    "PayloadMass",
    "SocFraction",
    "TravelTime",
    "energy_from_normalized_rate",
    "require_load_convention",
]
