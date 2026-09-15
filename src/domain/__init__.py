"""Shared scientific types that are not dataset parsing and not simulation."""

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
    "PayloadMass",
    "SocFraction",
    "TravelTime",
    "energy_from_normalized_rate",
]
