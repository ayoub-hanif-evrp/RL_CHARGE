"""Typed physical quantities.

Each quantity is a distinct frozen type. Arithmetic is defined only within a
type, or as multiplication/division by a dimensionless scalar. Mixing units
such as ``battery_energy - distance`` is a ``TypeError``.

Conversion between kinds of quantity requires an explicit helper, never an
implicit cast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T", bound="_Quantity")


def _require_same(left: T, right: object) -> None:
    if type(right) is not type(left):
        raise TypeError(
            f"cannot combine {type(left).__name__} with {type(right).__name__}"
        )


@dataclass(frozen=True)
class _Quantity:
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))

    def __add__(self: T, other: object) -> T:
        _require_same(self, other)
        return type(self)(self.value + other.value)  # type: ignore[arg-type]

    def __sub__(self: T, other: object) -> T:
        _require_same(self, other)
        return type(self)(self.value - other.value)  # type: ignore[arg-type]

    def __mul__(self: T, other: object) -> T:
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return type(self)(self.value * float(other))
        return NotImplemented

    def __rmul__(self: T, other: object) -> T:
        return self.__mul__(other)

    def __truediv__(self, other: object):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return type(self)(self.value / float(other))
        if type(other) is type(self):
            return self.value / other.value  # type: ignore[union-attr]
        return NotImplemented

    def __neg__(self: T) -> T:
        return type(self)(-self.value)

    def __abs__(self: T) -> T:
        return type(self)(abs(self.value))

    def __lt__(self, other: object) -> bool:
        _require_same(self, other)
        return self.value < other.value  # type: ignore[union-attr]

    def __le__(self, other: object) -> bool:
        _require_same(self, other)
        return self.value <= other.value  # type: ignore[union-attr]

    def __gt__(self, other: object) -> bool:
        _require_same(self, other)
        return self.value > other.value  # type: ignore[union-attr]

    def __ge__(self, other: object) -> bool:
        _require_same(self, other)
        return self.value >= other.value  # type: ignore[union-attr]


@dataclass(frozen=True)
class Distance(_Quantity):
    """Planar Euclidean length in instance coordinate units."""


@dataclass(frozen=True)
class TravelTime(_Quantity):
    """Elapsed time in Schneider / EVRPTW-GR time units (file ``v``)."""


@dataclass(frozen=True)
class Energy(_Quantity):
    """Normalized Schneider energy units, the same space as battery ``Q``.

    Positive values are consumption. Negative values are regenerated energy
    before the battery ceiling is applied.
    """


@dataclass(frozen=True)
class BatteryEnergy(_Quantity):
    """Energy stored in the battery, in Schneider energy units (same as ``Q``)."""


@dataclass(frozen=True)
class SocFraction(_Quantity):
    """State of charge as ``battery_energy / Q``."""


@dataclass(frozen=True)
class PayloadMass(_Quantity):
    """Cargo mass in kilograms (pickup load; not curb mass)."""


@dataclass(frozen=True)
class Altitude(_Quantity):
    """Elevation in instance coordinate units, not SI metres."""


@dataclass(frozen=True)
class Gradient(_Quantity):
    """Road slope as ``sin(arctan(Δh / d))``, matching EVRPTW-GR Model 2."""


def energy_from_normalized_rate(normalized_rate: float, distance: Distance) -> Energy:
    """Convert a dimensionless consumption rate and a distance into energy.

    ``normalized_rate`` is Model 2's ``hh = consump / consump_base``. Empty
    vehicle on a flat arc has rate 1, so energy equals distance in Schneider
    units.
    """
    if not isinstance(distance, Distance):
        raise TypeError("distance must be a Distance")
    return Energy(float(normalized_rate) * distance.value)
