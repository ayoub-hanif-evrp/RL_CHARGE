"""Charging-time models.

EVRPTW-GR specifies linear charging ``dt = g * ΔE``. It does **not** publish a
nonlinear charging curve. ``GenericPiecewiseLinearChargingModel`` therefore
requires explicit knots and refuses to invent them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from domain.quantities import BatteryEnergy, Energy, SocFraction, TravelTime

from .battery import BatteryState
from .errors import InvalidPhysicsParameterError, MissingPhysicsParameterError


@dataclass(frozen=True)
class ChargingTimeResult:
    energy_before: BatteryEnergy
    energy_target: BatteryEnergy
    energy_added: Energy
    soc_before: SocFraction
    soc_target: SocFraction
    charging_duration: TravelTime
    model_name: str


class ChargingModel(Protocol):
    def charging_time(
        self,
        energy_before: BatteryEnergy,
        energy_target: BatteryEnergy,
        battery: BatteryState,
        station_id: str,
        current_time: TravelTime,
    ) -> ChargingTimeResult: ...


def _validate_charge_bounds(
    energy_before: BatteryEnergy,
    energy_target: BatteryEnergy,
    battery: BatteryState,
) -> Energy:
    if energy_target.value < energy_before.value - 1e-12:
        raise InvalidPhysicsParameterError(
            "target energy is below current battery energy"
        )
    if energy_target.value > battery.max_energy.value + 1e-12:
        raise InvalidPhysicsParameterError(
            "target energy exceeds maximum SOC * Q"
        )
    if energy_before.value < battery.min_energy.value - 1e-12:
        raise InvalidPhysicsParameterError("battery energy is below the SOC floor")
    return Energy(max(energy_target.value - energy_before.value, 0.0))


class BenchmarkCompatibleLinearChargingModel:
    """Schneider / EVRPTW-GR linear charging: ``charging_time = g * energy_added``.

    ``g`` is the instance footer inverse refueling rate. Time and energy share
    the Schneider unit system used by Model 2: ``g * (YY - y)``.
    """

    name = "benchmark_linear"

    def __init__(self, inverse_refueling_rate: float):
        if inverse_refueling_rate < 0:
            raise InvalidPhysicsParameterError(
                f"inverse refueling rate g must be non-negative, got {inverse_refueling_rate}"
            )
        self.inverse_refueling_rate = float(inverse_refueling_rate)

    def charging_time(
        self,
        energy_before: BatteryEnergy,
        energy_target: BatteryEnergy,
        battery: BatteryState,
        station_id: str,
        current_time: TravelTime,
    ) -> ChargingTimeResult:
        added = _validate_charge_bounds(energy_before, energy_target, battery)
        duration = TravelTime(self.inverse_refueling_rate * added.value)
        return ChargingTimeResult(
            energy_before=energy_before,
            energy_target=energy_target,
            energy_added=added,
            soc_before=SocFraction(energy_before.value / battery.capacity.value),
            soc_target=SocFraction(energy_target.value / battery.capacity.value),
            charging_duration=duration,
            model_name=self.name,
        )


class GenericPiecewiseLinearChargingModel:
    """Piecewise-linear SOC → cumulative charging time.

    Knots must be supplied by a config or dataset. There is no default curve.
    """

    name = "piecewise_linear"

    def __init__(
        self,
        soc_knots: Optional[Sequence[float]] = None,
        cumulative_time_knots: Optional[Sequence[float]] = None,
    ):
        if soc_knots is None or cumulative_time_knots is None:
            raise MissingPhysicsParameterError(
                "GenericPiecewiseLinearChargingModel requires explicit SOC knots "
                "and cumulative charging-time knots; no default nonlinear curve is invented"
            )
        if len(soc_knots) < 2 or len(soc_knots) != len(cumulative_time_knots):
            raise InvalidPhysicsParameterError(
                "piecewise charging curve needs at least two matching SOC and time knots"
            )
        socs = [float(s) for s in soc_knots]
        times = [float(t) for t in cumulative_time_knots]
        for previous, current in zip(socs, socs[1:]):
            if current <= previous:
                raise InvalidPhysicsParameterError("SOC knots must be strictly increasing")
        for previous, current in zip(times, times[1:]):
            if current < previous:
                raise InvalidPhysicsParameterError(
                    "cumulative charging-time knots must be nondecreasing"
                )
        if socs[0] < 0.0 - 1e-12 or socs[-1] > 1.0 + 1e-12:
            raise InvalidPhysicsParameterError("SOC knots must lie in [0, 1]")
        if times[0] < 0.0:
            raise InvalidPhysicsParameterError("cumulative charging time cannot be negative")
        self.soc_knots = tuple(socs)
        self.cumulative_time_knots = tuple(times)

    def charging_time(
        self,
        energy_before: BatteryEnergy,
        energy_target: BatteryEnergy,
        battery: BatteryState,
        station_id: str,
        current_time: TravelTime,
    ) -> ChargingTimeResult:
        added = _validate_charge_bounds(energy_before, energy_target, battery)
        soc_before = energy_before.value / battery.capacity.value
        soc_target = energy_target.value / battery.capacity.value
        duration = TravelTime(
            _interpolate(soc_target, self.soc_knots, self.cumulative_time_knots)
            - _interpolate(soc_before, self.soc_knots, self.cumulative_time_knots)
        )
        if duration.value < -1e-12:
            raise InvalidPhysicsParameterError("piecewise curve produced negative charging time")
        duration = TravelTime(max(duration.value, 0.0))
        return ChargingTimeResult(
            energy_before=energy_before,
            energy_target=energy_target,
            energy_added=added,
            soc_before=SocFraction(soc_before),
            soc_target=SocFraction(soc_target),
            charging_duration=duration,
            model_name=self.name,
        )


def _interpolate(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if x < xs[0] - 1e-12 or x > xs[-1] + 1e-12:
        raise InvalidPhysicsParameterError(
            f"SOC {x} is outside the supplied knot domain [{xs[0]}, {xs[-1]}]"
        )
    x = min(max(x, xs[0]), xs[-1])
    for left, right, y_left, y_right in zip(xs, xs[1:], ys, ys[1:]):
        if x <= right or right == xs[-1]:
            if right == left:
                return y_left
            span = (x - left) / (right - left)
            return y_left + span * (y_right - y_left)
    return ys[-1]
