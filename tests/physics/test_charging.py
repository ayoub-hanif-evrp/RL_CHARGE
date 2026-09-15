"""Charging-model abstraction: linear benchmark, no invented nonlinear curve."""

import pytest

from domain.quantities import BatteryEnergy, TravelTime
from physics.battery import BatteryState
from physics.charging import (
    BenchmarkCompatibleLinearChargingModel,
    GenericPiecewiseLinearChargingModel,
)
from physics.errors import MissingPhysicsParameterError


def _battery(q=77.75):
    return BatteryState(
        energy=BatteryEnergy(0.4 * q),
        capacity=BatteryEnergy(q),
        min_soc_fraction=0.0,
        max_soc_fraction=1.0,
    )


def test_linear_charging_time_is_g_times_energy():
    battery = _battery()
    model = BenchmarkCompatibleLinearChargingModel(3.47)
    target = BatteryEnergy(0.8 * 77.75)
    result = model.charging_time(
        battery.energy, target, battery, "S0", TravelTime(0.0)
    )
    assert result.energy_added.value == pytest.approx(target.value - battery.energy.value)
    assert result.charging_duration.value == pytest.approx(3.47 * result.energy_added.value)


def test_piecewise_model_refuses_missing_knots():
    with pytest.raises(MissingPhysicsParameterError, match="no default"):
        GenericPiecewiseLinearChargingModel()


def test_piecewise_model_interpolates_explicit_knots():
    model = GenericPiecewiseLinearChargingModel(
        soc_knots=(0.0, 0.5, 1.0),
        cumulative_time_knots=(0.0, 10.0, 40.0),
    )
    battery = _battery(q=100.0)
    battery = BatteryState(
        energy=BatteryEnergy(20.0),
        capacity=BatteryEnergy(100.0),
        min_soc_fraction=0.0,
        max_soc_fraction=1.0,
    )
    result = model.charging_time(
        battery.energy, BatteryEnergy(50.0), battery, "S0", TravelTime(0.0)
    )
    # 0.2 -> 0.5 on a curve with 0.0->0, 0.5->10
    assert result.charging_duration.value == pytest.approx(10.0 - 4.0)
