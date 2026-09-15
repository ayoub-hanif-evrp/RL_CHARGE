"""Typed quantity safety: distance cannot be subtracted from battery energy."""

import pytest

from domain.quantities import BatteryEnergy, Distance, Energy, energy_from_normalized_rate


def test_same_type_arithmetic():
    assert Distance(3) + Distance(4) == Distance(7)
    assert Energy(5) - Energy(2) == Energy(3)


def test_cannot_subtract_distance_from_battery():
    with pytest.raises(TypeError):
        BatteryEnergy(10.0) - Distance(3.0)


def test_cannot_add_energy_and_distance():
    with pytest.raises(TypeError):
        Energy(1.0) + Distance(1.0)


def test_energy_from_normalized_rate_rejects_raw_float_distance():
    with pytest.raises(TypeError):
        energy_from_normalized_rate(1.0, 10.0)


def test_empty_flat_rate_one_equals_distance():
    assert energy_from_normalized_rate(1.0, Distance(12.5)) == Energy(12.5)
