"""Named physics profiles. Missing scientific values fail loudly."""

import pytest

from data.parser import parse_instance
from physics.errors import MissingPhysicsParameterError
from physics.parameters import PhysicsProfile


def test_official_profile_overrides_small_network_capacity(write_instance):
    path, root = write_instance()
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance, name="official_evrptwgr")
    assert profile.curb_mass_kg == 6350.0
    assert profile.payload_capacity_kg == 3650.0
    assert profile.battery_capacity == instance.vehicle.tank_capacity
    assert profile.inverse_refueling_rate == instance.vehicle.inverse_refueling_rate
    assert profile.min_soc_fraction == 0.0
    assert profile.initial_soc_fraction == 1.0
    assert profile.file_consumption_rate_unused == instance.vehicle.consumption_rate


def test_raw_file_profile_fails_without_curb_weight(write_instance):
    path, root = write_instance()
    instance = parse_instance(path, root=root)
    assert instance.vehicle.curb_weight is None
    with pytest.raises(MissingPhysicsParameterError, match="curb weight"):
        PhysicsProfile.from_instance(instance, name="raw_file")


def test_raw_file_profile_reads_m_and_c(write_instance):
    from conftest import FOOTER_WITH_CURB_WEIGHT

    path, root = write_instance(footer=FOOTER_WITH_CURB_WEIGHT)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance, name="raw_file")
    assert profile.curb_mass_kg == 6350.0
    assert profile.payload_capacity_kg == 3650.0
    assert profile.use_instance_curb_mass is True
