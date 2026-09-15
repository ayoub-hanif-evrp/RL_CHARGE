"""Scientific sanity tests for the EVRPTW-GR energy model."""

import math

import pytest

from data.parser import parse_instance
from data.paths import RAW_EVRPTW_GR_DIR
from domain.quantities import PayloadMass
from physics.energy import EnergyModel
from physics.network import DirectedArcNetwork
from physics.official_replica import model2_empty_flat_baseline, model2_normalized_rate
from physics.parameters import PhysicsProfile
from physics.travel_time import travel_time


def _models(relative):
    path = RAW_EVRPTW_GR_DIR / relative
    if not path.is_file():
        pytest.skip(f"missing {relative}")
    instance = parse_instance(path)
    profile = PhysicsProfile.from_instance(instance)
    network = DirectedArcNetwork(instance, profile.average_velocity)
    return instance, profile, EnergyModel(network, profile)


def test_empty_flat_baseline_matches_published_audit_value():
    _, profile, model = _models("Small_Network/5_Customers/Level/c101C5_L.txt")
    baseline = model2_empty_flat_baseline(profile.curb_mass_kg, profile.energy)
    assert baseline == pytest.approx(0.370677, rel=1e-5)
    assert model.empty_flat_physical_kwh_per_km == pytest.approx(baseline)


def test_level_flat_arc_normalized_rate_is_one():
    instance, profile, model = _models("Small_Network/5_Customers/Level/c101C5_L.txt")
    result = model.energy_for_arc("D0", "C30", PayloadMass(0.0))
    assert abs(result.angle_deg) < 1e-12
    assert abs(result.gradient_percent) < 1e-12
    assert result.normalized_rate == pytest.approx(1.0, rel=1e-9)
    assert result.net_energy.value == pytest.approx(result.distance.value, rel=1e-9)


def test_travel_time_is_distance_over_file_v_not_sixty():
    instance, profile, model = _models("Small_Network/5_Customers/Level/c101C5_L.txt")
    arc = model.network.arc("D0", "C30")
    assert profile.average_velocity == 1.0
    assert arc.travel_time == travel_time(arc.distance, profile.average_velocity)
    assert arc.travel_time.value == pytest.approx(arc.distance.value)
    assert profile.energy.physics_speed_km_h == 60.0


def test_same_length_uphill_uses_more_energy_than_flat(write_instance):
    from conftest import node_row
    from data.parser import parse_instance
    from physics.energy import EnergyModel
    from physics.network import DirectedArcNetwork
    from physics.parameters import PhysicsProfile
    from domain.quantities import PayloadMass

    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("C1", "c", 10.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.0),
        node_row("C2", "c", 10.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 1.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    model = EnergyModel(DirectedArcNetwork(instance, profile.average_velocity), profile)
    flat = model.energy_for_arc("D0", "C1", PayloadMass(0.0))
    uphill = model.energy_for_arc("D0", "C2", PayloadMass(0.0))
    assert flat.distance.value == pytest.approx(uphill.distance.value)
    assert uphill.net_energy.value > flat.net_energy.value


def test_uphill_consumes_more_than_reverse_downhill():
    sloped = _models("Small_Network/5_Customers/Nearly_Level/c101C5_NL.txt")[2]
    downhill = sloped.energy_for_arc("D0", "S5", PayloadMass(0.0))
    uphill = sloped.energy_for_arc("S5", "D0", PayloadMass(0.0))
    assert uphill.net_energy.value > downhill.net_energy.value
    assert uphill.distance.value == pytest.approx(downhill.distance.value)


def test_payload_increases_energy_on_demanding_arc():
    _, _, model = _models("Small_Network/5_Customers/Nearly_Level/c101C5_NL.txt")
    empty = model.energy_for_arc("S5", "D0", PayloadMass(0.0))
    heavy = model.energy_for_arc("S5", "D0", PayloadMass(3650.0))
    assert heavy.gross_mass_kg > empty.gross_mass_kg
    assert heavy.net_energy.value > empty.net_energy.value


def test_downhill_regeneration_and_directionality():
    _, _, model = _models("Small_Network/5_Customers/Nearly_Level/c101C5_NL.txt")
    forward = model.energy_for_arc("D0", "S5", PayloadMass(0.0))
    reverse = model.energy_for_arc("S5", "D0", PayloadMass(0.0))
    assert forward.net_energy.value != reverse.net_energy.value
    assert forward.gradient_percent == pytest.approx(-reverse.gradient_percent)
    assert forward.net_energy.value < reverse.net_energy.value


def test_energy_matches_official_replica():
    instance, profile, model = _models("Small_Network/5_Customers/Very_Gentle/c101C5_VG.txt")
    payload = PayloadMass(500.0)
    result = model.energy_for_arc("D0", "C12", payload)
    arc = model.network.arc("D0", "C12")
    p_tract, p_total, consump, hh = model2_normalized_rate(
        profile.curb_mass_kg + payload.value,
        arc.angle_deg,
        profile.curb_mass_kg,
        profile.energy,
    )
    assert result.traction_power_kw == pytest.approx(p_tract, rel=1e-9)
    assert result.battery_power_kw == pytest.approx(p_total, rel=1e-9)
    assert result.normalized_rate == pytest.approx(hh, rel=1e-9)
    assert result.net_energy.value == pytest.approx(hh * arc.distance.value, rel=1e-9)


def test_file_r_is_recorded_and_not_multiplied():
    instance, profile, model = _models("Small_Network/5_Customers/Level/c101C5_L.txt")
    assert profile.file_consumption_rate_unused == 1.0
    result = model.energy_for_arc("D0", "C30", PayloadMass(0.0))
    # If r were multiplied in, empty-flat energy would still be distance * r = distance
    # because r happens to be 1. The replica formula does not take r as an argument.
    assert result.normalized_rate == pytest.approx(1.0, rel=1e-9)


def test_zero_length_arc_is_stable():
    _, _, model = _models("Small_Network/5_Customers/Level/c101C5_L.txt")
    result = model.energy_for_arc("D0", "S0", PayloadMass(100.0))
    assert result.distance.value == 0.0
    assert result.net_energy.value == 0.0
    assert result.angle_deg == 0.0


def test_distance_is_planar_not_slope_length():
    _, _, model = _models("Small_Network/5_Customers/Nearly_Level/c101C5_NL.txt")
    arc = model.network.arc("D0", "C12")
    node_from = model.network.node("D0")
    node_to = model.network.node("C12")
    dx = node_to.x - node_from.x
    dy = node_to.y - node_from.y
    planar = math.hypot(dx, dy)
    slope_length = math.hypot(planar, node_to.altitude - node_from.altitude)
    assert arc.distance.value == pytest.approx(planar)
    assert arc.distance.value != pytest.approx(slope_length)
