"""Freeze-boundary tests: linear SynthCharge physics, TEST rejection, time cap."""

from __future__ import annotations

import pytest

from domain.load_convention import LoadConvention
from physics.energy import EnergyModel
from physics.network import DirectedArcNetwork
from physics.parameters import PhysicsProfile
from rl.train_loop import assert_learning_split
from routing.synthcharge_instance import instance_from_synthcharge
from simulation.simulator import FixedRouteSimulator
from simulation.time_envelope import time_soc_cap


def _raw():
    return {
        "depot": (0.5, 0.5),
        "customers": [(0.5, 0.8), (0.8, 0.5)],
        "stations": [(0.5, 0.5), (0.2, 0.2)],
        "demands": [0.1, 0.1],
        "service_times": [0.02, 0.02],
        "time_windows": [(0.0, 10.0), (0.0, 10.0)],
        "battery_capacity": 1.0,
        "load_capacity": 1.5,
        "consumption_rate": 0.25,
        "refuel_rate": 2.0,
        "velocity": 1.0,
        "time_horizon": 10.0,
    }


def test_linear_energy_ignores_payload_and_matches_r_times_distance():
    instance = instance_from_synthcharge(
        _raw(), instance_id="sc_R_n2_s1", relative_path="x", layout="R", seed=1
    )
    profile = PhysicsProfile.from_instance(instance, name="synthcharge_linear")
    official = PhysicsProfile.from_instance(instance, name="official_evrptwgr")
    assert profile.energy_law == "linear_distance"
    assert official.energy_law == "demir_model2"
    network = DirectedArcNetwork(instance, profile.average_velocity)
    model = EnergyModel(network, profile)
    from domain.quantities import PayloadMass

    light = model.energy_for_arc("D0", "C1", PayloadMass(0.0))
    heavy = model.energy_for_arc("D0", "C1", PayloadMass(1.0))
    assert light.net_energy.value == pytest.approx(0.25 * light.distance.value)
    assert heavy.net_energy.value == pytest.approx(light.net_energy.value)
    assert light.gradient_percent == 0.0


def test_time_cap_uses_synthcharge_g_and_q():
    instance = instance_from_synthcharge(
        _raw(), instance_id="sc_R_n2_s1", relative_path="x", layout="R", seed=1
    )
    profile = PhysicsProfile.from_instance(instance, name="synthcharge_linear")
    simulator = FixedRouteSimulator(
        instance, ("C1", "C2"), profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    cap = time_soc_cap(simulator, "S0")
    assert cap.max_delay > 0.0
    battery = simulator.battery_model.state_from_soc(simulator.state.soc.value)
    expected_energy = float(battery.energy.value) + cap.max_delay / profile.inverse_refueling_rate
    expected = min(1.0, expected_energy / float(battery.capacity.value))
    assert cap.soc_upper == pytest.approx(expected)
    assert profile.inverse_refueling_rate == pytest.approx(2.0)
    assert float(battery.capacity.value) == pytest.approx(1.0)


def test_official_and_linear_energy_differ_on_the_same_geometry():
    instance = instance_from_synthcharge(
        _raw(), instance_id="sc_R_n2_s1", relative_path="x", layout="R", seed=1
    )
    official = PhysicsProfile.from_instance(instance, name="official_evrptwgr")
    network = DirectedArcNetwork(instance, official.average_velocity)
    model = EnergyModel(network, official)
    from domain.quantities import PayloadMass

    result = model.energy_for_arc("D0", "C1", PayloadMass(0.0))
    assert result.net_energy.value != pytest.approx(0.25 * result.distance.value)


def test_synthcharge_spec_is_predeclared_and_disjoint():
    from routing.synthcharge_benchmark import candidate_design, empty_quotas, length_bin, load_spec

    spec = load_spec()
    assert spec["source_commit"] == "7934a73bbbe20b9be2a9b800127e5e8d4702d345"
    starts = spec["seed_starts"]
    assert starts["train"] == 100000
    assert starts["validation"] == 200000
    assert starts["test"] == 300000
    span = spec["max_candidate_seeds_per_split"]
    ranges = {
        name: range(int(start), int(start) + int(span)) for name, start in starts.items()
    }
    assert set(ranges["train"]).isdisjoint(ranges["validation"])
    assert set(ranges["validation"]).isdisjoint(ranges["test"])
    assert set(ranges["train"]).isdisjoint(ranges["test"])
    assert length_bin(3, spec) == "short"
    assert length_bin(10, spec) == "medium"
    assert length_bin(15, spec) == "long"
    assert length_bin(2, spec) is None
    assert length_bin(16, spec) is None
    def total(split: str) -> int:
        return sum(sum(counts.values()) for counts in empty_quotas(spec, split).values())

    assert total("train") == 180
    assert total("validation") == 90
    assert total("test") == 90
    layouts = [candidate_design(spec, i)[0] for i in range(9)]
    assert layouts == ["R", "C", "RC", "R", "C", "RC", "R", "C", "RC"]
    customers = [candidate_design(spec, i)[1] for i in range(9)]
    assert customers == [15, 15, 15, 30, 30, 30, 50, 50, 50]
    assert candidate_design(spec, 0)[2] == 3
    assert candidate_design(spec, 3)[2] == 6
    assert candidate_design(spec, 6)[2] == 10


def test_training_rejects_the_test_split():
    with pytest.raises(ValueError):
        assert_learning_split("test")
    assert_learning_split("train")
    assert_learning_split("validation")
