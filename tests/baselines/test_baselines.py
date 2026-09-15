"""Deterministic baselines: min/full charge progress and lookahead (no vacuous asserts)."""

from baselines.discrete_ppo import DISCRETE_U_GRID, snap_u
from baselines.frvcpy_adapter import NOT_EQUIVALENT, FRVCPFixture, evrptwgr_to_frvcp_surrogate, round_trip_fixture
from baselines.greedy import GreedyFullCharge, GreedyMinimumSufficientCharge
from baselines.lookahead import OneStepLookahead
from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from simulation.simulator import FixedRouteSimulator


def _need_charge_sim(write_instance):
    # D0→C1 exceeds Q; S0 is the outbound charger; S1 sits on C1 so min-charge
    # can top up for the return via S0. Distances in instance units (energy ≈ km).
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 70.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 80.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 80.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    sim = FixedRouteSimulator(
        instance, ("C1",), profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    return sim, instance


def _run(ctor, write_instance):
    sim, _ = _need_charge_sim(write_instance)
    start_energy = sim.state.metrics.total_energy_charged
    result = ctor()(sim)
    return result, sim, start_energy


def test_min_charge_completes_and_adds_energy(write_instance):
    result, sim, start = _run(GreedyMinimumSufficientCharge, write_instance)
    assert result.completed
    assert result.feasible
    assert sim.state.completed
    assert sim.state.metrics.total_energy_charged > start + 1e-9
    assert sim.state.metrics.number_of_station_visits < 50


def test_full_charge_completes(write_instance):
    result, sim, _ = _run(GreedyFullCharge, write_instance)
    assert result.completed
    assert result.feasible
    assert sim.state.metrics.total_energy_charged > 1e-9
    assert sim.state.metrics.number_of_station_visits < 50


def test_lookahead_completes(write_instance):
    result, sim, _ = _run(OneStepLookahead, write_instance)
    assert result.completed
    assert result.feasible
    assert sim.state.metrics.number_of_station_visits < 50


def test_no_baseline_accumulates_fifty_zero_energy_station_visits(write_instance):
    for ctor in (GreedyMinimumSufficientCharge, GreedyFullCharge, OneStepLookahead):
        result, sim, _ = _run(ctor, write_instance)
        assert result.n_steps < 50
        assert sim.state.metrics.number_of_station_visits < 50
        charged = sim.state.metrics.total_energy_charged
        visits = sim.state.metrics.number_of_station_visits
        if visits:
            assert charged > 1e-9


def test_snap_u_grid():
    assert snap_u(0.01) == 0.0
    assert snap_u(1.0) == 1.0
    assert DISCRETE_U_GRID[0] == 0.0


def test_frvcpy_fixture_round_trip_and_surrogate_flag(write_instance):
    fixture = FRVCPFixture(
        node_ids=["D0", "S0", "C1"],
        energy_matrix=[[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        station_ids=["S0"],
    )
    assert round_trip_fixture(fixture).to_dict() == fixture.to_dict()
    _, instance = _need_charge_sim(write_instance)
    surrogate = evrptwgr_to_frvcp_surrogate(instance)
    assert surrogate.equivalent_to_evrptwgr == NOT_EQUIVALENT
    assert surrogate.equivalent_to_evrptwgr == "not_equivalent"
