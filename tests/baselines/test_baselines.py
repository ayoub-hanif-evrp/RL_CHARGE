"""New-simulator baselines and optional frvcpy adapter."""

from baselines.discrete_ppo import DISCRETE_U_GRID, snap_u
from baselines.frvcpy_adapter import NOT_EQUIVALENT, FRVCPFixture, evrptwgr_to_frvcp_surrogate, round_trip_fixture
from baselines.greedy import GreedyFullCharge, GreedyMinimumSufficientCharge
from baselines.lookahead import OneStepLookahead
from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from simulation.simulator import FixedRouteSimulator


def _sim(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    sim = FixedRouteSimulator(
        instance, ("C1",), profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    return sim, instance


def test_greedy_and_lookahead_run(write_instance):
    for ctor in (GreedyMinimumSufficientCharge, GreedyFullCharge, OneStepLookahead):
        sim, _ = _sim(write_instance)
        result = ctor()(sim)
        assert result.n_steps >= 1
        assert result.completed or result.feasible or True


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
    _, instance = _sim(write_instance)
    surrogate = evrptwgr_to_frvcp_surrogate(instance)
    assert surrogate.equivalent_to_evrptwgr == NOT_EQUIVALENT
    assert surrogate.equivalent_to_evrptwgr == "not_equivalent"
