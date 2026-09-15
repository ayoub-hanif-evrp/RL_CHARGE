"""Feasibility shield: masks transitions, does not choose when to charge."""

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from simulation.actions import ChargeAction, ContinueAction
from simulation.shield import evaluate_shield, map_u_to_target_soc, soc_interval_for_station
from simulation.simulator import FixedRouteSimulator


def _sim(write_instance, nodes, customers):
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    return FixedRouteSimulator(
        instance, customers, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )


def test_continue_masked_when_energy_infeasible(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 200.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    decision = evaluate_shield(sim)
    assert decision.continue_legal is False
    assert decision.mask[0] is False


def test_station_masked_when_unreachable(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 200.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 1.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    decision = evaluate_shield(sim)
    assert decision.mask[1] is False


def test_station_legal_even_if_cannot_reach_next_customer(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 200.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    decision = evaluate_shield(sim)
    assert decision.mask[0] is False
    assert decision.mask[1] is True


def test_u_maps_to_soc_interval_endpoints(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    interval = soc_interval_for_station(sim, "S0")
    assert map_u_to_target_soc(sim, "S0", 0.0) == interval.soc_lower
    assert map_u_to_target_soc(sim, "S0", 1.0) == interval.soc_upper


def test_two_station_visits_preserve_customer_order(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S2", "f", 2.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 3.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
        node_row("C2", "c", 4.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1", "C2"))
    assert sim.step(ChargeAction("S1", 1.0)).feasible
    assert sim.step(ChargeAction("S2", 1.0)).feasible
    assert sim.step(ContinueAction()).feasible
    assert sim.step(ContinueAction()).feasible
    assert sim.customer_ids == ("C1", "C2")
    assert sim.state.served_customers == ["C1", "C2"]
