"""Feasibility shield: energy-continuation bound, no-op mask, A2 vs FULL intervals."""

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from simulation.actions import ChargeAction, ContinueAction
from simulation.feasibility import InfeasibilityReason
from simulation.shield import (
    ARRIVAL_TO_MAX,
    CONTINUATION_TO_MAX,
    continuation_departure_soc,
    continuation_hop_ranks,
    energy_continuable_stations,
    evaluate_shield,
    map_u_to_target_soc,
    soc_interval_for_station,
)
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


def test_no_path_station_masked(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 200.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    decision = evaluate_shield(sim)
    assert decision.mask[0] is False
    assert decision.mask[1] is False
    assert decision.station_reasons[0] is InfeasibilityReason.NO_ENERGY_CONTINUATION
    assert "S0" not in energy_continuable_stations(sim)


def test_direct_continuation_unmasks_station(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 40.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 90.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    decision = evaluate_shield(sim)
    assert decision.continue_legal is False
    assert decision.mask[1] is True
    assert "S0" in energy_continuable_stations(sim)
    bound = continuation_departure_soc(sim, "S0")
    assert bound is not None
    assert bound > 0.0


def test_station_station_customer_continuation(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 10.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S_mid", "f", 55.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 125.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    good = energy_continuable_stations(sim)
    assert "S_mid" in good
    assert "S0" in good
    decision = evaluate_shield(sim)
    ids = list(decision.station_ids)
    assert decision.mask[1 + ids.index("S0")] is True
    assert decision.mask[1 + ids.index("S_mid")] is True


def test_full_vs_a2_intervals_differ(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 40.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 90.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    full = soc_interval_for_station(sim, "S0", mode=CONTINUATION_TO_MAX)
    a2 = soc_interval_for_station(sim, "S0", mode=ARRIVAL_TO_MAX)
    assert full.soc_upper == a2.soc_upper
    assert full.soc_lower > a2.soc_lower + 1e-6
    assert map_u_to_target_soc(sim, "S0", 0.0, mode=CONTINUATION_TO_MAX) == full.soc_lower
    assert map_u_to_target_soc(sim, "S0", 0.0, mode=ARRIVAL_TO_MAX) == a2.soc_lower
    assert map_u_to_target_soc(sim, "S0", 1.0, mode=CONTINUATION_TO_MAX) == full.soc_upper


def test_computed_min_target_soc_matches_continuation(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 40.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 90.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    interval = soc_interval_for_station(sim, "S0", mode=CONTINUATION_TO_MAX)
    bound = continuation_departure_soc(sim, "S0")
    arrival = a2_lower = soc_interval_for_station(sim, "S0", mode=ARRIVAL_TO_MAX).soc_lower
    assert interval.soc_lower == max(arrival, bound)
    assert a2_lower == arrival


def test_same_station_zero_charge_cannot_loop(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    first = sim.step(ChargeAction("S0", 1.0))
    assert first.feasible
    decision = evaluate_shield(sim)
    assert decision.mask[1] is False
    assert decision.station_reasons[0] is InfeasibilityReason.ZERO_CHARGE_NOOP
    noop = sim.step(ChargeAction("S0", 1.0))
    assert not noop.feasible
    assert noop.reason is InfeasibilityReason.ZERO_CHARGE_NOOP
    sim2 = _sim(write_instance, nodes, ("C1",))
    mid = sim2.step(ChargeAction("S0", 0.992))
    assert mid.feasible
    again = sim2.step(ChargeAction("S0", 1.0))
    assert again.feasible


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


def test_continuation_allows_geometrically_farther_station_hop(write_instance):
    """The only feasible chain goes S_near → S_far (farther from C1) → S1 → C1.

    Euclidean progress would reject S_near → S_far. Hop ranks must keep it.
    Q ≈ 77.75; empty-flat energy ≈ Euclidean distance.
    """
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S_near", "f", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S_far", "f", 0.0, 70.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 70.0, 70.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 100.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    c1 = sim.network.node("C1")
    near = sim.network.node("S_near")
    far = sim.network.node("S_far")
    d_near = (near.x - c1.x) ** 2 + (near.y - c1.y) ** 2
    d_far = (far.x - c1.x) ** 2 + (far.y - c1.y) ** 2
    assert d_far > d_near + 1e-6
    ranks = continuation_hop_ranks(sim)
    assert ranks["S1"] == 1
    assert ranks["S_far"] == 2
    assert ranks["S_near"] == 3
    assert "S_near" in energy_continuable_stations(sim)
    bound = continuation_departure_soc(sim, "S_near")
    assert bound is not None
    ids = list(evaluate_shield(sim).station_ids)
    assert evaluate_shield(sim).mask[1 + ids.index("S_near")] is True
    from simulation.shield import min_departure_soc_for_arc

    assert min_departure_soc_for_arc(sim, "S_near", "S1") is None
    assert min_departure_soc_for_arc(sim, "S_near", "C1") is None
    assert min_departure_soc_for_arc(sim, "S_near", "S_far") is not None
    assert bound == min_departure_soc_for_arc(sim, "S_near", "S_far")
