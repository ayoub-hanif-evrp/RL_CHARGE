"""Fixed-route simulator: waiting, pickup, continuous SOC, immutable sequence."""

import pytest

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from simulation.actions import ChargeAction, ContinueAction
from simulation.feasibility import InfeasibilityReason
from simulation.simulator import FixedRouteSimulator, run_continue_only


def _sim(
    write_instance,
    nodes=None,
    customers=("C1", "C2"),
    filename="c101C5_L.txt",
    load_convention=LoadConvention.OFFICIAL_REFERENCE_PICKUP,
    **kwargs,
):
    path, root = write_instance(filename=filename, nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    return (
        FixedRouteSimulator(instance, customers, profile, load_convention, **kwargs),
        instance,
        profile,
    )


def test_early_arrival_waits_before_service(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 5.0, 0.0, 10.0, 100.0, 200.0, 10.0, 0.0),
    ]
    sim, _, _ = _sim(write_instance, nodes=nodes, customers=("C1",))
    result = sim.step(ContinueAction())
    assert result.feasible
    service = [e for e in sim.state.events if e.kind == "customer_service"][0]
    assert service.waiting_time.value == pytest.approx(95.0)
    assert service.service_start.value == pytest.approx(100.0)
    assert service.service_end.value == pytest.approx(110.0)


def test_late_arrival_is_time_window_violation(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 5.0, 0.0, 10.0, 0.0, 3.0, 10.0, 0.0),
    ]
    sim, _, _ = _sim(write_instance, nodes=nodes, customers=("C1",))
    result = sim.step(ContinueAction())
    assert not result.feasible
    assert result.reason is InfeasibilityReason.TIME_WINDOW_VIOLATION


def test_pickup_increases_payload_station_does_not(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 5.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 5.0, 0.0, 80.0, 0.0, 10_000.0, 1.0, 0.0),
        node_row("C2", "c", 6.0, 0.0, 20.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim, _, _ = _sim(write_instance, nodes=nodes, customers=("C1", "C2"))
    assert sim.state.payload.value == 0.0
    assert sim.step(ContinueAction()).feasible
    assert sim.state.payload.value == pytest.approx(80.0)
    before = sim.state.payload.value
    assert sim.step(ChargeAction("S1", 1.0)).feasible
    assert sim.state.payload.value == pytest.approx(before)
    assert sim.step(ContinueAction()).feasible
    assert sim.state.payload.value == pytest.approx(100.0)


def test_station_visits_do_not_reorder_customers(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S2", "f", 2.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C8", "c", 3.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
        node_row("C17", "c", 4.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim, _, _ = _sim(write_instance, nodes=nodes, customers=("C8", "C17"))
    frozen = sim.customer_ids
    assert sim.step(ContinueAction()).feasible
    assert sim.step(ChargeAction("S1", 1.0)).feasible
    assert sim.step(ChargeAction("S2", 1.0)).feasible
    assert sim.step(ContinueAction()).feasible
    assert sim.customer_ids == frozen == ("C8", "C17")
    assert sim.state.served_customers == ["C8", "C17"]


def test_continuous_target_soc_not_six_levels(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 30.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 30.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim, _, _ = _sim(write_instance, nodes=nodes, customers=("C1",))
    assert sim.step(ContinueAction()).feasible
    assert sim.state.soc.value < 0.73
    result = sim.step(ChargeAction("S1", 0.73))
    assert result.feasible
    assert sim.state.soc.value == pytest.approx(0.73, abs=1e-9)


def test_capacity_violation(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 1.0, 0.0, 4000.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim, _, profile = _sim(write_instance, nodes=nodes, customers=("C1",))
    assert profile.payload_capacity_kg == 3650.0
    result = sim.step(ContinueAction())
    assert not result.feasible
    assert result.reason is InfeasibilityReason.CAPACITY_VIOLATION


def test_insufficient_energy(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 200.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim, _, profile = _sim(write_instance, nodes=nodes, customers=("C1",))
    assert profile.battery_capacity < 200.0
    result = sim.step(ContinueAction())
    assert not result.feasible
    assert result.reason is InfeasibilityReason.INSUFFICIENT_ENERGY


def test_continue_returns_to_depot(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim, _, _ = _sim(write_instance, nodes=nodes, customers=("C1",))
    assert sim.step(ContinueAction()).feasible
    result = sim.step(ContinueAction())
    assert result.feasible
    assert sim.state.completed
    assert sim.state.current_node_id == "D0"


def test_loop_guard_is_not_three_stops(write_instance, tmp_path):
    toml = tmp_path / "official_evrptwgr.toml"
    original = (tmp_path / "skip")
    from physics.parameters import PHYSICS_CONFIG_DIR
    import shutil

    shutil.copy(PHYSICS_CONFIG_DIR / "official_evrptwgr.toml", toml)
    text = toml.read_text(encoding="utf-8").replace(
        "loop_guard_station_visits = 50", "loop_guard_station_visits = 2"
    )
    toml.write_text(text, encoding="utf-8")
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance, config_dir=tmp_path)
    sim = FixedRouteSimulator(
        instance, ("C1",), profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    assert sim.step(ChargeAction("S1", 0.990)).feasible
    assert sim.step(ChargeAction("S1", 0.995)).feasible
    result = sim.step(ChargeAction("S1", 1.0))
    assert not result.feasible
    assert result.reason is InfeasibilityReason.LOOP_GUARD


def test_continue_only_is_not_a_filter(write_instance):
    sim, _, _ = _sim(write_instance)
    result = run_continue_only(sim)
    # Default synthetic customers may be energy-feasible; either outcome is logged.
    assert result.reason is None or isinstance(result.reason, InfeasibilityReason)
    assert sim.customer_ids == ("C1", "C2")


def test_metrics_have_no_total_cost(write_instance):
    sim, _, _ = _sim(write_instance)
    run_continue_only(sim)
    payload = sim.metrics(feasible=sim.state.completed).to_dict()
    assert "total_cost" not in payload
    assert "energy_objective" in payload
    assert "total_distance" in payload
    assert "total_energy_consumed" in payload
    assert payload["load_convention"] == LoadConvention.OFFICIAL_REFERENCE_PICKUP.value


def test_load_convention_is_required(write_instance):
    path, root = write_instance()
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    with pytest.raises((TypeError, ValueError)):
        FixedRouteSimulator(instance, ("C1", "C2"), profile)


def test_delivery_decreases_payload_and_changes_first_arc_energy(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 5.0, 0.0, 80.0, 0.0, 10_000.0, 1.0, 0.0),
        node_row("C2", "c", 6.0, 0.0, 20.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    pickup, instance, _ = _sim(
        write_instance,
        nodes=nodes,
        customers=("C1", "C2"),
        load_convention=LoadConvention.OFFICIAL_REFERENCE_PICKUP,
    )
    delivery, _, _ = _sim(
        write_instance,
        nodes=nodes,
        customers=("C1", "C2"),
        load_convention=LoadConvention.DELIVERY,
    )
    assert pickup.state.payload.value == pytest.approx(0.0)
    assert delivery.state.payload.value == pytest.approx(100.0)
    assert pickup.step(ContinueAction()).feasible
    assert delivery.step(ContinueAction()).feasible
    assert pickup.state.payload.value == pytest.approx(80.0)
    assert delivery.state.payload.value == pytest.approx(20.0)
    pickup_arc = [e for e in pickup.state.events if e.kind == "travel"][0]
    delivery_arc = [e for e in delivery.state.events if e.kind == "travel"][0]
    assert pickup_arc.energy_net.value != delivery_arc.energy_net.value
    assert pickup_arc.payload_before.value == pytest.approx(0.0)
    assert delivery_arc.payload_before.value == pytest.approx(100.0)
