"""Restricted label-setting: visited-station keys and Pareto labels."""

import inspect

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from exact.label_setting import _discrete_key, _pareto_accept, solve_label_setting
from physics.parameters import PhysicsProfile
from routing.fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute
from simulation.actions import ChargeAction
from simulation.simulator import FixedRouteSimulator
import exact.label_setting as ls_mod


def _route(instance, customers):
    return FrozenRoute(
        route_id=f"{instance.metadata.instance_id}__v0",
        source_dataset="EVRPTW-GR",
        doi="10.17632/srfdbp2twv.1",
        raw_instance_id=instance.metadata.instance_id,
        relative_path=instance.metadata.relative_path,
        network_group=instance.metadata.network_group.value,
        terrain_variant=instance.metadata.terrain_variant.value,
        customer_distribution=instance.metadata.customer_distribution,
        schedule_type=instance.metadata.schedule_type,
        generator="test",
        generator_version="0",
        seed=0,
        stop="MaxIterations",
        n_iterations=1,
        config_hash="t",
        physics_profile="official_evrptwgr",
        capacity_policy="official_3650kg",
        distance_scale=1000,
        demand_scale=10,
        rounding_policy="numpy_round_to_int64",
        routing_problem="vrptw_pickup_customers_only",
        python="test",
        os_name="test",
        arch="test",
        instance_sha256="0" * 64,
        vehicle_index=0,
        depot_id=instance.depot.string_id,
        customer_ids=tuple(customers),
        route_demand=0.0,
        route_distance=0.0,
        route_duration_lower_bound=0.0,
        n_customers=len(customers),
        routing_feasible=True,
        charging_feasibility_status=CHARGING_FEASIBILITY_UNVERIFIED,
        generation_runtime_s=0.0,
        routing_objective=0.0,
        base_instance=instance.metadata.base_instance,
        customer_folder=instance.metadata.customer_folder,
    )


def test_discrete_key_includes_visited_station_set(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S2", "f", 2.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 3.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    sim = FixedRouteSimulator(
        instance, ("C1",), profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    before = _discrete_key(sim)
    assert sim.step(ChargeAction("S1", 1.0)).feasible
    after_s1 = _discrete_key(sim)
    sim2 = FixedRouteSimulator(
        instance, ("C1",), profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    assert sim2.step(ChargeAction("S2", 1.0)).feasible
    after_s2 = _discrete_key(sim2)
    assert before != after_s1
    assert after_s1 != after_s2
    assert after_s1[1] != after_s2[1] or after_s1[2] != after_s2[2]
    source = inspect.getsource(ls_mod)
    assert "stations_visited_since_progress" in source
    assert "exact for continuous" not in source.lower() or "not" in source.lower()


def test_pareto_keeps_incomparable_time_soc_labels():
    frontier = []
    assert _pareto_accept(frontier, (10.0, 0.90))
    assert _pareto_accept(frontier, (12.0, 0.95))
    assert len(frontier) == 2
    assert not _pareto_accept(frontier, (11.0, 0.90))
    assert _pareto_accept(frontier, (9.0, 0.95))
    assert all(not (t == 12.0 and abs(s - 0.95) < 1e-12) for t, s in frontier)


def test_restricted_search_does_not_claim_full_exact(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    result = solve_label_setting(
        instance, _route(instance, ("C1",)), profile=profile, max_expansions=500
    )
    assert result.exact_for == "restricted_continuation_or_full_soc"
    assert result.status != "exact"


def test_stop_at_first_feasible_does_not_claim_exact(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    result = solve_label_setting(
        instance,
        _route(instance, ("C1",)),
        profile=profile,
        max_expansions=500,
        stop_at_first_feasible=True,
    )
    assert result.status in {"feasible_for_action_set", "infeasible", "timeout"}
    assert result.exact_for == "restricted_continuation_or_full_soc"
    if result.status == "feasible_for_action_set":
        assert result.feasible
        assert result.route_completion_time is not None
