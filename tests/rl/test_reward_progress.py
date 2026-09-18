"""Failure-progress term: G_success = -T, G_failure = -H - L_remaining."""

import inspect

import pytest

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.env import ShieldedRouteEnv
from routing.fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute
from simulation.progress import failure_step_reward, remaining_time_lower_bound
from simulation.shield import CONTINUE_INDEX
from simulation.simulator import FixedRouteSimulator
import simulation.progress as progress_mod


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


def _make(write_instance, customers=("C1", "C2")):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 5.0, 0.0),
        node_row("C2", "c", 4.0, 0.0, 10.0, 0.0, 10_000.0, 7.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    route = _route(instance, customers)
    sim = FixedRouteSimulator(
        instance, customers, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    env = ShieldedRouteEnv(
        instance, route, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    return env, sim, instance


def test_l_remaining_nonnegative(write_instance):
    _, sim, _ = _make(write_instance)
    bound = remaining_time_lower_bound(sim)
    assert bound >= 0.0
    assert remaining_time_lower_bound(sim) == bound


def test_l_remaining_uses_only_frozen_route_times(write_instance):
    _, sim, _ = _make(write_instance)
    imports = inspect.getsource(progress_mod)
    env_src = inspect.getsource(ShieldedRouteEnv)
    assert "experiments.split" not in imports
    assert "load_split_routes" not in imports
    assert "validation.json" not in imports
    assert "test.json" not in imports
    assert "validation.json" not in env_src
    assert "test.json" not in env_src
    assert "load_split_routes" not in env_src
    travel_c1 = sim.network.arc("D0", "C1").travel_time.value
    service_c1 = float(sim.network.node("C1").service_time)
    travel_c2 = sim.network.arc("C1", "C2").travel_time.value
    service_c2 = float(sim.network.node("C2").service_time)
    travel_depot = sim.network.arc("C2", "D0").travel_time.value
    expected = travel_c1 + service_c1 + travel_c2 + service_c2 + travel_depot
    assert remaining_time_lower_bound(sim) == pytest.approx(expected)


def test_successful_return_is_minus_completion_time(write_instance):
    env, _, _ = _make(write_instance)
    env.reset()
    steps = 0
    while not env.simulator.state.completed and steps < 20:
        info = env.step(CONTINUE_INDEX, 0.0)
        assert not info.failed
        steps += 1
    assert env.simulator.state.completed
    t = env.simulator.state.time.value
    assert env.return_value == pytest.approx(-t)


def test_failures_at_different_progress_have_distinct_returns(write_instance):
    env, sim, _ = _make(write_instance)
    h = sim.horizon
    l_start = remaining_time_lower_bound(sim)
    t0 = sim.state.time.value
    assert failure_step_reward(sim) == pytest.approx(-(h - t0) - l_start)
    env.reset()
    early = env.step(10_000, 0.0)
    assert early.failed
    g_start = env.return_value
    assert g_start == pytest.approx(-h - l_start)

    env.reset()
    info = env.step(CONTINUE_INDEX, 0.0)
    assert not info.failed
    l_after = remaining_time_lower_bound(env.simulator)
    t_after = env.simulator.state.time.value
    assert failure_step_reward(env.simulator) == pytest.approx(-(h - t_after) - l_after)
    late = env.step(10_000, 0.0)
    assert late.failed
    g_after = env.return_value
    assert l_after >= 0.0
    assert l_after < l_start
    assert g_after != g_start
    assert g_after > g_start
    assert g_after == pytest.approx(-h - l_after)
    assert g_start < -h
    assert g_after < -h
