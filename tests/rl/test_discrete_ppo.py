"""True DiscretePPO: categorical station + six charge levels, no Beta snap."""

import pytest
import torch

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.ablation import AblationConfig
from rl.policy import CHARGE_U_LEVELS, HybridPolicy, charge_level_from_u
from routing.fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute
from rl.env import ShieldedRouteEnv


def _env(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    route = FrozenRoute(
        route_id="t",
        source_dataset="EVRPTW-GR",
        doi="x",
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
        customer_ids=("C1",),
        route_demand=0.0,
        route_distance=0.0,
        route_duration_lower_bound=0.0,
        n_customers=1,
        routing_feasible=True,
        charging_feasibility_status=CHARGING_FEASIBILITY_UNVERIFIED,
        generation_runtime_s=0.0,
        routing_objective=0.0,
        base_instance=instance.metadata.base_instance,
        customer_folder=instance.metadata.customer_folder,
    )
    env = ShieldedRouteEnv(
        instance,
        route,
        profile,
        LoadConvention.OFFICIAL_REFERENCE_PICKUP,
        ablation=AblationConfig.from_name("A1"),
    )
    return env


def test_continue_log_prob_has_no_charge_level(write_instance):
    env = _env(write_instance)
    features = env.reset()
    policy = HybridPolicy(d_model=16, n_heads=4, n_layers=1, ablation=AblationConfig.from_name("A1"))
    policy.eval()
    with torch.no_grad():
        out = policy.evaluate_actions(features, 0, 0.4)
        net = policy.forward(features)
        dist = torch.distributions.Categorical(logits=net["logits"])
        expected = dist.log_prob(torch.tensor([0]))
    assert float(out["log_prob"].item()) == pytest.approx(float(expected.item()), abs=1e-5)
    assert float(out["charge_entropy"].item()) == pytest.approx(0.0)
    assert float(out["beta_entropy"].item()) == pytest.approx(0.0)


def test_station_log_prob_is_station_plus_charge_level(write_instance):
    env = _env(write_instance)
    features = env.reset()
    policy = HybridPolicy(d_model=16, n_heads=4, n_layers=1, ablation=AblationConfig.from_name("A1"))
    policy.eval()
    u = CHARGE_U_LEVELS[3]
    with torch.no_grad():
        out = policy.evaluate_actions(features, 1, u)
        net = policy.forward(features)
        dist = torch.distributions.Categorical(logits=net["logits"])
        log_disc = dist.log_prob(torch.tensor([1]))
        charge = torch.distributions.Categorical(logits=net["charge_logits"][0, 0])
        log_level = charge.log_prob(torch.tensor(3))
    assert float(out["log_prob"].item()) == pytest.approx(float((log_disc + log_level).item()), abs=1e-5)
    assert float(out["entropy"].item()) == pytest.approx(
        float((out["categorical_entropy"] + out["charge_entropy"]).item()), abs=1e-5
    )
    assert out["charge_level"] == 3


def test_discrete_eval_is_deterministic_and_on_grid(write_instance):
    env = _env(write_instance)
    features = env.reset()
    policy = HybridPolicy(d_model=16, n_heads=4, n_layers=1, ablation=AblationConfig.from_name("A1"))
    policy.eval()
    with torch.no_grad():
        first = policy.act(features, eval_mode=True)
        second = policy.act(features, eval_mode=True)
    assert int(first.discrete_index.item()) == int(second.discrete_index.item())
    assert float(first.u.item()) == pytest.approx(float(second.u.item()))
    if int(first.discrete_index.item()) > 0:
        u_val = float(first.u.item())
        assert any(abs(u_val - level) < 1e-5 for level in CHARGE_U_LEVELS)
        assert int(first.charge_level.item()) == charge_level_from_u(u_val)
    else:
        assert float(first.u.item()) == pytest.approx(0.0)


def test_charge_level_head_receives_gradient(write_instance):
    env = _env(write_instance)
    features = env.reset()
    policy = HybridPolicy(d_model=16, n_heads=4, n_layers=1, ablation=AblationConfig.from_name("A1"))
    policy.train()
    out = policy.evaluate_actions(features, 1, CHARGE_U_LEVELS[2])
    loss = -out["log_prob"]
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in policy.charge_level_head.parameters() if p.grad is not None]
    assert grads and sum(grads) > 0.0
    beta_grads = [p.grad for p in policy.beta_head.parameters() if p.grad is not None]
    assert not beta_grads or all(g.abs().sum().item() == 0.0 for g in beta_grads)
