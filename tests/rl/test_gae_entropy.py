"""GAE truncation bootstrap and hybrid log-prob / entropy consistency."""

import pytest
import torch

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.ablation import AblationConfig
from rl.buffers import RolloutBuffer, Transition
from rl.env import ShieldedRouteEnv
from rl.policy import HybridPolicy
from rl.ppo import HybridPPO, PPOConfig
from routing.fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute


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


def _env(write_instance, nodes=None, customers=("C1",)):
    nodes = nodes or [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    route = _route(instance, customers)
    env = ShieldedRouteEnv(
        instance, route, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    return env, instance, profile, route


def _t(value, reward, done):
    return Transition(
        discrete_index=0,
        u=0.0,
        log_prob=0.0,
        value=value,
        reward=reward,
        done=done,
        features={},
    )


def test_gae_terminal_rollout():
    buf = RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    buf.add(_t(1.0, -2.0, True))
    buf.compute_gae(bootstrap_value=99.0)
    assert buf.advantages[0] == pytest.approx(-3.0)
    assert buf.returns[0] == pytest.approx(-2.0)


def test_gae_truncated_nonterminal():
    buf = RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    buf.add(_t(1.0, -2.0, False))
    buf.compute_gae(bootstrap_value=4.0)
    assert buf.advantages[0] == pytest.approx(1.0)
    assert buf.returns[0] == pytest.approx(2.0)


def test_gae_episode_boundary_inside_rollout():
    buf = RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    buf.add(_t(1.0, -1.0, False))
    buf.add(_t(2.0, -3.0, True))
    buf.add(_t(4.0, -5.0, False))
    buf.compute_gae(bootstrap_value=6.0)
    assert buf.advantages[2] == pytest.approx(-3.0)
    assert buf.advantages[1] == pytest.approx(-5.0)
    assert buf.advantages[0] == pytest.approx(-5.0)
    assert buf.returns[0] == pytest.approx(-4.0)


def test_continue_log_prob_has_no_beta(write_instance):
    env, _, _, _ = _env(write_instance)
    features = env.reset()
    policy = HybridPolicy(d_model=16, n_heads=4, n_layers=1)
    policy.eval()
    with torch.no_grad():
        out = policy.evaluate_actions(features, 0, 0.3)
        net = policy.forward(features)
        dist = torch.distributions.Categorical(logits=net["logits"])
        expected = dist.log_prob(torch.tensor([0]))
    assert torch.isfinite(out["log_prob"]).all()
    assert float(out["log_prob"].item()) == pytest.approx(float(expected.item()), abs=1e-5)
    assert float(out["beta_entropy"].item()) == pytest.approx(0.0)


def test_station_hybrid_log_prob(write_instance):
    env, _, _, _ = _env(write_instance)
    features = env.reset()
    policy = HybridPolicy(d_model=16, n_heads=4, n_layers=1)
    policy.eval()
    with torch.no_grad():
        out = policy.evaluate_actions(features, 1, 0.4)
        net = policy.forward(features)
        dist = torch.distributions.Categorical(logits=net["logits"])
        log_disc = dist.log_prob(torch.tensor([1]))
        beta = torch.distributions.Beta(net["alpha"][0, 0], net["beta"][0, 0])
        log_beta = beta.log_prob(torch.tensor(0.4))
    assert torch.isfinite(out["log_prob"]).all()
    assert float(out["log_prob"].item()) == pytest.approx(
        float((log_disc + log_beta).item()), abs=1e-5
    )
    assert float(out["beta_entropy"].item()) == pytest.approx(float(beta.entropy().item()), abs=1e-5)


def test_conditional_beta_entropy_and_finite_near_endpoints(write_instance):
    env, _, _, _ = _env(write_instance)
    features = env.reset()
    policy = HybridPolicy(d_model=16, n_heads=4, n_layers=1)
    policy.eval()
    with torch.no_grad():
        cont = policy.evaluate_actions(features, 0, 0.0)
        low = policy.evaluate_actions(features, 1, 0.0, u_eps=1e-4)
        high = policy.evaluate_actions(features, 1, 1.0, u_eps=1e-4)
    assert float(cont["entropy"].item()) == pytest.approx(
        float(cont["categorical_entropy"].item()), abs=1e-5
    )
    assert torch.isfinite(low["log_prob"]).all()
    assert torch.isfinite(high["log_prob"]).all()
    assert torch.isfinite(low["entropy"]).all()


def test_collect_stores_executed_u_after_snap(write_instance):
    env, _, _, _ = _env(write_instance)
    env.ablation = AblationConfig.from_name("A1")
    config = PPOConfig(
        learning_rate=1e-3,
        rollout_steps=4,
        minibatch_size=2,
        update_epochs=1,
        clip_eps=0.2,
        gamma=1.0,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        budget_updates=1,
        eval_interval=1,
        early_stopping_patience=1,
        seed=0,
    )
    trainer = HybridPPO(config, device="cpu", ablation=env.ablation)
    buffer = trainer.collect(env, n_steps=4)
    assert len(buffer.transitions) == 4
    assert hasattr(buffer, "advantages")
    grid = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
    for transition in buffer.transitions:
        if transition.discrete_index == 0:
            assert transition.u == 0.0
        else:
            assert transition.u in grid
        assert transition.log_prob == transition.log_prob
