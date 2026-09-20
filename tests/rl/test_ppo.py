"""Hybrid PPO, reward, normalization, Beta, and eval determinism."""

import math

import numpy as np
import pytest
import torch

from baselines.legacy_ddqn import LegacyTwoStageDDQN
from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.env import ShieldedRouteEnv
from rl.normalization import Normalizer
from rl.policy import HybridPolicy
from rl.ppo import HybridPPO, PPOConfig
from routing.fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute
from simulation.simulator import FixedRouteSimulator, run_continue_only


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


def test_reward_sum_matches_negative_completion_time(write_instance):
    env, instance, profile, route = _env(write_instance)
    sim = FixedRouteSimulator(
        instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    result = run_continue_only(sim)
    assert result.feasible
    features = env.reset()
    total = 0.0
    while True:
        info = env.step(0, 0.0)
        total += info.reward
        if info.done:
            break
    assert env.simulator.state.completed
    assert total == pytest.approx(-env.simulator.state.time.value)
    assert total == pytest.approx(-result.metrics.route_completion_time.value)


def test_train_only_normalization_excludes_val_test():
    train = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    val = np.array([[100.0, 100.0]], dtype=np.float64)
    fitted = Normalizer.empty().fit({"x": train})
    leaked = Normalizer.empty().fit({"x": np.concatenate([train, val], axis=0)})
    assert np.allclose(fitted.mean["x"], [1.0, 1.0])
    assert not np.allclose(fitted.mean["x"], leaked.mean["x"])


def test_beta_logprob_finite_and_eval_deterministic(write_instance):
    env, _, _, _ = _env(write_instance)
    features = env.reset()
    policy = HybridPolicy(d_model=16, n_heads=4, n_layers=1)
    policy.eval()
    with torch.no_grad():
        first = policy.act(features, eval_mode=True)
        second = policy.act(features, eval_mode=True)
    assert int(first.discrete_index.item()) == int(second.discrete_index.item())
    assert float(first.u.item()) == pytest.approx(float(second.u.item()))
    u = torch.tensor(0.5)
    dist = torch.distributions.Beta(first.alpha.detach(), first.beta.detach())
    logp = dist.log_prob(u.clamp(1e-4, 1 - 1e-4))
    assert torch.isfinite(logp).all()
    assert float(first.alpha.item()) > 1.0
    assert float(first.beta.item()) > 1.0


def test_ppo_smoke_cpu(write_instance):
    env, _, _, _ = _env(write_instance)
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
    trainer = HybridPPO(config, device="cpu")
    stats = trainer.smoke_train(env, updates=1)
    assert "loss" in stats
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"):
        assert key in stats
        assert math.isfinite(float(stats[key]))
    ids = {id(p) for p in trainer.optimizer.param_groups[0]["params"]}
    assert ids == {id(p) for p in trainer.policy.parameters()}


def test_ddqn_target_frozen_and_not_in_optimizer():
    agent = LegacyTwoStageDDQN(d_model=16)
    agent.sync_target()
    for parameter in agent.target.parameters():
        assert parameter.requires_grad is False
    assert agent.target_in_optimizer() is False
    online_ids = {id(p) for p in agent.online.parameters()}
    opt_ids = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
    assert opt_ids == online_ids
