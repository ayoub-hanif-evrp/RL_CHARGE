"""V4 reward variants, C_train, PBRS telescoping, V3 regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.env import ShieldedRouteEnv
from rl.rewards import RewardComputer, RewardConfig, RewardKind, compute_c_train
from routing.fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute
from simulation.progress import failure_step_reward, remaining_time_lower_bound
from simulation.shield import CONTINUE_INDEX
from simulation.simulator import FixedRouteSimulator

ROOT = Path(__file__).resolve().parents[2]


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


def _make(write_instance, reward_config=None, customers=("C1", "C2")):
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
    env = ShieldedRouteEnv(
        instance,
        route,
        profile,
        LoadConvention.OFFICIAL_REFERENCE_PICKUP,
        reward_config=reward_config,
    )
    return env, instance, route


def test_v3_time_regression_matches_legacy_failure_helper(write_instance):
    env, _, _ = _make(write_instance, RewardConfig.v3_time())
    env.reset()
    sim = env.simulator
    expected = failure_step_reward(sim, decision_time=sim.state.time.value)
    info = env.step(10_000, 0.0)
    assert info.failed
    assert info.reward == pytest.approx(expected)
    assert env.return_value == pytest.approx(-sim.horizon - remaining_time_lower_bound(sim))


def test_v3_success_return_minus_completion(write_instance):
    env, _, _ = _make(write_instance, RewardConfig.v3_time())
    env.reset()
    steps = 0
    while not env.simulator.state.completed and steps < 20:
        info = env.step(CONTINUE_INDEX, 0.0)
        assert not info.failed
        steps += 1
    assert env.simulator.state.completed
    assert env.return_value == pytest.approx(-env.simulator.state.time.value)
    assert env.shaping_return_value == pytest.approx(0.0)


def test_v4_base_scales_by_c_train(write_instance):
    c = 10.0
    env, _, _ = _make(write_instance, RewardConfig.v4(RewardKind.V4_BASE, c))
    env.reset()
    info = env.step(CONTINUE_INDEX, 0.0)
    assert not info.failed
    assert info.reward == pytest.approx(-(info.extra["delta_t"]) / c)
    assert info.extra["shaping_reward"] == pytest.approx(0.0)


def test_v4_pbrs_telescopes_on_success(write_instance):
    c = 10.0
    env, _, _ = _make(write_instance, RewardConfig.v4(RewardKind.V4_PBRS, c))
    env.reset()
    l0 = remaining_time_lower_bound(env.simulator)
    steps = 0
    while not env.simulator.state.completed and steps < 20:
        info = env.step(CONTINUE_INDEX, 0.0)
        assert not info.failed
        steps += 1
    t = env.simulator.state.time.value
    assert env.return_value == pytest.approx((-t + l0) / c)
    assert env.base_return_value == pytest.approx(-t / c)
    assert env.shaping_return_value == pytest.approx(l0 / c)


def test_v4_pbrs_failure_no_double_count_l(write_instance):
    c = 10.0
    env, _, _ = _make(write_instance, RewardConfig.v4(RewardKind.V4_PBRS, c))
    env.reset()
    l0 = remaining_time_lower_bound(env.simulator)
    h = env.simulator.horizon
    info = env.step(10_000, 0.0)
    assert info.failed
    assert env.return_value == pytest.approx((l0 - h) / c)
    # base failure omits L; shaping supplies +L/C
    assert info.extra["base_reward"] == pytest.approx(-(h - 0.0) / c)
    assert info.extra["shaping_reward"] == pytest.approx(l0 / c)


def test_feasible_dominates_failure_v4_pbrs(write_instance):
    c = 10.0
    # success
    env_s, _, _ = _make(write_instance, RewardConfig.v4(RewardKind.V4_PBRS, c))
    env_s.reset()
    steps = 0
    while not env_s.simulator.state.completed and steps < 20:
        env_s.step(CONTINUE_INDEX, 0.0)
        steps += 1
    g_ok = env_s.return_value
    # immediate failure
    env_f, _, _ = _make(write_instance, RewardConfig.v4(RewardKind.V4_PBRS, c))
    env_f.reset()
    env_f.step(10_000, 0.0)
    g_fail = env_f.return_value
    assert g_ok > g_fail


def test_v4_base_no_l_fail_omits_progress_term(write_instance):
    c = 10.0
    env, _, _ = _make(write_instance, RewardConfig.v4(RewardKind.V4_BASE_NO_L_FAIL, c))
    env.reset()
    h = env.simulator.horizon
    info = env.step(10_000, 0.0)
    assert info.failed
    assert env.return_value == pytest.approx(-h / c)
    assert env.objective_return_value == pytest.approx(-h)
    assert env.normalized_base_return_value == pytest.approx(-h / c)
    assert env.shaping_return_value == pytest.approx(0.0)


def test_objective_return_success_is_minus_T(write_instance):
    c = 10.0
    env, _, _ = _make(write_instance, RewardConfig.v4(RewardKind.V4_BASE_NO_L_FAIL, c))
    env.reset()
    steps = 0
    while not env.simulator.state.completed and steps < 20:
        env.step(CONTINUE_INDEX, 0.0)
        steps += 1
    t = env.simulator.state.time.value
    assert env.objective_return_value == pytest.approx(-t)
    assert env.normalized_base_return_value == pytest.approx(-t / c)
    assert env.return_value == pytest.approx(-t / c)


def test_pbrs_gamma_must_match_ppo():
    cfg = RewardConfig.v4(RewardKind.V4_PBRS, 10.0)
    cfg.assert_compatible_with_ppo_gamma(1.0)
    with pytest.raises(ValueError, match="gamma mismatch"):
        cfg.assert_compatible_with_ppo_gamma(0.99)


def test_c_train_train_only_median():
    from routing.serialize import read_jsonl

    train = read_jsonl(ROOT / "data" / "routes_v2" / "synthcharge_final" / "train" / "corpus.jsonl")
    c = compute_c_train(train)
    assert c == pytest.approx(10.0)
    # sanity: function refuses empty
    with pytest.raises(ValueError):
        compute_c_train([])


def test_reward_computer_deterministic(write_instance):
    cfg = RewardConfig.v4(RewardKind.V4_PBRS, 10.0)
    env, _, _ = _make(write_instance, cfg)
    env.reset()
    a = RewardComputer(cfg).potential(env.simulator)
    b = RewardComputer(cfg).potential(env.simulator)
    assert a == pytest.approx(b)
    assert cfg.to_dict()["kind"] == "V4_PBRS"
    assert RewardConfig.from_dict(cfg.to_dict()).kind == RewardKind.V4_PBRS


def test_v3_freeze_integrity_untouched():
    v3 = ROOT / "results" / "v3_hppo"
    lock = json.loads((v3 / "TEST_LOCK.json").read_text(encoding="utf-8"))
    consumed = json.loads((v3 / "EVALUATION_CONSUMED.json").read_text(encoding="utf-8"))
    assert consumed.get("consumed") is True
    assert (v3 / "raw" / "synthcharge_test.jsonl").is_file()
    assert lock.get("files")
    # V4 artifacts must not live inside v3_hppo
    assert not (v3 / "v4_reward").exists()


def test_figure_builder_is_pure_from_logs(tmp_path, monkeypatch):
    import importlib.util

    path = ROOT / "scripts" / "paper" / "build_v4_training_figures.py"
    spec = importlib.util.spec_from_file_location("build_v4_training_figures", path)
    figmod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(figmod)

    variant = "V4_PBRS"
    root = tmp_path
    abl = root / "results" / "v4_reward" / "ablation" / variant
    for seed in (42, 43):
        d = abl / f"seed_{seed}"
        d.mkdir(parents=True)
        rows = []
        for u in range(0, 5):
            rows.append(
                {
                    "update": u,
                    "base_episode_return_mean": -0.5 - 0.01 * u,
                    "shaped_episode_return_mean": -0.4 - 0.01 * u,
                    "policy_loss": 0.2,
                    "value_loss": 1.0 / (u + 1),
                    "grad_norm_preclip": 1.5,
                    "val_parent_balanced_feasibility": 0.5 + 0.05 * u if u % 2 == 0 else None,
                }
            )
        (d / "curves.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(figmod, "ROOT", root)
    monkeypatch.setattr(figmod, "ABLATION", root / "results" / "v4_reward" / "ablation")
    monkeypatch.setattr(figmod, "FINAL", root / "results" / "v4_reward" / "final_clean")
    monkeypatch.setattr(figmod, "FIG", root / "results_v4" / "figures")
    monkeypatch.setattr(figmod, "SEEDS", (42, 43))
    # ensure common-key fields exist for new preferred metrics
    for seed in (42, 43):
        path = abl / f"seed_{seed}" / "curves.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            row["objective_episode_return_mean"] = row["base_episode_return_mean"]
            row["normalized_base_episode_return_mean"] = row["base_episode_return_mean"]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    p1 = figmod.fig_reward_evolution(variant)
    p2 = figmod.fig_loss_evolution(variant)
    assert p1 is not None and p1.is_file()
    assert p2 is not None and p2.is_file()
    h1 = p1.read_bytes()
    figmod.fig_reward_evolution(variant)
    assert p1.read_bytes() == h1
