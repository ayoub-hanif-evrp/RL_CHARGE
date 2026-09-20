"""Part 3B experiment tests: sampler, ablations, stats, exact, AttentionPPO, schema."""

from __future__ import annotations

import json

import pytest
import torch

from baselines.frvcp_greedy import greedy_frvcp
from baselines.frvcpy_adapter import NOT_EQUIVALENT, evrptwgr_to_frvcp_surrogate
from baselines.frvcpy_solver import frvcpy_available, optimality_gap_percent, solve_native
from conftest import node_row
from data.parser import parse_instance
from data.paths import EXTERNAL_DIR, REPO_ROOT
from domain.load_convention import LoadConvention
from exact.label_setting import solve_label_setting
from experiments.evaluate import evaluate_policy
from experiments.seeds import load_seed_list
from experiments.stats import cluster_bootstrap_ci, holm, permutation_pvalue
from physics.montoya import montoya_piecewise_model
from physics.parameters import PhysicsProfile
from rl.ablation import AblationConfig
from rl.attention_ppo import ATTENTION_LABEL, attention_policy
from rl.env import ShieldedRouteEnv
from rl.features import extract_features
from rl.policy import HybridPolicy
from rl.ppo import PPOConfig
from rl.sampler import HierarchicalSampler
from routing.fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute


def _route(instance, customers, terrain="Level", vehicle=0, base=None):
    return FrozenRoute(
        route_id=f"{instance.metadata.instance_id}__v{vehicle}",
        source_dataset="EVRPTW-GR",
        doi="10.17632/srfdbp2twv.1",
        raw_instance_id=instance.metadata.instance_id,
        relative_path=instance.metadata.relative_path,
        network_group=instance.metadata.network_group.value,
        terrain_variant=terrain,
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
        vehicle_index=vehicle,
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
        base_instance=base or instance.metadata.base_instance,
        customer_folder=instance.metadata.customer_folder,
    )


def _env(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 2.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    route = _route(instance, ("C1",))
    env = ShieldedRouteEnv(instance, route, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP)
    return env, instance, profile, route


def test_seed_lists():
    paper = load_seed_list("paper")
    extended = load_seed_list("extended")
    ablation = load_seed_list("ablation")
    assert paper == [42, 43, 44, 45, 46]
    assert extended[:5] == paper
    assert len(extended) == 10
    assert ablation == [42, 43, 44]
    assert load_seed_list("42,99") == [42, 99]
    assert load_seed_list("42") == [42]


def test_hierarchical_sampler_equal_parents(write_instance):
    env, instance, _, _ = _env(write_instance)
    routes = [
        _route(instance, ("C1",), terrain="Level", vehicle=0, base="p1"),
        _route(instance, ("C1",), terrain="Nearly_Level", vehicle=0, base="p1"),
        _route(instance, ("C1",), terrain="Level", vehicle=0, base="p2"),
    ]
    sampler = HierarchicalSampler(routes, seed=0)
    counts = {"p1": 0, "p2": 0}
    for _ in range(400):
        counts[sampler.sample().parent] += 1
    assert abs(counts["p1"] - counts["p2"]) < 80
    assert sampler.name == "parent_then_route_then_terrain"


def test_ablation_a4_zeros_payload_and_a3_skips_remaining(write_instance):
    env, _, _, _ = _env(write_instance)
    full = extract_features(env.simulator, use_terrain_load_features=True)
    hidden = extract_features(env.simulator, use_terrain_load_features=False)
    assert hidden.global_features[2] == 0.0
    assert hidden.global_features[5] == 0.0
    skipped = extract_features(env.simulator, use_remaining_route=False)
    assert float(skipped.remaining.sum()) == 0.0
    assert AblationConfig.from_name("A1").discrete_u
    assert AblationConfig.from_name("A2").soc_interval == "arrival_to_max"
    assert AblationConfig.from_name("A5").station_encoder == "pool"


def test_attention_policy_has_type_embedding_and_is_not_a_paper_clone(write_instance):
    env, _, _, _ = _env(write_instance)
    policy = attention_policy(d_model=16, n_heads=4, n_layers=1)
    assert policy.encoder.node_type_embedding
    assert policy.encoder.type_embed is not None
    features = env.reset()
    with torch.no_grad():
        out = policy.act(features, eval_mode=True)
    assert torch.isfinite(out.value).all()
    assert "not a reproduction" in ATTENTION_LABEL.lower()


def test_label_setting_finishes_or_times_out_without_fake_exact(write_instance):
    env, instance, profile, route = _env(write_instance)
    result = solve_label_setting(instance, route, profile=profile, max_expansions=200)
    assert result.status in {"optimal_for_action_set", "infeasible", "timeout"}
    assert result.exact_for == "restricted_continuation_or_full_soc"
    assert result.status != "exact"
    if result.status == "timeout":
        assert result.route_completion_time is None or result.feasible
    if result.status == "optimal_for_action_set":
        assert result.feasible
        assert result.route_completion_time is not None


def test_episode_result_uses_horizon_for_failures(write_instance):
    env, instance, _, route = _env(write_instance)

    def fail(sim, eval_mode=True):
        from baselines.common import BaselineResult

        return BaselineResult(False, False, 0.0, 0)

    result = evaluate_policy(instance=instance, route=route, policy=fail)
    assert result.completion_time_all_routes() == pytest.approx(float(instance.depot.due_date))
    rec = result.to_record(method="x", split="test", seed=0)
    assert rec["completion_time_all_routes"] == result.horizon


def test_cluster_bootstrap_and_holm():
    rows = [
        {"base_instance": "a", "v": 1.0},
        {"base_instance": "a", "v": 3.0},
        {"base_instance": "b", "v": 2.0},
        {"base_instance": "b", "v": 2.0},
    ]
    ci = cluster_bootstrap_ci(rows, "v", n_boot=200, seed=0)
    assert ci["n_clusters"] == 2
    assert ci["mean"] == pytest.approx(2.0)
    adjusted = holm([("x", 0.01), ("y", 0.04)])
    assert adjusted[0][2] <= adjusted[1][2]
    p = permutation_pvalue([1.0, 1.0, 1.0], n_perm=200, seed=0)
    assert 0.0 < p <= 1.0


def test_montoya_sensitivity_is_not_original_evrptwgr():
    model = montoya_piecewise_model("fast", scale_full_charge_time=10.0)
    assert "not_original" in model.name
    assert model.soc_knots[0] == 0.0
    assert model.soc_knots[-1] == pytest.approx(1.0)
    assert model.cumulative_time_knots[-1] == pytest.approx(10.0)


def test_frvcpy_surrogate_still_not_equivalent(write_instance):
    _, instance, _, _ = _env(write_instance)
    assert evrptwgr_to_frvcp_surrogate(instance).equivalent_to_evrptwgr == NOT_EQUIVALENT


def test_tiny_native_frvcp_greedy_and_optional_solver():
    path = EXTERNAL_DIR / "frvcpy" / "tiny-instance.json"
    instance = json.loads(path.read_text(encoding="utf-8"))
    greedy = greedy_frvcp(instance, [0, 1, 0], 16.0, full_charge=True)
    assert greedy["feasible"]
    gap = optimality_gap_percent(12.0, 10.0)
    assert gap == pytest.approx(20.0)
    solved = solve_native(instance, [0, 1, 0], 16.0)
    if not frvcpy_available():
        assert solved.status == "frvcpy_not_installed"
    else:
        assert solved.status in {"optimal", "frvcpy_not_installed"}


def test_official_frvcp_benchmark_is_isolated_from_splits():
    bench = EXTERNAL_DIR / "frvcpy" / "benchmark"
    assert (EXTERNAL_DIR / "frvcpy" / "tiny-instance.json").is_file()
    assert (EXTERNAL_DIR / "frvcpy" / "routes.json").is_file()
    assert (bench / "README.md").is_file()
    assert (bench / "hashes.json").is_file()
    assert (bench / "routes.json").is_file()
    xmls = list((bench / "xml").glob("*.xml"))
    assert len(xmls) >= 20
    payload = json.loads((bench / "routes.json").read_text(encoding="utf-8"))
    assert payload["never_join_to_evrptwgr_splits"] is True
    assert payload["equivalent_to_evrptwgr"] == "native_frvcp"
    assert payload["official_published_tours"] is False
    assert len(payload["routes"]) >= 20
    testdata = EXTERNAL_DIR / "frvcpy" / "testdata.json"
    assert testdata.is_file()
    assert len(json.loads(testdata.read_text(encoding="utf-8"))) >= 100
    text = (bench / "README.md").read_text(encoding="utf-8").lower()
    assert "never" in text
    assert "2016-0020" in text or "montoya" in text


def test_attention_and_pool_encoders_run(write_instance):
    env, _, _, _ = _env(write_instance)
    for name in ("FULL", "A3", "A5", "ATTENTION"):
        policy = HybridPolicy(d_model=16, n_heads=4, n_layers=1, ablation=AblationConfig.from_name(name))
        with torch.no_grad():
            out = policy.act(env.reset(), eval_mode=True)
        assert torch.isfinite(out.log_prob).all()


def test_hybrid_ppo_paper_config_exists():
    cfg = PPOConfig.from_toml(REPO_ROOT / "configs" / "rl" / "hybrid_ppo.toml")
    assert cfg.budget_updates == 400
    assert cfg.rollout_steps == 256
    assert cfg.eval_interval == 10
    assert cfg.early_stopping_patience == 20
    paper = PPOConfig.from_toml(REPO_ROOT / "configs" / "rl" / "hybrid_ppo_paper.toml")
    assert paper.budget_updates == 400
    assert paper.early_stopping_patience == 20
    discrete = PPOConfig.from_toml(REPO_ROOT / "configs" / "rl" / "discrete_ppo.toml")
    attention = PPOConfig.from_toml(REPO_ROOT / "configs" / "rl" / "attention_ppo.toml")
    for other in (discrete, attention):
        assert other.budget_updates == 400
        assert other.rollout_steps == 256
        assert other.eval_interval == 10
        assert other.early_stopping_patience == 20
    smoke = PPOConfig.from_toml(REPO_ROOT / "configs" / "rl" / "hybrid_ppo_smoke.toml")
    assert smoke.budget_updates == 2
    pilot = PPOConfig.from_toml(REPO_ROOT / "configs" / "rl" / "hybrid_ppo_pilot.toml")
    assert pilot.budget_updates == 400
    assert pilot.early_stopping_patience == 20
