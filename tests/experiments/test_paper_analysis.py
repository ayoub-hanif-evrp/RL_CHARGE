"""Analysis-only paper-scope guards. Do not retrain because of these tests."""

from experiments.stats import (
    exact_sign_flip_pvalue,
    exclude_non_paper_methods,
    map_ablation_method,
    permutation_pvalue,
    dedupe_soc_greedy,
    hierarchical_bootstrap_ci,
)


def test_exact_sign_flip_enumerates_all_assignments():
    p = exact_sign_flip_pvalue([1.0, 1.0, 1.0])
    assert p == 0.25
    assert permutation_pvalue([1.0, -1.0], n_perm=999) == exact_sign_flip_pvalue([1.0, -1.0])


def test_ddqn_rows_are_dropped_from_paper_analysis():
    rows = [
        {"method": "HybridPPO", "scenario": "main_test"},
        {"method": "LegacyTwoStageDDQN", "scenario": "main_test"},
    ]
    kept = exclude_non_paper_methods(rows)
    assert [r["method"] for r in kept] == ["HybridPPO"]


def test_ablation_maps_hybrid_and_discrete_to_full_and_a1():
    assert map_ablation_method("HybridPPO") == "FULL"
    assert map_ablation_method("DiscretePPO") == "A1"
    assert map_ablation_method("HybridPPO_A3") == "A3"


def test_soc_greedy_is_deduplicated_by_route_and_level():
    rows = [
        {"method": "GreedyMinimumSufficientCharge", "route_id": "r1", "min_soc_fraction": 0.05, "seed": 42},
        {"method": "GreedyMinimumSufficientCharge", "route_id": "r1", "min_soc_fraction": 0.05, "seed": 43},
        {"method": "HybridPPO", "route_id": "r1", "min_soc_fraction": 0.05, "seed": 42},
        {"method": "HybridPPO", "route_id": "r1", "min_soc_fraction": 0.05, "seed": 43},
    ]
    out = dedupe_soc_greedy(rows)
    greedy = [r for r in out if r["method"] == "GreedyMinimumSufficientCharge"]
    hybrid = [r for r in out if r["method"] == "HybridPPO"]
    assert len(greedy) == 1
    assert len(hybrid) == 2


def test_hierarchical_aggregation_uses_seed_then_parent():
    rows = []
    for seed in (42, 43):
        for parent, value in (("p1", 10.0), ("p2", 20.0)):
            rows.append({"seed": seed, "base_instance": parent, "completion_time_all_routes": value})
    ci = hierarchical_bootstrap_ci(rows, "completion_time_all_routes", n_boot=40, seed=0)
    assert ci["n_seeds"] == 2
    assert ci["resampling_unit"] == "training_seed_then_base_instance"


def test_make_figures_and_tables_isolate_scenarios():
    from data.paths import REPO_ROOT

    figures = (REPO_ROOT / "scripts" / "make_figures.py").read_text(encoding="utf-8")
    tables = (REPO_ROOT / "scripts" / "make_tables.py").read_text(encoding="utf-8")
    analyze = (REPO_ROOT / "scripts" / "analyze_results.py").read_text(encoding="utf-8")
    assert "required=True" in figures
    assert "single scenario" in figures.lower() or "Never mix scenarios" in figures
    assert "exclude_non_paper_methods" in tables
    assert "map_ablation_method" in tables
    assert "dedupe_soc_greedy" in tables
    assert "hierarchical_bootstrap_ci" in tables
    assert "official published tours" not in tables.lower()
    assert "official upstream e-VRO/frvcpy reference routes/objectives" in tables
    assert "excluded_from_paper" in tables
    assert "parent_balanced_feasibility_ci95_lo" in tables
    assert "parent_balanced_completion_all_ci95_lo" in tables
    assert "matched L/NL/VG" in tables
    assert "seed_mean_route_weighted_feasibility" in tables
    assert "exact_sign_flip" in analyze
    assert "feas_rate" in analyze
    assert "MAIN_COMPARATORS" in analyze
    assert '"FULL", "A1", "A2", "A3", "A4", "A5"' in tables or "FULL" in tables
    assert 'or ("feasible" if row.get("feasible") else "unknown")' in tables


def test_matched_terrain_groups_require_all_three_terrains():
    from experiments.stats import matched_terrain_sibling_groups
    from routing.fixed_route import FrozenRoute

    def route(route_id, terrain, customers=("C1", "C2"), vehicle=0, network="Small_Network"):
        return FrozenRoute(
            route_id=route_id,
            source_dataset="t",
            doi="t",
            raw_instance_id="c101C5",
            relative_path="x",
            network_group=network,
            terrain_variant=terrain,
            customer_distribution="c",
            schedule_type=1,
            generator="pyvrp",
            generator_version="0",
            seed=42,
            stop="MaxIterations",
            n_iterations=1,
            config_hash="h",
            physics_profile="official_evrptwgr",
            capacity_policy="file",
            distance_scale=1,
            demand_scale=1,
            rounding_policy="none",
            routing_problem="vrptw",
            python="3",
            os_name="t",
            arch="t",
            instance_sha256="h",
            vehicle_index=vehicle,
            depot_id="D0",
            customer_ids=customers,
            route_demand=1.0,
            route_distance=1.0,
            route_duration_lower_bound=1.0,
            n_customers=len(customers),
            routing_feasible=True,
            charging_feasibility_status="unverified",
            generation_runtime_s=0.0,
            routing_objective=1.0,
            base_instance="c101",
            customer_folder="5_Customers",
        )

    complete = [
        route("a_L", "L"),
        route("a_NL", "NL"),
        route("a_VG", "VG"),
        route("b_NL", "NL", customers=("C9",), network="Large_Network"),
    ]
    groups = matched_terrain_sibling_groups(complete)
    assert len(groups) == 1
    assert set(groups[0]["route_ids"].values()) == {"a_L", "a_NL", "a_VG"}


def test_seed_feasibility_summary_is_per_seed_not_pooled_unique_routes():
    from experiments.stats import seed_feasibility_summary

    rows = []
    for seed, n_feas in ((42, 2), (43, 0)):
        for i in range(3):
            rows.append(
                {
                    "seed": seed,
                    "route_id": f"r{i}",
                    "base_instance": "p1" if i < 2 else "p2",
                    "feasible": i < n_feas,
                }
            )
    summary = seed_feasibility_summary(rows)
    assert summary["n_seeds"] == 2
    assert summary["mean_feasible_routes_per_seed"] == 1.0
    assert summary["per_seed"]["42"]["n_feasible_routes"] == 2
    assert summary["per_seed"]["43"]["n_feasible_routes"] == 0


def test_logical_checkpoint_path_strips_windows_abs_path():
    from experiments.stats import logical_checkpoint_path

    raw = r"C:\Users\AYOUB\RL_CHARGE_PAPER\checkpoints\HybridPPO\seed_42\best.pt"
    assert logical_checkpoint_path(raw) == "checkpoints/HybridPPO/seed_42/best.pt"
    folder = r"C:\Users\AYOUB\RL_CHARGE_PAPER\checkpoints\HybridPPO\seed_42"
    assert logical_checkpoint_path(folder, "best.pt") == "checkpoints/HybridPPO/seed_42/best.pt"
