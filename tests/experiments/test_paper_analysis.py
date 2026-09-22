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
    assert "native Montoya/FRVCP nonlinear charging sensitivity" in tables
    assert "exact_sign_flip" in analyze
    assert "feas_rate" in analyze
    assert "MAIN_COMPARATORS" in analyze
    assert '"FULL", "A1", "A2", "A3", "A4", "A5"' in tables or "FULL" in tables
    assert 'or ("feasible" if row.get("feasible") else "unknown")' in tables
