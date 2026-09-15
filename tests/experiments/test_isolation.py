"""Scenario isolation, expected counts, hierarchical pairing."""

from experiments.isolation import (
    ExpectedCountError,
    assert_no_soc_in_hybrid_main,
    filter_scenario,
    require_scenario,
    validate_expected_counts,
)
from experiments.stats import hierarchical_bootstrap_ci, paired_parent_diff_hierarchical
from rl.ablation import AblationConfig
from rl.train_loop import _lexicographic_better


def test_a2_is_arrival_to_max_not_min_to_max():
    assert AblationConfig.from_name("A2").soc_interval == "arrival_to_max"
    assert AblationConfig.from_name("FULL").soc_interval == "continuation_to_max"


def test_lexicographic_feasibility_first():
    assert _lexicographic_better(1.0, 100.0, 0.5, 10.0)
    assert not _lexicographic_better(0.4, 1.0, 0.5, 100.0)
    assert _lexicographic_better(0.5, 9.0, 0.5, 10.0)
    assert not _lexicographic_better(0.5, 10.0, 0.5, 10.0)


def test_require_scenario_and_no_mix():
    require_scenario("main_test")
    rows = [
        {"scenario": "main_test", "split": "test", "method": "HybridPPO"},
        {"scenario": "soc_reserve", "split": "test", "method": "HybridPPO", "min_soc_fraction": 0.1},
    ]
    main = filter_scenario(rows, "main_test", split="test")
    assert len(main) == 1
    assert_no_soc_in_hybrid_main(rows)


def test_expected_counts_stateless_and_learned():
    routes = ["r1", "r2"]
    rows = []
    for rid in routes:
        rows.append(
            {
                "experiment_id": "e",
                "scenario": "main_test",
                "split": "test",
                "method": "GreedyMinimumSufficientCharge",
                "seed": 0,
                "route_id": rid,
            }
        )
        for seed in (42, 43):
            rows.append(
                {
                    "experiment_id": "e",
                    "scenario": "main_test",
                    "split": "test",
                    "method": "HybridPPO",
                    "seed": seed,
                    "route_id": rid,
                    "base_instance": rid,
                    "completion_time_all_routes": float(seed),
                }
            )
    validate_expected_counts(
        rows,
        n_routes=2,
        seeds_by_method={"GreedyMinimumSufficientCharge": [0], "HybridPPO": [42, 43]},
        split="test",
        scenario="main_test",
    )
    bad = rows[:-1]
    try:
        validate_expected_counts(
            bad,
            n_routes=2,
            seeds_by_method={"HybridPPO": [42, 43]},
            split="test",
            scenario="main_test",
        )
    except ExpectedCountError:
        pass
    else:
        raise AssertionError("expected count should fail")


def test_hierarchical_bootstrap_seed_then_parent():
    rows = []
    for seed in (42, 43):
        for parent, value in (("p1", 1.0), ("p2", 3.0)):
            rows.append(
                {
                    "seed": seed,
                    "base_instance": parent,
                    "completion_time_all_routes": value + 0.1 * (seed - 42),
                }
            )
    ci = hierarchical_bootstrap_ci(rows, "completion_time_all_routes", n_boot=50, seed=0)
    assert ci["n_seeds"] == 2
    assert ci["resampling_unit"] == "training_seed_then_base_instance"
    diffs = paired_parent_diff_hierarchical(
        rows,
        [{"base_instance": "p1", "seed": 0, "completion_time_all_routes": 0.0},
         {"base_instance": "p2", "seed": 0, "completion_time_all_routes": 0.0}],
        "completion_time_all_routes",
        a_mean_over_seeds=True,
        b_mean_over_seeds=False,
    )
    assert set(diffs) == {"p1", "p2"}
