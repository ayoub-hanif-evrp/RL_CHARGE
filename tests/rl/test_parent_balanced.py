"""Parent-balanced validation checkpoint selection."""

from experiments.evaluate import EpisodeResult
from rl.train_loop import _lexicographic_better, parent_balanced_metrics


def _rec(feasible: bool, completion: float) -> EpisodeResult:
    return EpisodeResult(
        route_id="r",
        feasible=feasible,
        completed=feasible,
        route_completion_time=completion,
        total_net_energy=0.0,
        total_distance=0.0,
        n_station_visits=0,
        return_value=-completion,
        horizon=1000.0,
    )


def test_parent_balanced_gives_equal_parent_weight():
    """10 routes on parent A (9 feasible) and 2 on parent B (0 feasible).

    Route-weighted feasibility is 9/12 = 0.75.
    Parent-balanced feasibility is mean(0.9, 0.0) = 0.45.
    """
    parents = ["big"] * 10 + ["small"] * 2
    records = [_rec(True, 10.0) for _ in range(9)] + [_rec(False, 1000.0)]
    records += [_rec(False, 1000.0), _rec(False, 1000.0)]
    balanced = parent_balanced_metrics(parents, records)
    assert sum(1 for r in records if r.feasible) / len(records) == 0.75
    assert balanced["n_parents"] == 2
    assert abs(balanced["feasibility"] - 0.45) < 1e-12


def test_checkpoint_selection_uses_parent_not_route_count():
    many_easy = parent_balanced_metrics(
        ["r102"] * 40 + ["c201"] * 2,
        [_rec(True, 100.0) for _ in range(19)]
        + [_rec(False, 500.0) for _ in range(21)]
        + [_rec(False, 500.0), _rec(False, 500.0)],
    )
    even = parent_balanced_metrics(
        ["r102"] * 4 + ["c201"] * 4,
        [_rec(True, 120.0) for _ in range(2)]
        + [_rec(False, 500.0) for _ in range(2)]
        + [_rec(True, 80.0) for _ in range(3)]
        + [_rec(False, 500.0)],
    )
    assert many_easy["feasibility"] == (19 / 40 + 0.0) / 2
    assert even["feasibility"] == (0.5 + 0.75) / 2
    assert even["feasibility"] > many_easy["feasibility"]
    assert _lexicographic_better(
        even["feasibility"],
        even["completion_all"],
        many_easy["feasibility"],
        many_easy["completion_all"],
    )
