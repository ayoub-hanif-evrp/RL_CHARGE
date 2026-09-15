"""Scenario isolation and expected-count checks. Paper tables never mix scenarios."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Optional, Sequence


SCENARIOS = (
    "main_test",
    "ablation",
    "soc_reserve",
    "terrain_analysis",
    "frvcpy_native",
    "exact_small",
    "nonlinear_sensitivity",
    "smoke",
    "pilot",
)

STATELESS_METHODS = {
    "GreedyMinimumSufficientCharge",
    "GreedyFullCharge",
    "OneStepLookahead",
    "FRVCPGreedyMin",
    "FRVCPGreedyFull",
    "frvcpy_Solver",
    "LabelSettingRCSPP",
}

LEARNED_METHODS = {
    "HybridPPO",
    "DiscretePPO",
    "AttentionPPO",
    "LegacyTwoStageDDQN",
    "HybridPPO_FULL",
    "HybridPPO_A1",
    "HybridPPO_A2",
    "HybridPPO_A3",
    "HybridPPO_A4",
    "HybridPPO_A5",
}

REQUIRED_FIELDS = ("experiment_id", "scenario", "split", "method", "seed")


class ExpectedCountError(ValueError):
    pass


def require_scenario(scenario: str) -> str:
    scenario = str(scenario)
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    return scenario


def filter_scenario(rows: Sequence[dict], scenario: str, split: Optional[str] = None) -> List[dict]:
    scenario = require_scenario(scenario)
    out = [row for row in rows if row.get("scenario") == scenario]
    if split is not None:
        out = [row for row in out if row.get("split") == split]
    return out


def n_routes_for_split(split: str) -> int:
    from experiments.dataset import load_split_routes

    return len(load_split_routes(split))


def validate_expected_counts(
    rows: Sequence[dict],
    *,
    n_routes: int,
    seeds_by_method: Mapping[str, Sequence[int]],
    split: str,
    scenario: str,
) -> None:
    """Fail loud if row counts do not match the predetermined population."""
    require_scenario(scenario)
    filtered = [row for row in rows if row.get("scenario") == scenario and row.get("split") == split]
    errors: List[str] = []
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for row in filtered:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"{row.get('method')} missing fields {missing}")
        by_method[str(row.get("method"))].append(row)
    for method, seeds in seeds_by_method.items():
        group = by_method.get(method, [])
        if method in STATELESS_METHODS or method in {
            "greedy_min",
            "greedy_full",
            "lookahead",
        }:
            expected = int(n_routes)
        else:
            expected = int(n_routes) * len(tuple(seeds))
        if len(group) != expected:
            errors.append(
                f"{method}: expected {expected} rows "
                f"(n_routes={n_routes} × seeds={list(seeds) if method not in STATELESS_METHODS else 1}), "
                f"got {len(group)}"
            )
            continue
        route_seed = Counter((row.get("route_id"), row.get("seed")) for row in group)
        if method in STATELESS_METHODS:
            if any(count != 1 for count in route_seed.values()):
                errors.append(f"{method}: duplicate route rows")
            if len({row.get("route_id") for row in group}) != n_routes:
                errors.append(f"{method}: unique route_id count != n_routes")
        else:
            expected_pairs = expected
            if len(route_seed) != expected_pairs:
                errors.append(f"{method}: unique (route_id, seed) != {expected_pairs}")
    if errors:
        raise ExpectedCountError("; ".join(errors))


def assert_no_soc_in_hybrid_main(rows: Sequence[dict]) -> None:
    main = filter_scenario(rows, "main_test", split="test")
    leaked = [
        row
        for row in main
        if row.get("method") == "HybridPPO" and row.get("min_soc_fraction") not in {None, 0, 0.0}
    ]
    if leaked:
        raise ExpectedCountError("SOC-reserve rows leaked into HybridPPO main_test")
