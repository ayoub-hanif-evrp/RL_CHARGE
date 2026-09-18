"""Post-fix TRAIN/VAL deterministic diagnostics. Does not use TEST."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.greedy import GreedyFullCharge, GreedyMinimumSufficientCharge
from baselines.lookahead import OneStepLookahead
from domain.load_convention import LoadConvention
from experiments.dataset import load_split_routes, parse_route_instance
from physics.parameters import PhysicsProfile
from routing.serialize import canonical_dumps
from simulation.actions import ChargeAction, ContinueAction
from simulation.feasibility import InfeasibilityReason
from simulation.progress import failure_step_reward
from simulation.shield import action_from_discrete, evaluate_shield
from simulation.simulator import FixedRouteSimulator

OUT = Path(__file__).resolve().parent
FORBIDDEN = {"test", "testing"}


def _reject_test(split: str) -> None:
    if str(split).lower() in FORBIDDEN:
        raise SystemExit("TEST is forbidden")


def _make_sim(route) -> FixedRouteSimulator:
    instance = parse_route_instance(route)
    profile = PhysicsProfile.from_instance(instance)
    return FixedRouteSimulator(
        instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )


def _reason_name(value) -> str:
    if value is None:
        return "none"
    return getattr(value, "value", str(value))


def _chooser(method) -> object:
    name = method.name
    if name == "GreedyMinimumSufficientCharge":
        from baselines.greedy import _greedy_choose

        return lambda sim: _greedy_choose(sim, 0.0)
    if name == "GreedyFullCharge":
        from baselines.greedy import _greedy_choose

        return lambda sim: _greedy_choose(sim, 1.0)
    if name == "OneStepLookahead":
        from baselines.lookahead import _lookahead_choose

        return _lookahead_choose
    raise TypeError(name)


def instrumented_rollout(simulator: FixedRouteSimulator, choose) -> dict:
    h = simulator.horizon
    g = 0.0
    steps = 0
    n_continue = 0
    n_charge = 0
    n_revisits = 0
    consecutive_same = 0
    last_station = None
    loop_guard = False
    reason = None
    max_steps = 10_000
    while not simulator.state.completed and steps < max_steps:
        shield = evaluate_shield(simulator)
        if not shield.any_legal:
            g += failure_step_reward(simulator)
            reason = "NO_FEASIBLE_ACTION"
            break
        t_before = simulator.state.time.value
        discrete, u = choose(simulator)
        if discrete >= len(shield.mask) or not shield.mask[discrete]:
            g += failure_step_reward(simulator)
            reason = "MASK_VIOLATION"
            break
        action = action_from_discrete(simulator, discrete, u)
        visited = set(simulator.state.stations_visited_since_progress)
        is_revisit = isinstance(action, ChargeAction) and action.station_id in visited
        is_repeat = (
            isinstance(action, ChargeAction)
            and last_station is not None
            and action.station_id == last_station
        )
        if is_revisit:
            n_revisits += 1
        result = simulator.step(action)
        t_after = simulator.state.time.value
        if not result.feasible:
            g += failure_step_reward(simulator)
            reason = _reason_name(result.reason)
            if result.reason is InfeasibilityReason.LOOP_GUARD:
                loop_guard = True
            break
        g += -(t_after - t_before)
        steps += 1
        if isinstance(action, ContinueAction):
            n_continue += 1
            last_station = None
        else:
            n_charge += 1
            last_station = action.station_id
            if is_repeat:
                consecutive_same += 1
        if (
            simulator.state.n_station_visits_since_last_customer
            >= simulator.profile.loop_guard_station_visits
        ):
            loop_guard = True
    completed = bool(simulator.state.completed)
    if completed:
        reason = None
    metrics = simulator.metrics(feasible=completed)
    return {
        "feasible": completed,
        "completed": completed,
        "return_value": g,
        "horizon": h,
        "reason": reason,
        "n_steps": steps,
        "n_continue": n_continue,
        "n_charge": n_charge,
        "n_station_visits": metrics.number_of_station_visits,
        "n_station_revisits": n_revisits,
        "consecutive_same_station": consecutive_same,
        "loop_guard_hit": loop_guard or reason == "LOOP_GUARD",
        "completion_time": metrics.route_completion_time.value if completed else h,
    }


def parent_balanced(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_instance"])].append(row)
    if not grouped:
        return {"feasibility": 0.0, "completion_all": 0.0, "n_parents": 0, "per_parent": {}}
    parent_feas = []
    parent_comp = []
    per_parent = {}
    for parent, recs in grouped.items():
        n = len(recs)
        feas = sum(1 for r in recs if r["feasible"]) / n
        comp = sum(r["completion_time"] for r in recs) / n
        parent_feas.append(feas)
        parent_comp.append(comp)
        per_parent[parent] = {
            "n_routes": n,
            "n_feasible": sum(1 for r in recs if r["feasible"]),
            "feasibility": feas,
            "mean_completion_all": comp,
        }
    return {
        "feasibility": float(sum(parent_feas) / len(parent_feas)),
        "completion_all": float(sum(parent_comp) / len(parent_comp)),
        "n_parents": len(grouped),
        "per_parent": per_parent,
    }


def summarize(rows: list[dict], split: str, method: str) -> dict:
    group = [r for r in rows if r["split"] == split and r["method"] == method]
    n = len(group)
    feas = [r for r in group if r["feasible"]]
    fails = [r for r in group if not r["feasible"]]
    visits = [r["n_station_visits"] for r in group]
    revisits = [r["n_station_revisits"] for r in group]
    reasons = Counter(r["reason"] or "success" for r in group)
    balanced = parent_balanced(group)
    return {
        "split": split,
        "method": method,
        "n_routes": n,
        "n_feasible": len(feas),
        "feasibility": len(feas) / max(n, 1),
        "parent_balanced_feasibility": balanced["feasibility"],
        "parent_balanced_completion_all": balanced["completion_all"],
        "n_parents": balanced["n_parents"],
        "per_parent": balanced["per_parent"],
        "mean_completion_feasible": (
            statistics.mean(r["completion_time"] for r in feas) if feas else None
        ),
        "failure_reasons": dict(Counter(r["reason"] for r in fails)),
        "reason_histogram_including_success": dict(reasons),
        "mean_station_visits": statistics.mean(visits) if visits else 0.0,
        "sum_station_revisits": int(sum(revisits)),
        "n_routes_with_station_revisit": sum(1 for r in group if r["n_station_revisits"] > 0),
        "sum_consecutive_same_station": sum(r["consecutive_same_station"] for r in group),
        "n_loop_guard": sum(1 for r in group if r["loop_guard_hit"]),
        "n_zero_charge_noop": sum(1 for r in fails if r["reason"] == "ZERO_CHARGE_NOOP"),
        "n_no_feasible_action": sum(1 for r in fails if r["reason"] == "NO_FEASIBLE_ACTION"),
        "n_station_revisit_fail": sum(1 for r in fails if r["reason"] == "STATION_REVISIT"),
    }


def union_feasibility(rows: list[dict], split: str) -> dict:
    by_route = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_route[row["route_id"]].append(row["feasible"])
    n = len(by_route)
    n_union = sum(1 for flags in by_route.values() if any(flags))
    return {
        "split": split,
        "n_routes": n,
        "n_union_feasible": n_union,
        "union_rate": n_union / max(n, 1),
        "note": (
            "Diagnostic lower bound only. Routes outside the union are not "
            "claimed infeasible. This is not an exact feasibility oracle."
        ),
    }


def main() -> int:
    methods = [
        GreedyMinimumSufficientCharge(),
        GreedyFullCharge(),
        OneStepLookahead(),
    ]
    rows = []
    for split in ("train", "validation"):
        _reject_test(split)
        routes = load_split_routes(split)
        for method in methods:
            choose = _chooser(method)
            for route in routes:
                stats = instrumented_rollout(_make_sim(route), choose)
                rows.append(
                    {
                        "split": split,
                        "method": method.name,
                        "route_id": route.route_id,
                        "base_instance": route.base_instance,
                        "n_customers": route.n_customers,
                        "network_group": route.network_group,
                        **stats,
                    }
                )
            print(f"baselines {split} {method.name}: {len(routes)} routes", flush=True)

    summaries = [
        summarize(rows, split, method.name)
        for split in ("train", "validation")
        for method in methods
    ]
    payload = {
        "label": "post_action_reward_fix_deterministic_diagnostics",
        "test_used": False,
        "methods": [m.name for m in methods],
        "summaries": summaries,
        "union": {
            "train": union_feasibility(rows, "train"),
            "validation": union_feasibility(rows, "validation"),
        },
        "note": (
            "Union of the three heuristics is a diagnostic lower bound, "
            "not an exact feasibility oracle."
        ),
    }
    (OUT / "diagnostics_summary.json").write_text(
        canonical_dumps(payload) + "\n", encoding="utf-8"
    )
    fieldnames = [
        "split",
        "method",
        "route_id",
        "base_instance",
        "n_customers",
        "network_group",
        "feasible",
        "reason",
        "completion_time",
        "n_station_visits",
        "n_station_revisits",
        "consecutive_same_station",
        "loop_guard_hit",
        "return_value",
        "horizon",
        "n_continue",
        "n_charge",
    ]
    with (OUT / "baseline_route_level.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps({"union": payload["union"], "n_summaries": len(summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
