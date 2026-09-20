"""Restricted TRAIN/VAL label-setting search. TEST is forbidden.

This is a lower bound on charging feasibility for CONTINUE plus linear
charging extreme points {arrival SOC, max SOC}. Timeouts and infeasible
statuses are not claims about the continuous Hybrid PPO action set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exact.label_setting import solve_label_setting  # noqa: E402
from experiments.dataset import load_split_routes, parse_route_instance  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from routing.serialize import canonical_dumps  # noqa: E402


FORBIDDEN = {"test", "testing"}


def _reject_test(split: str) -> None:
    if str(split).lower() in FORBIDDEN:
        raise SystemExit("TEST is forbidden")


def _parent_balanced_feasibility(rows: list[dict]) -> float:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_instance"])].append(bool(row["feasible"]))
    if not grouped:
        return 0.0
    rates = [sum(vals) / max(len(vals), 1) for vals in grouped.values()]
    return float(sum(rates) / len(rates))


def _record(route, result, elapsed_s: float) -> dict:
    return {
        "route_id": route.route_id,
        "base_instance": route.base_instance,
        "n_customers": route.n_customers,
        "network_group": route.network_group,
        "status": result.status,
        "feasible": bool(result.feasible),
        "route_completion_time": result.route_completion_time,
        "n_expansions": result.n_expansions,
        "n_station_visits": result.n_station_visits,
        "exact_for": result.exact_for,
        "exact": False,
        "elapsed_s": elapsed_s,
    }


def _summarize(split: str, rows: list[dict]) -> dict:
    n = len(rows)
    feas = [r for r in rows if r["feasible"]]
    statuses = Counter(r["status"] for r in rows)
    return {
        "split": split,
        "n_routes": n,
        "n_feasible": len(feas),
        "feasibility": len(feas) / max(n, 1),
        "parent_balanced_feasibility": _parent_balanced_feasibility(rows),
        "n_timeout": statuses.get("timeout", 0),
        "n_infeasible_for_action_set": statuses.get("infeasible", 0),
        "n_optimal_for_action_set": statuses.get("optimal_for_action_set", 0),
        "n_feasible_for_action_set": statuses.get("feasible_for_action_set", 0),
        "status_histogram": dict(statuses),
        "note": (
            "Feasible counts are a lower bound for CONTINUE + {arrival, max} "
            "SOC only, stopping at the first feasible label. Infeasible/timeout "
            "does not prove continuous infeasibility."
        ),
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--max-expansions", type=int, default=20_000)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--max-customers", type=int, default=None)
    parser.add_argument("--max-routes", type=int, default=None)
    parser.add_argument("--network-group", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "pilot" / "correctness_audit" / "restricted_search.json",
    )
    args = parser.parse_args(argv)
    splits = [part.strip() for part in str(args.splits).split(",") if part.strip()]
    payload = {
        "test_used": False,
        "max_expansions": args.max_expansions,
        "max_seconds": args.max_seconds,
        "stop_at_first_feasible": True,
        "exact": False,
        "splits": {},
    }
    print(
        f"restricted search splits={splits} max_expansions={args.max_expansions} "
        f"max_seconds={args.max_seconds} stop_at_first_feasible=True exact=False",
        flush=True,
    )
    for split in splits:
        _reject_test(split)
        routes = load_split_routes(
            split,
            max_customers=args.max_customers,
            network_group=args.network_group,
        )
        if args.max_routes is not None:
            routes = routes[: args.max_routes]
        rows = []
        for i, route in enumerate(routes, start=1):
            instance = parse_route_instance(route)
            profile = PhysicsProfile.from_instance(instance)
            t0 = time.perf_counter()
            result = solve_label_setting(
                instance,
                route,
                profile=profile,
                max_expansions=args.max_expansions,
                stop_at_first_feasible=True,
                max_seconds=args.max_seconds,
            )
            elapsed = time.perf_counter() - t0
            rows.append(_record(route, result, elapsed))
            print(
                f"{split} {i}/{len(routes)} {route.route_id} "
                f"status={result.status} feas={int(result.feasible)} "
                f"exp={result.n_expansions} s={elapsed:.2f}",
                flush=True,
            )
            payload["splits"][split] = _summarize(split, rows)
            payload["splits"][split]["routes"] = rows
            _write(args.out, payload)
        payload["splits"][split] = _summarize(split, rows)
        payload["splits"][split]["routes"] = rows
        _write(args.out, payload)
    compact = {
        k: {kk: vv for kk, vv in v.items() if kk != "routes"}
        for k, v in payload["splits"].items()
    }
    print(json.dumps(compact, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
