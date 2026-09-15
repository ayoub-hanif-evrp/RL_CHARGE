"""Exact label-setting on small frozen EVRPTW-GR routes. Timeouts are not called exact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exact.label_setting import solve_label_setting  # noqa: E402
from experiments.batch import dump_run  # noqa: E402
from experiments.dataset import load_split_routes, parse_route_instance  # noqa: E402
from experiments.provenance import detect_device, frozen_hashes, git_sha  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-customers", type=int, default=5)
    parser.add_argument("--max-expansions", type=int, default=20_000)
    parser.add_argument("--max-routes", type=int, default=None)
    parser.add_argument("--run-id", default="exact_small")
    args = parser.parse_args(argv)
    routes = load_split_routes(args.split, max_customers=args.max_customers)
    if args.max_routes is not None:
        routes = routes[: args.max_routes]
    hashes = frozen_hashes()
    records = []
    for route in routes:
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance)
        result = solve_label_setting(
            instance, route, profile=profile, max_expansions=args.max_expansions
        )
        exact = result.status == "optimal_for_action_set"
        records.append(
            {
                "method": "LabelSettingRCSPP",
                "route_id": route.route_id,
                "base_instance": route.base_instance,
                "terrain": route.terrain_variant,
                "network_group": route.network_group,
                "split": args.split,
                "seed": 0,
                "feasible": result.feasible,
                "reason": result.status,
                "route_completion_time": result.route_completion_time,
                "completion_time_all_routes": result.route_completion_time
                if result.feasible and exact
                else float(instance.depot.due_date),
                "n_station_visits": result.n_station_visits,
                "n_expansions": result.n_expansions,
                "exact": exact,
                "n_customers": route.n_customers,
                "git_sha": git_sha(),
                "device": detect_device(),
                "corpus_sha256": hashes["corpus.jsonl"],
                "split_sha256": hashes[f"{args.split}.json"],
            }
        )
    path = dump_run(records, args.run_id)
    n_timeout = sum(1 for r in records if r["reason"] == "timeout")
    print(f"wrote {len(records)} records timeouts={n_timeout} to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
