"""Evaluate greedy and lookahead baselines on a predetermined split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.batch import dump_run, evaluate_population  # noqa: E402
from experiments.dataset import load_split_routes  # noqa: E402
from experiments.methods import build_stateless  # noqa: E402
from experiments.provenance import detect_device, frozen_hashes, git_sha  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--max-routes", type=int, default=None)
    parser.add_argument("--run-id", default="baselines")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args(argv)
    from experiments.isolation import require_scenario

    scenario = require_scenario(args.scenario)
    routes = load_split_routes(args.split)
    if args.max_routes is not None:
        routes = routes[: args.max_routes]
    hashes = frozen_hashes()
    experiment_id = args.experiment_id or args.run_id
    ctx = {
        "git_sha": git_sha(),
        "device": detect_device(),
        "corpus_sha256": hashes["corpus.jsonl"],
        "split_sha256": hashes[f"{args.split}.json"],
        "config_hash": None,
        "checkpoint_sha256": None,
        "scenario": scenario,
        "experiment_id": experiment_id,
    }
    records = []
    for name in ("GreedyMinimumSufficientCharge", "GreedyFullCharge", "OneStepLookahead"):
        records.extend(
            evaluate_population(
                routes=routes,
                method=name,
                policy=build_stateless(name),
                split=args.split,
                seed=0,
                extra_context=ctx,
                scenario=scenario,
                experiment_id=experiment_id,
            )
        )
    path = dump_run(records, args.run_id, extra_manifest={"scenario": scenario, "experiment_id": experiment_id})
    print(f"wrote {len(records)} records to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
