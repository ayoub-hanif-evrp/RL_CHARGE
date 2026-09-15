"""SOC-reserve sensitivity. Not used for model selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import CHECKPOINTS_DIR  # noqa: E402
from experiments.batch import dump_run, evaluate_population  # noqa: E402
from experiments.dataset import load_split_routes  # noqa: E402
from experiments.methods import build_from_checkpoint, build_stateless  # noqa: E402
from experiments.provenance import detect_device, frozen_hashes, git_sha  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="0,0.05,0.10,0.15")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario", default="soc_reserve")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-routes", type=int, default=None)
    parser.add_argument("--run-id", default="soc_reserve")
    args = parser.parse_args(argv)
    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    routes = load_split_routes(args.split)
    if args.max_routes is not None:
        routes = routes[: args.max_routes]
    hashes = frozen_hashes()
    ckpt = CHECKPOINTS_DIR / "HybridPPO" / f"seed_{args.seed}" / "best.pt"
    if not ckpt.is_file():
        ckpt = CHECKPOINTS_DIR / "HybridPPO" / f"seed_{args.seed}" / "last.pt"
    hybrid = build_from_checkpoint("hybrid_ppo", ckpt) if ckpt.is_file() else None
    greedy = build_stateless("GreedyMinimumSufficientCharge")
    records = []
    for level in levels:
        ctx = {
            "git_sha": git_sha(),
            "device": detect_device(),
            "corpus_sha256": hashes["corpus.jsonl"],
            "min_soc_fraction": level,
            "note": "soc_reserve_sensitivity_not_used_for_selection",
            "scenario": args.scenario,
            "experiment_id": args.run_id,
        }
        records.extend(
            evaluate_population(
                routes=routes,
                method="GreedyMinimumSufficientCharge",
                policy=greedy,
                split=args.split,
                seed=0,
                extra_context=ctx,
                min_soc_fraction=level,
                scenario=args.scenario,
                experiment_id=args.run_id,
            )
        )
        if hybrid is not None:
            records.extend(
                evaluate_population(
                    routes=routes,
                    method="HybridPPO",
                    policy=hybrid,
                    split=args.split,
                    seed=args.seed,
                    extra_context=ctx,
                    min_soc_fraction=level,
                    scenario=args.scenario,
                    experiment_id=args.run_id,
                )
            )
    path = dump_run(records, args.run_id)
    print(f"wrote {len(records)} records to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
