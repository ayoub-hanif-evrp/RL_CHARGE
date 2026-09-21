"""Evaluate named methods on a predetermined split. No route filtering.

Paper CLIs (do not mix seed groups in one --methods all run):

    python scripts/evaluate.py --split test --scenario main_test \\
        --methods hybrid_ppo,discrete_ppo,attention_ppo --seeds paper
    python scripts/evaluate.py --split test --scenario ablation \\
        --methods A1,A2,A3,A4,A5 --seeds ablation
"""

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
from experiments.isolation import require_scenario  # noqa: E402
from experiments.methods import build_from_checkpoint, build_stateless, method_display_name  # noqa: E402
from experiments.provenance import frozen_hashes, git_sha, detect_device  # noqa: E402


ALL_METHODS = [
    "greedy_min",
    "greedy_full",
    "lookahead",
    "hybrid_ppo",
    "discrete_ppo",
    "attention_ppo",
    "legacy_ddqn",
    "FULL",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
]

BASELINE_KEYS = {
    "greedy_min": "GreedyMinimumSufficientCharge",
    "greedy_full": "GreedyFullCharge",
    "lookahead": "OneStepLookahead",
}

LEARNED_KEYS = {
    "hybrid_ppo",
    "discrete_ppo",
    "attention_ppo",
    "legacy_ddqn",
    "FULL",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
}


def _checkpoint(method: str, seed: int, path: Path | None) -> Path | None:
    if path is not None:
        return path
    names = {
        "hybrid_ppo": "HybridPPO",
        "discrete_ppo": "DiscretePPO",
        "attention_ppo": "AttentionPPO",
        "legacy_ddqn": "LegacyTwoStageDDQN",
        "FULL": "ablation_FULL",
        "A1": "ablation_A1",
        "A2": "ablation_A2",
        "A3": "ablation_A3",
        "A4": "ablation_A4",
        "A5": "ablation_A5",
    }
    if method not in names:
        return None
    folder = CHECKPOINTS_DIR / names[method] / f"seed_{seed}"
    candidate = folder / "best.pt"
    if candidate.is_file():
        return candidate
    last = folder / "last.pt"
    return last if last.is_file() else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--methods", required=True, help="Comma-separated method keys. 'all' is smoke-only.")
    parser.add_argument("--seeds", required=True, help="paper | ablation | extended | comma-separated ints")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-routes", type=int, default=None)
    parser.add_argument("--network-group", default=None)
    parser.add_argument("--eval-mode", action="store_true", default=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args(argv)

    scenario = require_scenario(args.scenario)
    if args.methods == "all":
        if scenario != "smoke":
            raise SystemExit("--methods all is only allowed with --scenario smoke")
        methods = list(ALL_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    has_learned = any(m in LEARNED_KEYS for m in methods)
    has_stateless = any(m in BASELINE_KEYS for m in methods)
    if has_learned and has_stateless and scenario not in {"smoke"}:
        raise SystemExit("do not mix stateless baselines and learned methods in one paper eval")
    if args.seeds in {"paper", "extended", "ablation"}:
        from experiments.seeds import load_seed_list

        seeds = load_seed_list(args.seeds)
    else:
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds is required (no seed-0 default)")
    routes = load_split_routes(args.split, network_group=args.network_group)
    if args.max_routes is not None:
        routes = routes[: args.max_routes]
    hashes = frozen_hashes()
    run_id = args.run_id or f"eval_{scenario}_{args.split}"
    experiment_id = args.experiment_id or run_id
    context_base = {
        "git_sha": git_sha(),
        "device": detect_device(),
        "corpus_sha256": hashes["corpus.jsonl"],
        "split_sha256": hashes[f"{args.split}.json"] if f"{args.split}.json" in hashes else hashes["split_metadata.json"],
        "config_hash": None,
        "checkpoint_sha256": None,
        "scenario": scenario,
        "experiment_id": experiment_id,
    }
    all_records = []
    for method in methods:
        display = method_display_name(method)
        if method in {"FULL", "A1", "A2", "A3", "A4", "A5"}:
            display = f"HybridPPO_{method}"
        if method in BASELINE_KEYS:
            policy = build_stateless(BASELINE_KEYS[method])
            recs = evaluate_population(
                routes=routes,
                method=display,
                policy=policy,
                split=args.split,
                seed=seeds[0],
                extra_context=context_base,
                scenario=scenario,
                experiment_id=experiment_id,
            )
            all_records.extend(recs)
            print(f"{display}: {len(recs)} records", flush=True)
            continue
        for seed in seeds:
            ckpt = _checkpoint(method, seed, args.checkpoint)
            if ckpt is None or not Path(ckpt).is_file():
                print(f"skip {method} seed={seed}: no checkpoint", flush=True)
                continue
            policy = build_from_checkpoint(method, ckpt)
            from experiments.provenance import sha256_file

            ctx = dict(context_base)
            ctx["checkpoint_sha256"] = sha256_file(ckpt)
            recs = evaluate_population(
                routes=routes,
                method=display,
                policy=policy,
                split=args.split,
                seed=seed,
                extra_context=ctx,
                scenario=scenario,
                experiment_id=experiment_id,
            )
            all_records.extend(recs)
            print(f"{display} seed={seed}: {len(recs)} records", flush=True)
    path = dump_run(all_records, run_id, extra_manifest={"scenario": scenario, "experiment_id": experiment_id})
    print(f"wrote {len(all_records)} records to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
