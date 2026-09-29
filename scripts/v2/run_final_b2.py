"""Final B2 training. Uses frozen hyperparameters. Does not read TEST routes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl.ablation import AblationConfig  # noqa: E402
from rl.ppo import PPOConfig, RL_CONFIG_DIR  # noqa: E402
from rl.train_loop import assert_learning_split, train_hybrid_ppo  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from routing.v2_corpus import global_return_scale  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
FORBIDDEN = {"c101", "c205", "r110", "r201", "rc102", "rc208"}


def _routes(dataset: str, split: str):
    assert_learning_split(split)
    if dataset == "gold":
        corpus = ROOT / "data" / "routes_v2" / "gold_official" / "corpus.jsonl"
        payload = json.loads((ROOT / "data" / "splits_v2" / f"gold_{split}.json").read_text(encoding="utf-8"))
        allowed = set(payload["instance_ids"])
        routes = [route for route in read_jsonl(corpus) if route.raw_instance_id in allowed]
    elif dataset == "synthcharge":
        corpus = ROOT / "data" / "routes_v2" / "synthcharge_final" / split / "corpus.jsonl"
        routes = read_jsonl(corpus)
        test_ids = set(
            json.loads((ROOT / "data" / "splits_v2" / "synthcharge_test.json").read_text(encoding="utf-8"))["route_ids"]
        )
        if any(route.route_id in test_ids for route in routes):
            raise SystemExit("SynthCharge TEST route entered learning data")
    else:
        raise SystemExit(f"unknown dataset {dataset}")
    parents = {route.base_instance for route in routes}
    if parents & FORBIDDEN:
        raise SystemExit(f"consumed V1 TEST parent in {dataset} {split}")
    if not routes:
        raise SystemExit(f"no routes for {dataset} {split}")
    return routes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("gold", "synthcharge"))
    parser.add_argument("--method", required=True, choices=("HybridPPO", "DiscretePPO"))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if args.seed not in SEEDS:
        raise SystemExit("final training seeds are 42, 43, 44, 45, 46")
    assert_learning_split("train")
    train_routes = _routes(args.dataset, "train")
    val_routes = _routes(args.dataset, "validation")
    if args.method == "HybridPPO":
        config = PPOConfig.from_toml()
        ablation = AblationConfig(name="FULL", time_aware=True, soc_interval="continuation_to_max")
    else:
        config = PPOConfig.from_toml(RL_CONFIG_DIR / "discrete_ppo.toml")
        ablation = AblationConfig(
            name="A1", discrete_u=True, time_aware=True, soc_interval="continuation_to_max"
        )
    config.seed = int(args.seed)
    scale = global_return_scale(train_routes)
    out = ROOT / "checkpoints_v2" / "final" / args.dataset / args.method / f"seed_{args.seed}"
    manifest = train_hybrid_ppo(
        train_routes=train_routes,
        val_routes=val_routes,
        config=config,
        ablation=ablation,
        method=args.method,
        out_dir=out,
        return_scale=scale,
        learning_split="train",
    )
    print(json.dumps({"dataset": args.dataset, "method": args.method, "seed": args.seed, "return_scale": scale, "best_update": manifest.get("best_update"), "status": manifest.get("status"), "git_sha": manifest.get("git_sha"), "git_dirty": manifest.get("git_dirty")}), flush=True)


if __name__ == "__main__":
    main()
