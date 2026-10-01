"""Final B2 training. Uses frozen hyperparameters. Does not read TEST routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts" / "v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "v2"))

from final_common import METHOD_FREEZE_SHA, git_porcelain, method_tree_diff  # noqa: E402

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


def _refuse(message: str) -> None:
    raise SystemExit(message)


def _assert_frozen_start(dataset: str, method: str, seed: int) -> None:
    if git_porcelain().strip():
        _refuse("refusing to train: git working tree is dirty")
    if method_tree_diff().strip():
        _refuse("refusing to train: src/configs/tests/third_party differ from the method freeze")
    freeze = json.loads((ROOT / "results" / "v2" / "final" / "METHOD_FREEZE.json").read_text(encoding="utf-8"))
    expected = freeze["ppo_hyperparameters"]["config_sha256"]
    for rel, digest in expected.items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if got != digest:
            _refuse(f"refusing to train: config hash mismatch for {rel}")
    if freeze["V2_METHOD_FREEZE_SHA"] != METHOD_FREEZE_SHA:
        _refuse("refusing to train: METHOD_FREEZE.json SHA does not match the frozen commit")
    if seed not in SEEDS:
        _refuse("final training seeds are 42, 43, 44, 45, 46")
    if dataset not in ("gold", "synthcharge") or method not in ("HybridPPO", "DiscretePPO"):
        _refuse("dataset must be gold or synthcharge and method must be HybridPPO or DiscretePPO")
    out = ROOT / "checkpoints_v2" / "final" / dataset / method / f"seed_{seed}"
    if (out / "manifest.json").is_file() and (out / "best.pt").is_file():
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("best_update") is not None:
            _refuse(f"refusing to overwrite completed final run {out.relative_to(ROOT).as_posix()}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != METHOD_FREEZE_SHA and method_tree_diff().strip():
        _refuse("method tree is not identical to the freeze")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("gold", "synthcharge"))
    parser.add_argument("--method", required=True, choices=("HybridPPO", "DiscretePPO"))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    _assert_frozen_start(args.dataset, args.method, args.seed)
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
