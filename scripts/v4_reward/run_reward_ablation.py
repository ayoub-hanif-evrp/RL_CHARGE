"""V4 reward ablation on SynthCharge TRAIN/VAL (frozen V3 architecture).

Isolates reward_kind only. Does NOT touch V3 TEST or frozen V3 artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl.ablation import AblationConfig  # noqa: E402
from rl.checkpoint import load_hybrid_actor  # noqa: E402
from rl.ppo import PPOConfig  # noqa: E402
from rl.rewards import RewardConfig, RewardKind, compute_c_train  # noqa: E402
from rl.train_loop import assert_learning_split, evaluate_routes, train_hybrid_ppo  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from routing.v2_corpus import global_return_scale  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
OUT_ROOT = ROOT / "results" / "v4_reward" / "ablation"
CKPT_ROOT = ROOT / "checkpoints_v4" / "reward_ablation"
VARIANTS = {
    "V3_TIME": {"kind": RewardKind.V3_TIME, "use_ppo_scale": True},
    "V4_BASE": {"kind": RewardKind.V4_BASE, "use_ppo_scale": False},
    "V4_PBRS": {"kind": RewardKind.V4_PBRS, "use_ppo_scale": False},
    "V4_BASE_NO_L_FAIL": {"kind": RewardKind.V4_BASE_NO_L_FAIL, "use_ppo_scale": False},
}


def _routes(split: str):
    assert_learning_split(split)
    corpus = ROOT / "data" / "routes_v2" / "synthcharge_final" / split / "corpus.jsonl"
    routes = read_jsonl(corpus)
    test_ids = set(
        json.loads((ROOT / "data" / "splits_v2" / "synthcharge_v3_test.json").read_text(encoding="utf-8"))["route_ids"]
    )
    # also block older synthcharge_test ids if present
    older = ROOT / "data" / "splits_v2" / "synthcharge_test.json"
    if older.is_file():
        test_ids |= set(json.loads(older.read_text(encoding="utf-8"))["route_ids"])
    if any(route.route_id in test_ids for route in routes):
        raise SystemExit("TEST route entered V4 learning data")
    return routes


def run_one(variant: str, seed: int, *, budget_updates: int | None = None) -> None:
    if variant not in VARIANTS or seed not in SEEDS:
        raise SystemExit("bad variant/seed")
    spec = VARIANTS[variant]
    out_ckpt = CKPT_ROOT / variant / f"seed_{seed}"
    dest = OUT_ROOT / variant / f"seed_{seed}"
    if (dest / "validation.json").is_file() and (out_ckpt / "best.pt").is_file():
        print(f"skip {variant} seed={seed}", flush=True)
        return

    train_routes = _routes("train")
    val_routes = _routes("validation")
    c_train = compute_c_train(train_routes)
    if spec["kind"] == RewardKind.V3_TIME:
        reward = RewardConfig.v3_time()
    else:
        reward = RewardConfig.v4(spec["kind"], c_train)

    config = PPOConfig.from_toml(ROOT / "configs" / "rl" / "hybrid_ppo_paper.toml")
    config.seed = int(seed)
    if budget_updates is not None:
        config.budget_updates = int(budget_updates)
        config.eval_interval = max(1, min(int(config.eval_interval), int(budget_updates)))

    ablation = AblationConfig(name="FULL", time_aware=True, soc_interval="continuation_to_max")
    ppo_scale = global_return_scale(train_routes) if spec["use_ppo_scale"] else 1.0

    manifest = train_hybrid_ppo(
        train_routes=train_routes,
        val_routes=val_routes,
        config=config,
        ablation=ablation,
        method=f"V4_REWARD_{variant}",
        out_dir=out_ckpt,
        return_scale=ppo_scale,
        reward_config=reward,
    )
    actor = load_hybrid_actor(out_ckpt / "best.pt" if (out_ckpt / "best.pt").is_file() else out_ckpt / "last.pt")
    actor.ablation = ablation
    val = evaluate_routes(val_routes, actor)
    dest.mkdir(parents=True, exist_ok=True)
    if (out_ckpt / "curves.jsonl").is_file():
        (dest / "curves.jsonl").write_text((out_ckpt / "curves.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    if (out_ckpt / "reward_config.json").is_file():
        (dest / "reward_config.json").write_text((out_ckpt / "reward_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "stage": "V4_REWARD",
        "label": "TRAIN/VAL reward ablation — NOT confirmatory TEST",
        "variant": variant,
        "seed": seed,
        "reward_kind": reward.kind.value,
        "c_train": float(c_train),
        "c_train_rule": reward.c_train_rule,
        "ppo_return_scale": float(ppo_scale),
        "architecture": "V3_FA_HPPO_FULL_time_aware",
        "dataset": "synthcharge_final",
        "parent_balanced_val_feasibility": val["parent_balanced_feasibility"],
        "parent_balanced_completion_all": val["parent_balanced_completion_all"],
        "route_weighted_feasibility": val["feasibility"],
        "best_update": manifest.get("best_update"),
        "status": manifest.get("status"),
        "confirmatory_test_consumed": False,
        "not_v3_test": True,
    }
    (dest / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--budget-updates", type=int, default=None, help="optional override for smoke runs")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    # freeze C_train once for documentation
    c_train = compute_c_train(_routes("train"))
    (OUT_ROOT.parent / "C_TRAIN.json").write_text(
        json.dumps(
            {
                "c_train": c_train,
                "rule": "median_depot_horizon_over_TRAIN_routes_only",
                "dataset": "synthcharge_final/train",
                "n_train_routes": len(_routes("train")),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    seeds = list(SEEDS) if args.seed is None else [args.seed]
    for variant in variants:
        for seed in seeds:
            run_one(variant, seed, budget_updates=args.budget_updates)


if __name__ == "__main__":
    main()
