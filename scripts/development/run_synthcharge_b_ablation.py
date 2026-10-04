"""POST-HOC same-domain SynthCharge TRAIN/VAL B0/B1/B3/B2 2×2.

Reproduces the gold time-cap × return-scaling factorial on SynthCharge splits.
Does NOT touch V3 TEST. Not predeclared confirmatory evidence.

Label everywhere:
  post-hoc same-domain development robustness analysis
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
from rl.train_loop import assert_learning_split, evaluate_routes, train_hybrid_ppo  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from routing.v2_corpus import global_return_scale  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
OUT_ROOT = ROOT / "results" / "development" / "synthcharge_b_ablation"
CKPT_ROOT = ROOT / "checkpoints_development" / "synthcharge_b_ablation"
VARIANTS = {
    "B0": {"time_aware": False, "scale": False, "label": "Base HPPO (B0)"},
    "B1": {"time_aware": True, "scale": False, "label": "+ Time-aware cap (B1)"},
    "B3": {"time_aware": False, "scale": True, "label": "+ Return scaling (B3)"},
    "B2": {"time_aware": True, "scale": True, "label": "FA-HPPO full (B2)"},
}


def _routes(split: str):
    assert_learning_split(split)
    corpus = ROOT / "data" / "routes_v2" / "synthcharge_final" / split / "corpus.jsonl"
    routes = read_jsonl(corpus)
    test_ids = set(
        json.loads((ROOT / "data" / "splits_v2" / "synthcharge_test.json").read_text(encoding="utf-8"))["route_ids"]
    )
    if any(route.route_id in test_ids for route in routes):
        raise SystemExit("SynthCharge TEST route entered learning data")
    return routes


def run_one(variant: str, seed: int) -> None:
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
    config = PPOConfig.from_toml()
    config.seed = int(seed)
    ablation = AblationConfig(name="FULL", time_aware=bool(spec["time_aware"]), soc_interval="continuation_to_max")
    scale = global_return_scale(train_routes) if spec["scale"] else None
    manifest = train_hybrid_ppo(
        train_routes=train_routes,
        val_routes=val_routes,
        config=config,
        ablation=ablation,
        method=f"DEV_SC_{variant}",
        out_dir=out_ckpt,
        return_scale=scale,
    )
    actor = load_hybrid_actor(out_ckpt / "best.pt" if (out_ckpt / "best.pt").is_file() else out_ckpt / "last.pt")
    actor.ablation = ablation
    val = evaluate_routes(val_routes, actor)
    dest.mkdir(parents=True, exist_ok=True)
    if (out_ckpt / "curves.jsonl").is_file():
        (dest / "curves.jsonl").write_text((out_ckpt / "curves.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "label": "POST-HOC same-domain development robustness analysis — NOT V3 CONFIRMATORY",
        "variant": variant,
        "display_name": spec["label"],
        "seed": seed,
        "time_aware": spec["time_aware"],
        "return_scale_enabled": bool(spec["scale"]),
        "return_scale": scale if scale is not None else 1.0,
        "dataset": "synthcharge_final",
        "parent_balanced_val_feasibility": val["parent_balanced_feasibility"],
        "parent_balanced_completion_all": val["parent_balanced_completion_all"],
        "route_weighted_feasibility": val["feasibility"],
        "best_update": manifest.get("best_update"),
        "status": manifest.get("status"),
        "not_v3_test": True,
        "post_hoc_development": True,
    }
    (dest / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variant": variant, "seed": seed, "val_feas": summary["parent_balanced_val_feasibility"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "PROTOCOL.md").write_text(
        "# SynthCharge B0/B1/B3/B2 same-domain ablation\n\n"
        "**POST-HOC same-domain development robustness analysis — NOT V3 CONFIRMATORY.**\n\n"
        "Compute estimate: ~20 cells × ~45–65 min ≈ 15–22 CPU-hours.\n\n"
        "Launch: `python scripts/development/run_synthcharge_b_ablation.py --variant all`\n",
        encoding="utf-8",
    )
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    seeds = list(SEEDS) if args.seed is None else [args.seed]
    for variant in variants:
        for seed in seeds:
            run_one(variant, seed)


if __name__ == "__main__":
    main()
