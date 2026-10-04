"""POST-HOC DEVELOPMENT envelope / lower-bound ablation on SynthCharge TRAIN/VAL.

Variants (existing AblationConfig knobs; no frozen src changes):

  A_arrival_to_max     soc_interval=arrival_to_max,       time_aware=False
  B_energy_to_max      soc_interval=continuation_to_max,  time_aware=False
  C_energy_to_time     soc_interval=continuation_to_max,  time_aware=True

All use TRAIN-only return scaling (same optimization budget as FA-HPPO B2).

LABEL: POST-HOC DEVELOPMENT / TRAIN-VAL ONLY — NOT V3 CONFIRMATORY TEST EVIDENCE

Does not read V3 TEST routes. Does not modify paired_primary.json.
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
OUT_ROOT = ROOT / "results" / "development" / "envelope_ablation"
CKPT_ROOT = ROOT / "checkpoints_development" / "envelope_ablation"

VARIANTS = {
    "A_arrival_to_max": {
        "soc_interval": "arrival_to_max",
        "time_aware": False,
        "label": "Arrival-to-Max",
    },
    "B_energy_to_max": {
        "soc_interval": "continuation_to_max",
        "time_aware": False,
        "label": "EnergyLower-to-Max",
    },
    "C_energy_to_time": {
        "soc_interval": "continuation_to_max",
        "time_aware": True,
        "label": "EnergyLower-to-TimeUpper",
    },
}


def _synth_routes(split: str):
    assert_learning_split(split)
    corpus = ROOT / "data" / "routes_v2" / "synthcharge_final" / split / "corpus.jsonl"
    routes = read_jsonl(corpus)
    test_ids = set(
        json.loads((ROOT / "data" / "splits_v2" / "synthcharge_test.json").read_text(encoding="utf-8"))["route_ids"]
    )
    if any(route.route_id in test_ids for route in routes):
        raise SystemExit("SynthCharge TEST route entered learning data")
    # Also refuse V3 TEST ids if present in a separate lock list
    v3_test = ROOT / "data" / "splits_v2" / "synthcharge_v3_test.json"
    if v3_test.is_file():
        v3_ids = set(json.loads(v3_test.read_text(encoding="utf-8")).get("route_ids", []))
        if any(route.route_id in v3_ids for route in routes):
            raise SystemExit("V3 TEST route entered learning data")
    return routes


def run_one(variant: str, seed: int) -> None:
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant}")
    if seed not in SEEDS:
        raise SystemExit("seeds must be 42..46")
    spec = VARIANTS[variant]
    out_ckpt = CKPT_ROOT / variant / f"seed_{seed}"
    dest = OUT_ROOT / variant / f"seed_{seed}"
    if (dest / "validation.json").is_file() and (out_ckpt / "best.pt").is_file():
        print(f"skip completed {variant} seed={seed}", flush=True)
        return
    train_routes = _synth_routes("train")
    val_routes = _synth_routes("validation")
    config = PPOConfig.from_toml()
    config.seed = int(seed)
    ablation = AblationConfig(
        name=f"ENV_{variant}",
        soc_interval=spec["soc_interval"],
        time_aware=bool(spec["time_aware"]),
    )
    scale = global_return_scale(train_routes)
    manifest = train_hybrid_ppo(
        train_routes=train_routes,
        val_routes=val_routes,
        config=config,
        ablation=ablation,
        method=f"DEV_ENV_{variant}",
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
        "label": "POST-HOC DEVELOPMENT / TRAIN-VAL ONLY — NOT V3 CONFIRMATORY TEST EVIDENCE",
        "variant": variant,
        "display_name": spec["label"],
        "seed": seed,
        "soc_interval": spec["soc_interval"],
        "time_aware": spec["time_aware"],
        "return_scale": scale,
        "dataset": "synthcharge_final",
        "n_train": len(train_routes),
        "n_val": len(val_routes),
        "best_update": manifest.get("best_update"),
        "status": manifest.get("status"),
        "parent_balanced_val_feasibility": val["parent_balanced_feasibility"],
        "parent_balanced_completion_all": val["parent_balanced_completion_all"],
        "route_weighted_feasibility": val["feasibility"],
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
    (OUT_ROOT / "README.md").write_text(
        "# Envelope / lower-bound ablation (development)\n\n"
        "**POST-HOC DEVELOPMENT / TRAIN-VAL ONLY — NOT V3 CONFIRMATORY.**\n\n"
        "Launch: `python scripts/development/run_envelope_ablation.py --variant all`\n",
        encoding="utf-8",
    )
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    seeds = list(SEEDS) if args.seed is None else [args.seed]
    for variant in variants:
        for seed in seeds:
            run_one(variant, seed)


if __name__ == "__main__":
    main()
