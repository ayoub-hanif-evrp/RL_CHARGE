"""Train ablation variants A1–A5 plus FULL on TRAIN; val only for that variant's early stopping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import CHECKPOINTS_DIR, REPO_ROOT  # noqa: E402
from experiments.dataset import load_split_routes  # noqa: E402
from experiments.provenance import detect_device  # noqa: E402
from experiments.seeds import load_seed_list  # noqa: E402
from rl.ablation import AblationConfig  # noqa: E402
from rl.ppo import PPOConfig  # noqa: E402
from rl.train_loop import train_hybrid_ppo  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="ablation")
    parser.add_argument("--variants", default="FULL,A1,A2,A3,A4,A5")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-routes", type=int, default=None)
    parser.add_argument("--max-val-routes", type=int, default=None)
    args = parser.parse_args(argv)
    seeds = load_seed_list(args.seeds)
    train_routes = load_split_routes("train")
    val_routes = load_split_routes("validation")
    if args.max_train_routes is not None:
        train_routes = train_routes[: args.max_train_routes]
    if args.max_val_routes is not None:
        val_routes = val_routes[: args.max_val_routes]
    cfg_name = "hybrid_ppo_smoke.toml" if args.smoke else "hybrid_ppo.toml"
    cfg_path = REPO_ROOT / "configs" / "rl" / cfg_name
    device = detect_device()
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        ablation = AblationConfig.from_name(variant)
        for seed in seeds:
            config = PPOConfig.from_toml(cfg_path)
            config.seed = int(seed)
            out = CHECKPOINTS_DIR / f"ablation_{variant}" / f"seed_{seed}"
            manifest = train_hybrid_ppo(
                train_routes=train_routes,
                val_routes=val_routes,
                config=config,
                ablation=ablation,
                method=f"HybridPPO_{variant}",
                device=device,
                out_dir=out,
            )
            print(variant, seed, manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
