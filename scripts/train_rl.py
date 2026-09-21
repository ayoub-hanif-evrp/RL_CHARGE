"""Train Hybrid PPO, Discrete PPO, or AttentionPPO on TRAIN routes only.

Legacy DDQN remains available as an internal CLI (`--method legacy_ddqn`)
but is not part of the paper experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.ddqn_train import train_ddqn  # noqa: E402
from data.paths import CHECKPOINTS_DIR, REPO_ROOT  # noqa: E402
from experiments.dataset import load_split_routes  # noqa: E402
from experiments.provenance import detect_device  # noqa: E402
from experiments.seeds import load_seed_list  # noqa: E402
from rl.ablation import AblationConfig  # noqa: E402
from rl.ppo import PPOConfig  # noqa: E402
from rl.train_loop import train_hybrid_ppo  # noqa: E402


METHOD_CONFIG = {
    "hybrid_ppo": ("FULL", "hybrid_ppo.toml", "hybrid_ppo_smoke.toml", "HybridPPO"),
    "discrete_ppo": ("A1", "discrete_ppo.toml", "discrete_ppo_smoke.toml", "DiscretePPO"),
    "attention_ppo": ("ATTENTION", "attention_ppo.toml", "attention_ppo_smoke.toml", "AttentionPPO"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["hybrid_ppo", "discrete_ppo", "attention_ppo", "legacy_ddqn"], default="hybrid_ppo")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seeds", default="paper")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-train-routes", type=int, default=None)
    parser.add_argument("--max-val-routes", type=int, default=None)
    parser.add_argument("--network-group", default=None)
    parser.add_argument("--wall-clock-s", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if str(args.split).lower() in {"test", "testing"}:
        raise SystemExit("TEST is forbidden for training, checkpoint selection, and config decisions.")

    train_routes = load_split_routes(args.split, network_group=args.network_group)
    val_routes = load_split_routes("validation", network_group=args.network_group)
    if args.max_train_routes is not None:
        train_routes = train_routes[: args.max_train_routes]
    if not train_routes:
        raise SystemExit("no train routes")
    device = detect_device()
    seeds = load_seed_list(args.seeds)
    print(f"method={args.method} seeds={seeds} device={device} n_train={len(train_routes)} n_val={len(val_routes)}")

    if args.method == "legacy_ddqn":
        name = "legacy_ddqn_smoke.toml" if args.smoke else "legacy_ddqn.toml"
        cfg_path = args.config or (REPO_ROOT / "configs" / "rl" / name)
        import tomllib

        with Path(cfg_path).open("rb") as handle:
            raw = tomllib.load(handle)
        for seed in seeds:
            out = args.out_dir or (CHECKPOINTS_DIR / "LegacyTwoStageDDQN" / f"seed_{seed}")
            manifest = train_ddqn(
                train_routes=train_routes,
                val_routes=val_routes,
                config_seed=seed,
                gradient_steps=int(raw["gradient_steps"]),
                batch_size=int(raw["batch_size"]),
                target_sync=int(raw["target_sync"]),
                replay_size=int(raw["replay_size"]),
                device=device,
                d_model=int(raw["d_model"]),
                wall_clock_s=args.wall_clock_s,
                out_dir=out,
                val_interval=int(raw.get("val_interval", 50)),
                val_max_routes=args.max_val_routes,
                early_stopping_patience=int(raw.get("early_stopping_patience", 20)),
                learning_rate=float(raw.get("learning_rate", 1e-3)),
            )
            print(manifest["status"], out)
        return 0

    ablation_name, paper_name, smoke_name, method_name = METHOD_CONFIG[args.method]
    cfg_name = smoke_name if args.smoke else paper_name
    cfg_path = args.config or (REPO_ROOT / "configs" / "rl" / cfg_name)
    ablation = AblationConfig.from_name(ablation_name)
    for seed in seeds:
        config = PPOConfig.from_toml(cfg_path)
        config.seed = int(seed)
        out = args.out_dir or (CHECKPOINTS_DIR / method_name / f"seed_{seed}")
        manifest = train_hybrid_ppo(
            train_routes=train_routes,
            val_routes=val_routes,
            config=config,
            ablation=ablation,
            method=method_name,
            device=device,
            out_dir=out,
            wall_clock_s=args.wall_clock_s,
            val_max_routes=(
                args.max_val_routes
                if args.max_val_routes is not None
                else (16 if args.smoke else None)
            ),
        )
        print(manifest["status"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
