"""Smoke-train Hybrid PPO. Not the paper run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.parser import parse_instance  # noqa: E402
from data.paths import RAW_EVRPTW_GR_DIR, REPO_ROOT  # noqa: E402
from domain.load_convention import LoadConvention  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from rl.env import ShieldedRouteEnv  # noqa: E402
from rl.ppo import HybridPPO, PPOConfig  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "rl" / "hybrid_ppo_smoke.toml")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--route", type=Path, required=True)
    args = parser.parse_args(argv)
    config = PPOConfig.from_toml(args.config)
    route = read_jsonl(args.route)[0]
    instance = parse_instance(RAW_EVRPTW_GR_DIR / route.relative_path)
    profile = PhysicsProfile.from_instance(instance)
    env = ShieldedRouteEnv(
        instance, route, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    trainer = HybridPPO(config, device="cpu")
    stats = trainer.smoke_train(env, updates=args.updates)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
