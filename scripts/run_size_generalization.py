"""Train Hybrid PPO on Small_Network train routes; evaluate Medium/Large as extra size-gen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import CHECKPOINTS_DIR, REPO_ROOT  # noqa: E402
from experiments.batch import dump_run, evaluate_population  # noqa: E402
from experiments.dataset import load_split_routes  # noqa: E402
from experiments.methods import build_from_checkpoint, build_stateless  # noqa: E402
from experiments.provenance import detect_device, frozen_hashes, git_sha  # noqa: E402
from rl.ablation import AblationConfig  # noqa: E402
from rl.ppo import PPOConfig  # noqa: E402
from rl.train_loop import train_hybrid_ppo  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train = load_split_routes("train", network_group="Small_Network")
    val = load_split_routes("validation")
    cfg = "hybrid_ppo_smoke.toml" if args.smoke else "hybrid_ppo.toml"
    config = PPOConfig.from_toml(REPO_ROOT / "configs" / "rl" / cfg)
    config.seed = args.seed
    out = CHECKPOINTS_DIR / "HybridPPO_sizegen" / f"seed_{args.seed}"
    train_hybrid_ppo(
        train_routes=train,
        val_routes=val,
        config=config,
        ablation=AblationConfig(),
        method="HybridPPO_sizegen",
        device=detect_device(),
        out_dir=out,
    )
    ckpt = out / "best.pt"
    if not ckpt.is_file():
        ckpt = out / "last.pt"
    actor = build_from_checkpoint("hybrid_ppo", ckpt)
    greedy = build_stateless("GreedyMinimumSufficientCharge")
    records = []
    hashes = frozen_hashes()
    ctx = {
        "git_sha": git_sha(),
        "note": "size_generalization_extra_not_headline",
        "corpus_sha256": hashes["corpus.jsonl"],
        "device": detect_device(),
    }
    for split in ("validation", "test"):
        for group in ("Medium_Network", "Large_Network"):
            routes = load_split_routes(split, network_group=group)
            records.extend(
                evaluate_population(routes=routes, method="HybridPPO_sizegen", policy=actor, split=split, seed=args.seed, extra_context=ctx)
            )
            records.extend(
                evaluate_population(routes=routes, method="GreedyMinimumSufficientCharge", policy=greedy, split=split, seed=0, extra_context=ctx)
            )
    path = dump_run(records, "size_generalization")
    print(f"wrote {len(records)} extra size-gen records to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
