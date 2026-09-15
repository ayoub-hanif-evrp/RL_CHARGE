"""Nonlinear charging sensitivity on Small EVRPTW-GR test routes. Not original EVRPTW-GR."""

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
from experiments.dataset import load_split_routes, parse_route_instance  # noqa: E402
from experiments.methods import build_from_checkpoint, build_stateless  # noqa: E402
from experiments.provenance import detect_device, frozen_hashes, git_sha  # noqa: E402
from physics.montoya import montoya_piecewise_model  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cs-type", default="fast")
    parser.add_argument("--max-routes", type=int, default=None)
    parser.add_argument("--run-id", default="nonlinear_sensitivity")
    args = parser.parse_args(argv)
    routes = load_split_routes(args.split, network_group="Small_Network")
    if args.max_routes is not None:
        routes = routes[: args.max_routes]
    hashes = frozen_hashes()
    greedy = build_stateless("GreedyMinimumSufficientCharge")
    ckpt = CHECKPOINTS_DIR / "HybridPPO" / "seed_42" / "best.pt"
    if not ckpt.is_file():
        ckpt = CHECKPOINTS_DIR / "HybridPPO" / "seed_42" / "last.pt"
    hybrid = build_from_checkpoint("hybrid_ppo", ckpt) if ckpt.is_file() else None
    records = []
    for route in routes:
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance)
        full_linear = profile.inverse_refueling_rate * profile.battery_capacity
        model = montoya_piecewise_model(args.cs_type, scale_full_charge_time=full_linear)
        ctx = {
            "git_sha": git_sha(),
            "device": detect_device(),
            "corpus_sha256": hashes["corpus.jsonl"],
            "equivalent_to_evrptwgr": "not_original_evrptwgr",
            "charging_model": model.name,
            "cs_type": args.cs_type,
        }
        from experiments.evaluate import evaluate_policy

        result = evaluate_policy(instance=instance, route=route, policy=greedy, charging_model=model)
        records.append(result.to_record(method="GreedyMinimumSufficientCharge", split=args.split, seed=0, **ctx))
        if hybrid is not None:
            result_h = evaluate_policy(instance=instance, route=route, policy=hybrid, charging_model=model)
            records.append(result_h.to_record(method="HybridPPO", split=args.split, seed=42, **ctx))
    path = dump_run(records, args.run_id)
    print(f"wrote {len(records)} records to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
