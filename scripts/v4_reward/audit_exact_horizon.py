"""Audit whether any successful TRAIN/VAL route finishes at T=H. No TEST."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from domain.load_convention import LoadConvention  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from experiments.evaluate import evaluate_policy  # noqa: E402
from rl.ablation import AblationConfig  # noqa: E402
from rl.checkpoint import load_hybrid_actor  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402

OUT = ROOT / "results" / "v4_reward" / "analysis"
CKPT = ROOT / "checkpoints_v4" / "reward_ablation" / "V4_BASE_NO_L_FAIL" / "seed_42" / "best.pt"


def _audit_split(split: str, actor) -> dict:
    corpus = ROOT / "data" / "routes_v2" / "synthcharge_final" / split / "corpus.jsonl"
    routes = read_jsonl(corpus)
    diffs = []
    n_success = 0
    for route in routes:
        instance = parse_route_instance(route)
        result = evaluate_policy(
            instance=instance,
            route=route,
            policy=actor,
            load_convention=LoadConvention.OFFICIAL_REFERENCE_PICKUP,
            profile_name=route.physics_profile or "synthcharge_linear",
            eval_mode=True,
        )
        if result.feasible and result.completed:
            n_success += 1
            diffs.append(abs(float(result.route_completion_time) - float(result.horizon)))
    return {
        "split": split,
        "n_routes": len(routes),
        "n_success": n_success,
        "n_abs_diff_lt_1e-9": int(sum(d < 1e-9 for d in diffs)),
        "n_abs_diff_lt_1e-6": int(sum(d < 1e-6 for d in diffs)),
        "min_abs_diff": float(min(diffs)) if diffs else None,
        "median_abs_diff": float(np.median(diffs)) if diffs else None,
    }


def main() -> None:
    if not CKPT.is_file():
        raise SystemExit(f"missing {CKPT}")
    actor = load_hybrid_actor(CKPT)
    actor.ablation = AblationConfig(name="FULL", time_aware=True, soc_interval="continuation_to_max")
    train = _audit_split("train", actor)
    val = _audit_split("validation", actor)
    payload = {
        "checkpoint": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
        "train": train,
        "validation": val,
        "conclusion": (
            "No successful TRAIN or VAL route under this checkpoint finishes with |T-H|<1e-6. "
            "T=H remains a theoretical boundary for V4_BASE_NO_L_FAIL where success and failure "
            "returns coincide. No epsilon penalty was added."
            if (train["n_abs_diff_lt_1e-6"] == 0 and val["n_abs_diff_lt_1e-6"] == 0)
            else "Exact/near-horizon successes were observed; see counts."
        ),
        "confirmatory_test_consumed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "EXACT_HORIZON_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
