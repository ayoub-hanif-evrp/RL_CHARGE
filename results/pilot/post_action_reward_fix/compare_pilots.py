"""Compare archived pre-fix vs post-fix Hybrid PPO TRAIN/VAL pilots. No TEST."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routing.serialize import canonical_dumps

PRE = ROOT / "results" / "pilot" / "pre_action_fix"
POST = ROOT / "results" / "pilot" / "post_action_reward_fix"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _curves(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _loss_stats(rows: list[dict]) -> dict:
    losses = [float(r["loss"]) for r in rows if "loss" in r]
    finite = [x for x in losses if math.isfinite(x)]
    return {
        "n": len(losses),
        "finite": len(finite) == len(losses) and bool(finite),
        "nan_or_inf": any(not math.isfinite(x) for x in losses),
        "min": min(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "max": max(finite) if finite else None,
        "first10_mean": statistics.mean(finite[:10]) if finite else None,
        "last10_mean": statistics.mean(finite[-10:]) if finite else None,
    }


def _trajectory(rows: list[dict], key: str) -> list[list]:
    out = []
    for row in rows:
        if key in row:
            out.append([row["update"], row[key]])
    return out


def _best_update(rows: list[dict]) -> int | None:
    last = None
    for row in rows:
        if row.get("checkpoint_sha256"):
            last = int(row["update"])
    return last


def summarize_seed(seed: int, folder: Path, kind: str) -> dict:
    curves = _curves(folder / f"seed_{seed}" / "curves.jsonl")
    manifest = _load_json(folder / f"seed_{seed}" / "manifest.json")
    behavior_best = folder / f"seed_{seed}" / "policy_behavior.json"
    behavior_last = folder / f"seed_{seed}" / "policy_behavior_last.json"
    payload = {
        "seed": seed,
        "kind": kind,
        "status": manifest.get("status"),
        "runtime_s": manifest.get("runtime_s"),
        "device": manifest.get("device"),
        "best_update": manifest.get("best_update", _best_update(curves)),
        "best_val_feasibility": manifest.get("best_val_feasibility"),
        "best_val_completion_all": manifest.get("best_val_completion_all"),
        "best_val_route_weighted_feasibility": manifest.get(
            "best_val_route_weighted_feasibility"
        ),
        "checkpoint_selection": manifest.get("checkpoint_selection"),
        "updates_logged": curves[-1]["update"] if curves else None,
        "val_route_weighted_feasibility_trajectory": _trajectory(curves, "val_feasibility"),
        "val_parent_balanced_feasibility_trajectory": _trajectory(
            curves, "val_parent_balanced_feasibility"
        ),
        "val_parent_balanced_completion_trajectory": _trajectory(
            curves, "val_parent_balanced_completion_all"
        ),
        "val_route_weighted_completion_trajectory": _trajectory(curves, "val_completion_all"),
        "loss": _loss_stats(curves),
    }
    if behavior_best.is_file():
        payload["best_policy"] = _load_json(behavior_best)
    if behavior_last.is_file():
        payload["last_policy"] = _load_json(behavior_last)
    return payload


def _stable_region(traj: list[list], min_points: int = 5, floor: float = 1e-12) -> bool:
    """True if several consecutive VAL points stay above floor, not a short spike."""
    vals = [v for _, v in traj if v is not None]
    if len(vals) < min_points:
        return False
    run = 0
    best_run = 0
    for v in vals:
        if v > floor:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    return best_run >= min_points


def main() -> int:
    pre_summary = _load_json(PRE / "pilot_summary.json")
    seeds = {}
    for seed in (42, 43):
        seeds[str(seed)] = {
            "pre": {
                "from_archive_summary": pre_summary["seeds"][str(seed)],
                "curves": summarize_seed(seed, PRE, "pre_action_fix"),
            },
            "post": summarize_seed(seed, POST, "post_action_reward_fix"),
        }
    post_ok = []
    for seed, block in seeds.items():
        post = block["post"]
        pb = post["val_parent_balanced_feasibility_trajectory"]
        rw = post["val_route_weighted_feasibility_trajectory"]
        post_ok.append(
            _stable_region(pb) or _stable_region(rw)
        )
        block["post_has_stable_val_region"] = _stable_region(pb) or _stable_region(rw)
        block["post_parent_balanced_absent"] = not any(
            v > 1e-12 for _, v in pb
        ) if pb else True
    both_stable = all(post_ok)
    decision = (
        "PROCEED_FREEZE_BUDGET_FROM_TRAIN_VAL"
        if both_stable
        else "STOP_DO_NOT_FREEZE_PAPER_BUDGET"
    )
    payload = {
        "scenario": "pilot_post_action_reward_fix",
        "split_test_used": False,
        "pre_archive": str(PRE),
        "post_archive": str(POST),
        "decision": decision,
        "both_seeds_stable_train_val_region": both_stable,
        "seeds": seeds,
    }
    (POST / "pilot_comparison.json").write_text(
        canonical_dumps(payload) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "both_stable": both_stable}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
