"""Freeze V4 authoritative + V3 reference FA-HPPO checkpoints for V4 TEST."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v4_test"))

from common import SEEDS, V3_CKPT, V4, V4_CKPT, dump_json, git_head, lf_sha256, load_json, sha256  # noqa: E402


def _entry(method: str, seed: int, folder: Path, training_git_sha: str | None = None) -> dict:
    best = folder / "best.pt"
    if not best.is_file():
        raise SystemExit(f"missing checkpoint: {best}")
    payload = torch.load(best, map_location="cpu", weights_only=False)
    if bool(payload["ablation"].get("time_aware")) is not True:
        raise SystemExit(f"time_aware missing: {folder}")
    manifest_path = folder / "manifest.json"
    val_path = None
    # V4 authoritative also has validation.json under results/
    if method == "V4_FA_HPPO":
        val_path = (
            ROOT
            / "results"
            / "v4_reward"
            / "final_authoritative"
            / "V4_BASE_NO_L_FAIL"
            / f"seed_{seed}"
            / "validation.json"
        )
    meta = load_json(val_path) if val_path and val_path.is_file() else {}
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    return {
        "method": method,
        "seed": seed,
        "checkpoint": folder.relative_to(ROOT).as_posix() + "/best.pt",
        "checkpoint_sha256": sha256(best),
        "manifest_sha256": lf_sha256(manifest_path) if manifest_path.is_file() else None,
        "training_git_sha": training_git_sha
        or meta.get("training_git_sha")
        or manifest.get("training_git_sha")
        or manifest.get("git_sha"),
        "code_git_dirty": meta.get("code_git_dirty", manifest.get("code_git_dirty")),
        "run_kind": meta.get("run_kind", manifest.get("run_kind")),
        "best_update": meta.get("best_update", manifest.get("best_update")),
        "parent_balanced_val_feasibility": meta.get(
            "parent_balanced_val_feasibility", manifest.get("best_val_feasibility")
        ),
        "parent_balanced_completion_all": meta.get(
            "parent_balanced_completion_all", manifest.get("best_val_completion_all")
        ),
        "reward_kind": meta.get("reward_kind", "V3_TIME" if method == "V3_FA_HPPO" else None),
        "time_aware_envelope": True,
    }


def build() -> dict:
    freeze_reward = ROOT / "results" / "v4_reward" / "FINAL_REWARD_FREEZE.json"
    if not freeze_reward.is_file():
        raise SystemExit("FINAL_REWARD_FREEZE.json missing; freeze reward first")
    reward = load_json(freeze_reward)
    if reward.get("reward_kind") != "V4_BASE_NO_L_FAIL":
        raise SystemExit("unexpected reward kind in FINAL_REWARD_FREEZE")
    if reward.get("code_git_dirty") or reward.get("confirmatory_test_consumed"):
        raise SystemExit("reward freeze not clean / already consumed TEST")

    v4_entries = [_entry("V4_FA_HPPO", s, V4_CKPT / f"seed_{s}") for s in SEEDS]
    v3_entries = [_entry("V3_FA_HPPO", s, V3_CKPT / f"seed_{s}") for s in SEEDS]
    # Cross-check V4 hashes against reward freeze
    for item in v4_entries:
        expected = reward["seeds"][str(item["seed"])]["checkpoint_sha256"]
        if item["checkpoint_sha256"] != expected:
            raise SystemExit(f"V4 checkpoint hash mismatch vs reward freeze seed={item['seed']}")
    return {
        "experiment": "v4_reward_test",
        "freeze_written_at_git_head": git_head(),
        "evaluated_checkpoint": "best.pt",
        "methods_included": ["V4_FA_HPPO", "V3_FA_HPPO"],
        "baselines": ["GreedyMinimumSufficientCharge", "GreedyFullCharge", "OneStepLookahead"],
        "n_v4_checkpoints": 5,
        "n_v3_checkpoints": 5,
        "no_more_training_after_this_file": True,
        "test_must_not_influence_selection": True,
        "reward_freeze": "results/v4_reward/FINAL_REWARD_FREEZE.json",
        "reward_training_git_sha": reward.get("training_git_sha") or reward.get("git_sha"),
        "checkpoints": v4_entries + v3_entries,
    }


def main() -> None:
    payload = build()
    dump_json(V4 / "CHECKPOINT_FREEZE.json", payload)
    print(json.dumps({"n": len(payload["checkpoints"]), "head": payload["freeze_written_at_git_head"]}, indent=2))


if __name__ == "__main__":
    import json

    main()
