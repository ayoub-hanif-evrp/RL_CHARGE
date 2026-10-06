"""Freeze SynthCharge HybridPPO checkpoints for the V3 paper TEST (no DiscretePPO)."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from common import (  # noqa: E402
    METHOD_FREEZE_SHA,
    SEEDS,
    SYNTH_CKPT,
    V3,
    dump_json,
    git_head,
    lf_sha256,
    load_json,
    method_tree_diff,
    sha256,
)


def sha256_json(obj) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode("utf-8")).hexdigest()


def _entry(seed: int) -> dict:
    folder = SYNTH_CKPT / f"seed_{seed}"
    best = folder / "best.pt"
    manifest_path = folder / "manifest.json"
    curves = folder / "curves.jsonl"
    if not best.is_file() or not manifest_path.is_file():
        raise SystemExit(f"missing checkpoint: {folder}")
    manifest = load_json(manifest_path)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    if bool(payload["ablation"].get("time_aware")) is not True:
        raise SystemExit(f"time_aware missing: {folder}")
    if bool(payload["ablation"].get("discrete_u")):
        raise SystemExit(f"unexpected discrete_u: {folder}")
    return {
        "dataset": "synthcharge",
        "method": "HybridPPO",
        "seed": seed,
        "checkpoint": folder.relative_to(ROOT).as_posix() + "/best.pt",
        "checkpoint_sha256": sha256(best),
        "manifest_sha256": lf_sha256(manifest_path),
        "curves_sha256": lf_sha256(curves) if curves.is_file() else None,
        "normalizer_sha256": sha256_json(payload.get("normalizer")),
        "return_scale": float(payload["config"].get("return_scale", 1.0)),
        "best_update": manifest.get("best_update"),
        "status": manifest.get("status"),
        "best_val_parent_balanced_feasibility": manifest.get("best_val_feasibility"),
        "best_val_parent_balanced_completion_all": manifest.get("best_val_completion_all"),
        "training_git_sha": manifest.get("git_sha"),
        "run_kind": manifest.get("run_kind"),
        "time_aware_envelope": True,
        "discrete_u": False,
        "config": "configs/common/rl/hybrid_ppo.toml",
        "config_sha256": sha256(ROOT / "configs/common/rl/hybrid_ppo.toml"),
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "selection": "parent-balanced VAL feasibility then completion; reused from V2 freeze without reselection",
        "source_v2_freeze": "results/v2/final/CHECKPOINT_FREEZE.json",
    }


def build() -> dict:
    if method_tree_diff().strip():
        raise SystemExit("method tree differs from freeze")
    entries = [_entry(s) for s in SEEDS]
    if len({e["checkpoint_sha256"] for e in entries}) != 5:
        raise SystemExit("duplicate or missing HybridPPO checkpoint hashes")
    # Cross-check against V2 freeze for the same five SynthCharge HybridPPO seeds.
    v2 = load_json(ROOT / "results" / "v2" / "final" / "CHECKPOINT_FREEZE.json")
    v2_map = {
        (item["dataset"], item["method"], item["seed"]): item["checkpoint_sha256"]
        for item in v2["checkpoints"]
        if item["dataset"] == "synthcharge" and item["method"] == "HybridPPO"
    }
    for item in entries:
        key = (item["dataset"], item["method"], item["seed"])
        if v2_map.get(key) != item["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint hash diverged from V2 freeze: {key}")
    return {
        "experiment": "v3_hppo",
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "freeze_written_at_git_head": git_head(),
        "evaluated_checkpoint": "best.pt",
        "methods_included": ["HybridPPO"],
        "methods_excluded": ["DiscretePPO"],
        "n_checkpoints": 5,
        "no_more_training_after_this_file": True,
        "test_must_not_influence_selection": True,
        "reused_from_v2_without_reselection": True,
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda": False,
        },
        "checkpoints": entries,
    }


def verify() -> None:
    freeze = load_json(V3 / "CHECKPOINT_FREEZE.json")
    if freeze.get("n_checkpoints") != 5:
        raise SystemExit("expected 5 HybridPPO checkpoints")
    for item in freeze["checkpoints"]:
        if item["method"] != "HybridPPO":
            raise SystemExit("non-HybridPPO checkpoint in V3 freeze")
        path = ROOT / item["checkpoint"]
        if sha256(path) != item["checkpoint_sha256"]:
            raise SystemExit(f"hash mismatch: {item['checkpoint']}")
    print("V3 CHECKPOINT_FREEZE verified: 5 HybridPPO checkpoints")


def main() -> None:
    if "--verify" in sys.argv:
        verify()
        return
    if (V3 / "EVALUATION_CONSUMED.json").is_file():
        raise SystemExit("evaluation already consumed; refuse to rewrite freeze")
    payload = build()
    dump_json(V3 / "CHECKPOINT_FREEZE.json", payload)
    dump_json(V3 / "ENVIRONMENT.json", payload["environment"])
    verify()


if __name__ == "__main__":
    main()
