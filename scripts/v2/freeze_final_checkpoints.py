"""Archive training metadata and write CHECKPOINT_FREEZE.json before TEST."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v2"))

from final_common import (  # noqa: E402
    DATA_FREEZE_SHA,
    DATASETS,
    FINAL,
    METHOD_FREEZE_SHA,
    METHODS,
    SEEDS,
    TRAINING_EXECUTION_SHA,
    checkpoint_dir,
    config_path,
    dataset_files,
    dump_json,
    git_head,
    git_porcelain,
    lf_sha256,
    load_json,
    method_tree_diff,
    sha256,
)


def _entry(dataset: str, method: str, seed: int) -> dict:
    folder = checkpoint_dir(dataset, method, seed)
    best = folder / "best.pt"
    last = folder / "last.pt"
    manifest_path = folder / "manifest.json"
    curves = folder / "curves.jsonl"
    if not best.is_file() or not manifest_path.is_file() or not curves.is_file():
        raise SystemExit(f"incomplete run: {folder.relative_to(ROOT).as_posix()}")
    manifest = load_json(manifest_path)
    if manifest.get("best_update") is None:
        raise SystemExit(f"best_update is null: {folder}")
    if manifest.get("git_dirty"):
        raise SystemExit(f"dirty training run: {folder}")
    if manifest.get("run_kind") != "clean_sha":
        raise SystemExit(f"run_kind is not clean_sha: {folder}")
    if manifest.get("split_used_for_learning") != "train":
        raise SystemExit(f"learning split not train: {folder}")
    if manifest.get("split_used_for_selection") != "validation":
        raise SystemExit(f"selection split not validation: {folder}")
    if (manifest.get("normalizer_provenance") or {}).get("split") != "train":
        raise SystemExit(f"normalizer not TRAIN-only: {folder}")
    payload = torch.load(best, map_location="cpu", weights_only=False)
    if bool(payload["ablation"].get("time_aware")) is not True:
        raise SystemExit(f"time_aware missing: {folder}")
    if bool(payload["ablation"].get("discrete_u")) != (method == "DiscretePPO"):
        raise SystemExit(f"discrete_u mismatch: {folder}")
    files = dataset_files(dataset)
    archive = FINAL / "training" / dataset / method / f"seed_{seed}"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "manifest.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    (archive / "curves.jsonl").write_text(curves.read_text(encoding="utf-8"), encoding="utf-8")
    dump_json(archive / "normalizer_provenance.json", manifest.get("normalizer_provenance") or {})
    hashes = {
        "best.pt": sha256(best),
        "last.pt": sha256(last) if last.is_file() else None,
        "manifest.json": lf_sha256(manifest_path),
        "curves.jsonl": lf_sha256(curves),
        "normalizer_state": sha256_json(payload.get("normalizer")),
    }
    dump_json(archive / "checkpoint_hashes.json", hashes)
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "checkpoint": folder.relative_to(ROOT).as_posix() + "/best.pt",
        "checkpoint_sha256": hashes["best.pt"],
        "last_checkpoint_sha256": hashes["last.pt"],
        "manifest_sha256": hashes["manifest.json"],
        "curves_sha256": hashes["curves.jsonl"],
        "normalizer_sha256": hashes["normalizer_state"],
        "normalizer_fit_split": "train",
        "return_scale": float(payload["config"].get("return_scale", 1.0)),
        "return_scale_fit_split": "train",
        "best_update": manifest.get("best_update"),
        "status": manifest.get("status"),
        "runtime_s": manifest.get("runtime_s"),
        "device": manifest.get("device"),
        "best_val_parent_balanced_feasibility": manifest.get("best_val_feasibility"),
        "best_val_parent_balanced_completion_all": manifest.get("best_val_completion_all"),
        "best_val_route_weighted_feasibility": manifest.get("best_val_route_weighted_feasibility"),
        "n_train_routes": manifest.get("n_train_routes"),
        "n_val_routes": manifest.get("n_val_routes"),
        "training_git_sha": manifest.get("git_sha"),
        "run_kind": manifest.get("run_kind"),
        "time_aware_envelope": True,
        "discrete_u": method == "DiscretePPO",
        "config": config_path(method),
        "config_sha256": sha256(ROOT / config_path(method)),
        "training_corpus_sha256_lf": lf_sha256(ROOT / files["training_corpus"]),
        "validation_split_sha256_lf": lf_sha256(ROOT / files["validation_split"]),
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "data_freeze_sha": DATA_FREEZE_SHA,
        "training_execution_sha": TRAINING_EXECUTION_SHA,
        "command": (
            f"python scripts/v2/run_final_b2.py --dataset {dataset} --method {method} --seed {seed}"
        ),
        "archived_metadata": archive.relative_to(ROOT).as_posix(),
    }


def sha256_json(obj) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode("utf-8")).hexdigest()


def main() -> None:
    if method_tree_diff().strip():
        raise SystemExit("method tree differs from freeze")
    entries = [_entry(d, m, s) for d in DATASETS for m in METHODS for s in SEEDS]
    if len(entries) != 20:
        raise SystemExit("expected 20 checkpoints")
    if len({e["checkpoint_sha256"] for e in entries}) != 20:
        raise SystemExit("duplicate best.pt hashes")
    payload = {
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "data_freeze_sha": DATA_FREEZE_SHA,
        "training_execution_sha": TRAINING_EXECUTION_SHA,
        "freeze_written_at_git_head": git_head(),
        "checkpoint_selection": (
            "parent-balanced validation feasibility, then parent-balanced "
            "validation all-routes completion; evaluate best.pt only"
        ),
        "evaluated_checkpoint": "best.pt",
        "no_more_training_after_this_file": True,
        "test_must_not_influence_selection": True,
        "n_checkpoints": 20,
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda": False,
        },
        "checkpoints": entries,
    }
    dump_json(FINAL / "CHECKPOINT_FREEZE.json", payload)
    print(f"CHECKPOINT_FREEZE.json written with {len(entries)} entries")


if __name__ == "__main__":
    main()
