"""Shared constants for the frozen V2 finalization scripts. No algorithm code."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "results" / "v2" / "final"
METHOD_FREEZE_SHA = "df35705c012b4ce88674a5b8906176c97838b06c"
TRAINING_EXECUTION_SHA = "2f4825adc341b71cec43e208540ee383de29cfb4"
DATA_FREEZE_SHA = "1ed15b4601a3e000c11736c7ce83a8841cea504d"
SEEDS = (42, 43, 44, 45, 46)
DATASETS = ("gold", "synthcharge")
METHODS = ("HybridPPO", "DiscretePPO")
BASELINES = ("GreedyMinimumSufficientCharge", "GreedyFullCharge", "OneStepLookahead")
FORBIDDEN_PARENTS = frozenset({"c101", "c205", "r110", "r201", "rc102", "rc208"})
FROZEN_DIRS = ("src", "configs", "tests", "third_party")


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def method_tree_diff() -> str:
    return subprocess.check_output(
        ["git", "diff", "--name-only", METHOD_FREEZE_SHA, "HEAD", "--", *FROZEN_DIRS],
        cwd=ROOT,
        text=True,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint_dir(dataset: str, method: str, seed: int) -> Path:
    return ROOT / "models" / "v2" / "final" / dataset / method / f"seed_{seed}"


def dataset_files(dataset: str) -> dict:
    if dataset == "gold":
        return {
            "training_corpus": "data/routes_v2/gold_official/corpus.jsonl",
            "train_split": "data/splits_v2/gold_train.json",
            "validation_split": "data/splits_v2/gold_validation.json",
            "n_train": 119,
            "n_validation": 47,
        }
    return {
        "training_corpus": "data/routes_v2/synthcharge_final/train/corpus.jsonl",
        "validation_corpus": "data/routes_v2/synthcharge_final/validation/corpus.jsonl",
        "train_split": "data/splits_v2/synthcharge_train.json",
        "validation_split": "data/splits_v2/synthcharge_validation.json",
        "n_train": 180,
        "n_validation": 90,
    }


def config_path(method: str) -> str:
    if method == "HybridPPO":
        return "configs/common/rl/hybrid_ppo.toml"
    if method == "DiscretePPO":
        return "configs/common/rl/discrete_ppo.toml"
    raise ValueError(method)
