"""Shared helpers for V3-HPPO paper scripts. No algorithm changes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "results" / "v3_hppo"
METHOD_FREEZE_SHA = "df35705c012b4ce88674a5b8906176c97838b06c"
SEEDS = (42, 43, 44, 45, 46)
BASELINES = ("GreedyMinimumSufficientCharge", "GreedyFullCharge", "OneStepLookahead")
FROZEN_DIRS = ("src", "configs", "tests", "third_party")
# Scientific freeze check used by V3 gates: algorithm/physics configs + src + third_party.
# New V3 protocol tests under tests/test_v3_* are allowed without reopening the method freeze.
SCIENCE_PATHS = (
    "src",
    "third_party",
    "configs/rl",
    "configs/physics",
    "configs/routing",
    "configs/experiments",
    "configs/v2",
)
SYNTH_CKPT = ROOT / "checkpoints_v2" / "final" / "synthcharge" / "HybridPPO"
TEST_CORPUS = ROOT / "data" / "routes_v2" / "synthcharge_v3_test" / "corpus.jsonl"
TEST_SPLIT = ROOT / "data" / "splits_v2" / "synthcharge_v3_test.json"


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)


def method_tree_diff() -> str:
    return subprocess.check_output(
        ["git", "diff", "--name-only", METHOD_FREEZE_SHA, "HEAD", "--", *SCIENCE_PATHS],
        cwd=ROOT,
        text=True,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
