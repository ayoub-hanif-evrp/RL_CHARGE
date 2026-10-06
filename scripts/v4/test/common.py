"""Shared helpers for V4 confirmatory TEST scripts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V4 = ROOT / "results" / "v4" / "test"
SEEDS = (42, 43, 44, 45, 46)
BASELINES = ("GreedyMinimumSufficientCharge", "GreedyFullCharge", "OneStepLookahead")
V4_CKPT = ROOT / "models" / "v4" / "fa_hppo"
V3_CKPT = ROOT / "models" / "v2" / "final" / "synthcharge" / "HybridPPO"
TEST_CORPUS = ROOT / "data" / "routes_v2" / "synthcharge_v4_test" / "corpus.jsonl"
TEST_SPLIT = ROOT / "data" / "splits_v2" / "synthcharge_v4_test.json"
GENERATOR_SEED_START = 500000


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
