"""Run provenance: git SHA, corpus/split hashes, device."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from data.paths import REPO_ROOT, ROUTES_DIR, SPLITS_DIR


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha(repo: Path | None = None) -> str:
    root = Path(repo) if repo is not None else REPO_ROOT
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def git_dirty(repo: Path | None = None) -> bool:
    root = Path(repo) if repo is not None else REPO_ROOT
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return True


def device_info() -> dict[str, Any]:
    info = {"device": "cpu", "cuda_name": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def detect_device() -> str:
    return str(device_info()["device"])


def frozen_hashes() -> dict[str, str]:
    return {
        "corpus.jsonl": sha256_file(ROUTES_DIR / "corpus.jsonl"),
        "manifest.csv": sha256_file(ROUTES_DIR / "manifest.csv"),
        "corpus_metadata.json": sha256_file(ROUTES_DIR / "corpus_metadata.json"),
        "train.json": sha256_file(SPLITS_DIR / "train.json"),
        "validation.json": sha256_file(SPLITS_DIR / "validation.json"),
        "test.json": sha256_file(SPLITS_DIR / "test.json"),
        "split_metadata.json": sha256_file(SPLITS_DIR / "split_metadata.json"),
    }


def run_manifest(**extra: Any) -> dict[str, Any]:
    payload = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "hashes": frozen_hashes(),
        **device_info(),
    }
    payload.update(extra)
    return payload
