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


V2_CORPUS_FILES = (
    "data/routes_v2/gold_official/corpus.jsonl",
    "data/routes_v2/gold_official/manifest.csv",
    "data/routes_v2/gold_official/corpus_metadata.json",
    "data/routes_v2/gold_official/certificates.jsonl",
    "data/routes_v2/certified_pyvrp/corpus.jsonl",
    "data/routes_v2/certified_pyvrp/manifest.csv",
    "data/routes_v2/certified_pyvrp/corpus_metadata.json",
    "data/routes_v2/certified_pyvrp/certificates.jsonl",
    "data/routes_v2/certified_pyvrp/repair_report.json",
    "data/routes_v2/certified_pyvrp/quarantine.jsonl",
)

V2_SPLIT_FILES = (
    "data/splits_v2/gold_train.json",
    "data/splits_v2/gold_validation.json",
    "data/splits_v2/pyvrp_train.json",
    "data/splits_v2/pyvrp_validation.json",
)


def hash_existing(relative_paths: tuple[str, ...]) -> dict[str, str]:
    hashed = {}
    for relative in relative_paths:
        path = REPO_ROOT / relative
        if path.is_file():
            hashed[relative.replace("\\", "/")] = sha256_file(path)
    return hashed


def v2_corpus_hashes() -> dict[str, str]:
    return hash_existing(V2_CORPUS_FILES)


def v2_split_hashes() -> dict[str, str]:
    return hash_existing(V2_SPLIT_FILES)


def run_manifest(**extra: Any) -> dict[str, Any]:
    dirty = git_dirty()
    payload = {
        "git_sha": git_sha(),
        "git_dirty": dirty,
        "run_kind": "dirty_exploratory" if dirty else "clean_sha",
        "final_v2_requires_clean_frozen_sha": True,
        "hashes": frozen_hashes(),
        "v2_corpus_hashes": v2_corpus_hashes(),
        "v2_split_hashes": v2_split_hashes(),
        **device_info(),
    }
    payload.update(extra)
    return payload
