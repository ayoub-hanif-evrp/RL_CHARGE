"""Run provenance: git SHA, corpus/split hashes, device."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from data.paths import REPO_ROOT, ROUTES_DIR, SPLITS_DIR

# Generated experiment outputs must not mark the *code* tree dirty.
APPROVED_OUTPUT_PREFIXES = (
    "models/",
    "results/",
    # Legacy prefixes (pre-layout migration) kept for safety / frozen references
    "results_v4/",
    "results_v4_paper/",
    "results_paper/",
    "checkpoints/",
    "checkpoints_v2/",
    "checkpoints_v3/",
    "checkpoints_v4/",
    "checkpoints_development/",
)

_NOISE_PREFIXES = (
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
)


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


def _porcelain_paths(repo: Path) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return ["<git-status-unavailable>"]
    paths: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def _is_approved_output(path: str) -> bool:
    p = path.replace("\\", "/")
    if any(p.startswith(prefix) or f"/{prefix}" in f"/{p}" for prefix in APPROVED_OUTPUT_PREFIXES):
        return True
    if any(part.endswith("__pycache__") or part in {".pytest_cache", ".mypy_cache", ".ruff_cache"} for part in p.split("/")):
        return True
    if any(p.startswith(prefix) for prefix in _NOISE_PREFIXES):
        return True
    return False


def code_dirty_paths(repo: Path | None = None) -> list[str]:
    """Paths that count as source-code dirtiness (excludes approved outputs)."""
    root = Path(repo) if repo is not None else REPO_ROOT
    return [p for p in _porcelain_paths(root) if not _is_approved_output(p)]


def code_git_dirty(repo: Path | None = None) -> bool:
    return bool(code_dirty_paths(repo))


def git_dirty(repo: Path | None = None) -> bool:
    """Backward-compatible alias for code_git_dirty (approved outputs ignored)."""
    return code_git_dirty(repo)


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
    sha = git_sha()
    dirty = code_git_dirty()
    payload = {
        "git_sha": sha,
        "training_git_sha": sha,
        "code_git_dirty": dirty,
        # Legacy field: now means code dirtiness (approved outputs ignored).
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
