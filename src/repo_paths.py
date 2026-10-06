"""Canonical repository layout helpers and legacy path resolution.

Frozen JSON protocols may still mention historical paths. Prefer these
helpers in active code; do not rewrite frozen scientific files.
"""

from __future__ import annotations

from pathlib import Path

from data.paths import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs"

# Versioned result roots
RESULTS_V1 = RESULTS_DIR / "v1"
RESULTS_V2 = RESULTS_DIR / "v2"
RESULTS_V3 = RESULTS_DIR / "v3"
RESULTS_V4 = RESULTS_DIR / "v4"

RESULTS_V4_PAPER = RESULTS_V4 / "paper"
RESULTS_V4_TEST = RESULTS_V4 / "test"
RESULTS_V4_REWARD = RESULTS_V4 / "reward_development"
RESULTS_V4_TRAINING_FIGS = RESULTS_V4 / "training_figures"

MODELS_V1 = MODELS_DIR / "v1"
MODELS_V2 = MODELS_DIR / "v2"
MODELS_V3 = MODELS_DIR / "v3"
MODELS_V4 = MODELS_DIR / "v4"
MODELS_V4_FA_HPPO = MODELS_V4 / "fa_hppo"
MODELS_DEVELOPMENT = MODELS_DIR / "development"

# Historical path → canonical path (posix, relative to repo root, trailing slash for dirs)
LEGACY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("checkpoints_v4/final_authoritative/V4_BASE_NO_L_FAIL/", "models/v4/fa_hppo/"),
    ("checkpoints_v4/reward_ablation/", "models/v4/reward_ablation/"),
    ("checkpoints_v4/reward_ablation_smoke/", "models/v4/reward_ablation_smoke/"),
    ("checkpoints_v4/final_reward/V4_BASE_NO_L_FAIL/", "models/v4/final_reward/"),
    ("checkpoints_v4/", "models/v4/"),
    ("checkpoints_v3/", "models/v3/"),
    ("checkpoints_v2/", "models/v2/"),
    ("checkpoints_development/", "models/development/"),
    ("checkpoints/", "models/v1/"),
    ("results_v4_paper/", "results/v4/paper/"),
    ("results_v4/", "results/v4/training_figures/"),
    ("results_paper/", "results/v3/paper/"),
    ("results/v4_reward/", "results/v4/reward_development/"),
    ("results/v4_test/", "results/v4/test/"),
    ("results/v3_hppo/", "results/v3/"),
    ("results/final/", "results/v1/final/"),
    ("results/pilot/", "results/v1/pilot/"),
    ("results/summaries/", "results/v1/summaries/"),
    ("scripts/v4_reward/", "scripts/v4/reward/"),
    ("scripts/v4_test/", "scripts/v4/test/"),
    ("scripts/v3_hppo/", "scripts/v3/"),
    ("configs/rl/", "configs/common/rl/"),
    ("configs/physics/", "configs/common/physics/"),
    ("configs/experiments/", "configs/common/experiments/"),
    ("configs/routing/", "configs/common/routing/"),
)


def to_posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a possibly-legacy repo-relative path to an existing canonical Path when possible."""
    text = to_posix(path)
    # Absolute path: try strip to relative if under REPO_ROOT
    p = Path(path)
    if p.is_absolute():
        try:
            text = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except Exception:
            return p
    # Already exists at literal location
    candidate = REPO_ROOT / text
    if candidate.exists():
        return candidate
    # Apply longest matching legacy prefix
    for old, new in sorted(LEGACY_PREFIXES, key=lambda kv: -len(kv[0])):
        if text == old.rstrip("/") or text.startswith(old):
            mapped = new + text[len(old) :] if text.startswith(old) else new.rstrip("/")
            mapped_path = REPO_ROOT / mapped
            if mapped_path.exists() or True:
                return mapped_path
    return candidate


def v4_fa_hppo_checkpoint(seed: int, filename: str = "best.pt") -> Path:
    return MODELS_V4_FA_HPPO / f"seed_{seed}" / filename


def logical_model_path(path: str, filename: str | None = None) -> str:
    """Normalize absolute/legacy checkpoint paths to canonical models/ relative ids."""
    text = to_posix(path)
    # First map legacy checkpoint roots
    lowered = text.lower()
    for old, new in LEGACY_PREFIXES:
        if not old.startswith("checkpoints"):
            continue
        marker = old.rstrip("/")
        idx = lowered.rfind(marker.lower())
        if idx < 0:
            continue
        tail = text[idx + len(marker) :].lstrip("/")
        parts = [part for part in tail.split("/") if part]
        # special-case authoritative → fa_hppo already in mapping
        base = new.rstrip("/")
        if filename:
            if "seed_" in "/".join(parts):
                # keep through seed dir
                seed_parts = []
                for part in parts:
                    seed_parts.append(part)
                    if part.startswith("seed_"):
                        break
                return f"{base}/{'/'.join(seed_parts)}/{filename}"
            if parts:
                return f"{base}/{parts[0]}/{filename}"
            return f"{base}/{filename}"
        if parts:
            return f"{base}/{'/'.join(parts)}"
        return base
    # Already under models/
    for root in ("models/v4", "models/v3", "models/v2", "models/v1", "models/development"):
        idx = text.replace("\\", "/").rfind(root)
        if idx >= 0:
            return text.replace("\\", "/")[idx:]
    return text
