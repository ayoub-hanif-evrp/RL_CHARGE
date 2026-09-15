"""Seed lists. Paper matrix is five seeds; extended supports ten without code changes."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import List

from data.paths import REPO_ROOT

SEEDS_TOML = REPO_ROOT / "configs" / "experiments" / "seeds.toml"


def load_seed_list(name: str = "paper", path: Path | None = None) -> List[int]:
    text = str(name).strip()
    if "," in text:
        return [int(part) for part in text.split(",") if part.strip()]
    try:
        return [int(text)]
    except ValueError:
        pass
    with Path(path or SEEDS_TOML).open("rb") as handle:
        raw = tomllib.load(handle)
    if text not in raw:
        raise KeyError(f"seed list {name!r} missing from {path or SEEDS_TOML}")
    return [int(s) for s in raw[text]]
