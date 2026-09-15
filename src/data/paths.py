"""Filesystem locations for the EVRPTW-GR data layer.

Every path is derived from this file's own location, so imports and scripts
behave the same no matter which directory the process was started from.
"""

from pathlib import Path

from .errors import DatasetNotFoundError

# src/data/paths.py -> src/data -> src -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_EVRPTW_GR_DIR = RAW_DIR / "EVRPTW_GR"
PROCESSED_DIR = DATA_DIR / "processed"
ROUTES_DIR = DATA_DIR / "routes"
SPLITS_DIR = DATA_DIR / "splits"

INSTANCE_SUFFIX = ".txt"


def resolve_dataset_root(root=None):
    """Return the raw EVRPTW-GR root, checking that it holds instance files."""
    root = RAW_EVRPTW_GR_DIR if root is None else Path(root)
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise DatasetNotFoundError(f"dataset root does not exist: {root}")
    if not any(root.rglob(f"*{INSTANCE_SUFFIX}")):
        raise DatasetNotFoundError(f"no {INSTANCE_SUFFIX} instance files under: {root}")
    return root


def iter_instance_files(root=None):
    """Yield every instance file under ``root`` in a stable, portable order."""
    root = resolve_dataset_root(root)
    return sorted(root.rglob(f"*{INSTANCE_SUFFIX}"), key=lambda p: p.relative_to(root).as_posix())
