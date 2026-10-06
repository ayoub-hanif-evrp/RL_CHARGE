"""Resume repository migration after partial models move."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
from _migrate_repo_layout import (  # noqa: E402
    REPORT_DIR,
    MIGRATION_JSON,
    archives,
    cleanup_junk,
    deletes,
    migrate_configs,
    migrate_docs,
    migrate_results,
    migrate_scripts,
    moves,
    write_model_inventory,
    write_path_migration_md,
    move_tree,
    rel,
)


def clear_empty_checkpoints_v4() -> None:
    ck = ROOT / "checkpoints_v4"
    if not ck.exists():
        return
    # leave a stub README so OneDrive-locked empty dirs are documented
    stub = ck / "MOVED_TO_models_v4.txt"
    stub.write_text(
        "Checkpoint binaries were moved to models/v4/.\n"
        "See docs/reproducibility/PATH_MIGRATION.md.\n"
        "This directory may remain briefly due to OneDrive locks; safe to delete when unlocked.\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    clear_empty_checkpoints_v4()
    migrate_results()
    migrate_scripts()
    migrate_configs()
    migrate_docs()
    cleanup_junk()
    write_path_migration_md()
    write_model_inventory()
    payload = {
        "moves": moves,
        "deletes": deletes,
        "archives": archives,
        "n_moves": len(moves),
        "n_deletes": len(deletes),
        "hash_mismatches": sum(1 for m in moves if m.get("status") == "HASH_MISMATCH"),
        "phase": "resume_after_models",
    }
    MIGRATION_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["hash_mismatches"]:
        raise SystemExit("hash mismatches")
    print(json.dumps({"n_moves": payload["n_moves"], "n_deletes": payload["n_deletes"]}, indent=2))


if __name__ == "__main__":
    main()
