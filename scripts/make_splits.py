"""Write parent-level train/validation/test splits (seed 42, Large repair)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import SPLITS_DIR  # noqa: E402
from experiments.split import assign_splits, assert_no_leakage, load_parents, write_splits  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SPLITS_DIR)
    args = parser.parse_args(argv)
    parents = load_parents()
    assignment = assign_splits(parents)
    assert_no_leakage(assignment)
    write_splits(assignment, args.out)
    print(
        f"train_parents={len(assignment.train)} "
        f"val_parents={len(assignment.validation)} "
        f"test_parents={len(assignment.test)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
