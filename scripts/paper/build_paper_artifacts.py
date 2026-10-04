"""DEPRECATED — do not use as a publication figure pipeline.

Official publication builder (PDF + PNG under results_paper/):

    python scripts/paper/build_results_paper.py
    python scripts/paper/build_results_paper.py --verify

This entry point only delegates to the official builder. It does not rewrite
frozen V3 statistics, checkpoints, or TEST rows, and it does not maintain a
second figure system under paper/figures/.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("build_results_paper.py")
    print(
        "DEPRECATED: scripts/paper/build_paper_artifacts.py\n"
        f"Delegating to {target.as_posix()}",
        file=sys.stderr,
    )
    # Preserve CLI flags such as --verify.
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
