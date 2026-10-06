"""Audit the frozen-route corpus. Does not drop routes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import ROUTES_DIR  # noqa: E402
from routing.audit import audit_corpus  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, default=ROUTES_DIR)
    args = parser.parse_args(argv)
    report = audit_corpus(args.routes)
    print(f"routes={report.n_routes} instances={report.n_instances} issues={report.n_issues} ok={report.ok}")
    for issue in report.issues:
        print(f"{issue.kind}: {issue.message}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
