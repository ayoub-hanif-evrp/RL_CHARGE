"""Mandatory preflight before paper-scale experiment seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import RESULTS_DIR, ROUTES_DIR  # noqa: E402
from experiments.provenance import detect_device, device_info, frozen_hashes, git_dirty, git_sha  # noqa: E402
from experiments.split import assert_no_leakage, load_parents  # noqa: E402
from routing.audit import audit_corpus  # noqa: E402
from routing.serialize import canonical_dumps, read_jsonl  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_pytest:
        proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)
        if proc.returncode != 0:
            return proc.returncode
    report = audit_corpus(ROUTES_DIR)
    parents = load_parents()
    from experiments.split import assign_splits

    assignment = assign_splits(parents, seed=42)
    routes = read_jsonl(ROUTES_DIR / "corpus.jsonl") if (ROUTES_DIR / "corpus.jsonl").is_file() else None
    assert_no_leakage(assignment, routes=routes)
    payload = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "hashes": frozen_hashes(),
        "device": device_info(),
        "corpus_ok": report.ok,
        "n_routes": report.n_routes,
        "n_instances": report.n_instances,
        "leakage_ok": True,
        "note": "Paper-scale M1 training must start from a frozen commit of this SHA.",
    }
    out_dir = RESULTS_DIR / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_freeze.json").write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    print(canonical_dumps(payload))
    print(f"device={detect_device()} dirty={git_dirty()} sha={git_sha()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
