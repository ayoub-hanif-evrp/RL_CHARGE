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


PINNED_HASHES = {
    "corpus.jsonl": "1796d817ff058fe0559021ab79ca97087ef3646c57744c2c883ecc37e85451e0",
    "manifest.csv": "41bb8be3245f8f3b0800db4ba74cb19a2003236b6fbd0dbc1237e7cdfaee104d",
    "corpus_metadata.json": "1ca8725199842f0a8d4a4e6fac2a31931ea3c9aaca833c2d253c6147469d9a96",
    "train.json": "3aecd73e9472b8572e899a7e2da3f1f874f0e5a13250f5cd6d76d812002913ce",
    "validation.json": "1379aef7e92c308f90ab78674dec9994db72078c96cb1020ecc49d33eef69af4",
    "test.json": "9bf39fbf1f27aa57146879fb71922b0216f24f1eb37a4fd9a9a43b44b6c4b6d0",
    "split_metadata.json": "4740ab71737bfd9c04d37f258e57bfa4854a526da7c03b218ffe877818128900",
}


def _check_pinned(hashes: dict) -> list[str]:
    errors = []
    for name, expected in PINNED_HASHES.items():
        got = hashes.get(name)
        if got != expected:
            errors.append(f"{name}: expected {expected}, got {got}")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Fail if the tree is dirty, hashes drift, pytest fails, audit fails, or leakage is detected.",
    )
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
    hashes = frozen_hashes()
    pin_errors = _check_pinned(hashes)
    dirty = git_dirty()
    payload = {
        "git_sha": git_sha(),
        "git_dirty": dirty,
        "hashes": hashes,
        "device": device_info(),
        "corpus_ok": report.ok,
        "n_routes": report.n_routes,
        "n_instances": report.n_instances,
        "leakage_ok": True,
        "pinned_ok": not pin_errors,
        "pin_errors": pin_errors,
        "mode": "paper" if args.paper else "default",
        "note": "Paper-scale training must start from a frozen commit of this SHA with git_dirty=false.",
    }
    if args.paper:
        failures = []
        if dirty:
            failures.append("git_dirty=true")
        if pin_errors:
            failures.extend(pin_errors)
        if not report.ok:
            failures.append("corpus_audit_failed")
        if failures:
            print(canonical_dumps({"paper_preflight_failed": failures, **payload}))
            return 1
    out_dir = RESULTS_DIR / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_freeze.json").write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    print(canonical_dumps(payload))
    print(f"device={detect_device()} dirty={dirty} sha={git_sha()} paper={args.paper}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
