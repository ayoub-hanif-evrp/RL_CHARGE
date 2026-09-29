"""Write or verify results/v2/final/TEST_LOCK.json.

Hashes are SHA-256 of LF-normalized bytes, which equal the committed git blob
content on every platform.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "results" / "v2" / "final" / "TEST_LOCK.json"


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def locked_files() -> list[str]:
    files = [
        "data/routes_v2/synthcharge_final/test/corpus.jsonl",
        "data/routes_v2/synthcharge_final/test/certificates.jsonl",
        "data/routes_v2/synthcharge_final/test/manifest.csv",
        "data/routes_v2/synthcharge_final/test/corpus_metadata.json",
        "data/splits_v2/synthcharge_test.json",
        "data/routes_v2/legacy_evrptwgr_challenge/corpus.jsonl",
        "data/routes_v2/legacy_evrptwgr_challenge/certificates.jsonl",
        "data/routes_v2/legacy_evrptwgr_challenge/manifest.csv",
        "data/routes_v2/legacy_evrptwgr_challenge/corpus_metadata.json",
    ]
    corpus = ROOT / "data" / "routes_v2" / "synthcharge_final" / "test" / "corpus.jsonl"
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if line.strip():
            files.append(json.loads(line)["relative_path"])
    return files


def build() -> dict:
    split = json.loads((ROOT / "data" / "splits_v2" / "synthcharge_test.json").read_text(encoding="utf-8"))
    return {
        "hash_convention": "sha256 of LF-normalized file bytes",
        "synthcharge_test": {
            "role": "fresh_external_confirmatory_benchmark",
            "n_routes": split["n_routes"],
            "route_ids_in_order": split["route_ids"],
        },
        "legacy_evrptwgr_challenge": {
            "role": "legacy_same_domain_challenge",
            "not_a_fresh_test": True,
        },
        "files": {rel: lf_sha256(ROOT / rel) for rel in locked_files()},
    }


def verify() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = [rel for rel, digest in lock["files"].items() if lf_sha256(ROOT / rel) != digest]
    if bad:
        raise SystemExit(f"TEST_LOCK mismatch: {bad}")
    print(f"TEST_LOCK verified: {len(lock['files'])} files")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
        verify()
