"""Write or verify results/v3/TEST_LOCK.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from common import TEST_CORPUS, TEST_SPLIT, V3, dump_json, lf_sha256, load_json  # noqa: E402
from repo_paths import resolve_repo_path  # noqa: E402

LOCK = V3 / "TEST_LOCK.json"


def locked_files() -> list[str]:
    files = [
        "data/routes_v2/synthcharge_v3_test/corpus.jsonl",
        "data/routes_v2/synthcharge_v3_test/certificates.jsonl",
        "data/routes_v2/synthcharge_v3_test/manifest.csv",
        "data/routes_v2/synthcharge_v3_test/corpus_metadata.json",
        "data/splits_v2/synthcharge_v3_test.json",
        "results/v3/configs/synthcharge_paper_test.json",
        "results/v3/PAPER_PROTOCOL.json",
    ]
    for line in TEST_CORPUS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            files.append(json.loads(line)["relative_path"])
    return files


def build() -> dict:
    split = load_json(TEST_SPLIT)
    if int(split["n_routes"]) != 180:
        raise SystemExit(f"expected 180 TEST routes, got {split['n_routes']}")
    seeds = split["generator_seeds"]
    if min(seeds) < 400000:
        raise SystemExit("TEST seed below 400000")
    return {
        "hash_convention": "sha256 of LF-normalized file bytes",
        "benchmark": "synthcharge_v3_hppo_paper_test",
        "role": "fresh_external_confirmatory_TEST_for_FA_HPPO_paper",
        "n_routes": split["n_routes"],
        "seed_start": 400000,
        "generator_seeds_min": min(seeds),
        "generator_seeds_max": max(seeds),
        "route_ids_in_order": split["route_ids"],
        "quota_summary": split.get("quota_summary"),
        "files": {rel: lf_sha256(resolve_repo_path(rel)) for rel in locked_files()},
    }


def verify() -> None:
    lock = load_json(LOCK)
    bad = [rel for rel, digest in lock["files"].items() if lf_sha256(resolve_repo_path(rel)) != digest]
    if bad:
        raise SystemExit(f"V3 TEST_LOCK mismatch: {bad[:10]}")
    print(f"V3 TEST_LOCK verified: {len(lock['files'])} files")


def main() -> None:
    if "--verify" in sys.argv:
        verify()
        return
    if not TEST_CORPUS.is_file():
        raise SystemExit("generate TEST first")
    dump_json(LOCK, build())
    verify()


if __name__ == "__main__":
    main()
