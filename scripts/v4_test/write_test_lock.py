"""Write or verify results/v4_test/TEST_LOCK.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v4_test"))

from common import GENERATOR_SEED_START, TEST_CORPUS, TEST_SPLIT, V4, dump_json, lf_sha256, load_json  # noqa: E402

LOCK = V4 / "TEST_LOCK.json"


def locked_files() -> list[str]:
    files = [
        "data/routes_v2/synthcharge_v4_test/corpus.jsonl",
        "data/routes_v2/synthcharge_v4_test/certificates.jsonl",
        "data/routes_v2/synthcharge_v4_test/manifest.csv",
        "data/routes_v2/synthcharge_v4_test/corpus_metadata.json",
        "data/splits_v2/synthcharge_v4_test.json",
        "results/v4_test/configs/synthcharge_v4_test.json",
        "results/v4_test/PAPER_PROTOCOL.json",
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
    if min(seeds) < GENERATOR_SEED_START:
        raise SystemExit(f"TEST seed below {GENERATOR_SEED_START}")
    return {
        "hash_convention": "sha256 of LF-normalized file bytes",
        "benchmark": "synthcharge_v4_reward_paper_test",
        "role": "fresh_external_confirmatory_TEST_for_V4_reward",
        "n_routes": split["n_routes"],
        "seed_start": GENERATOR_SEED_START,
        "generator_seeds_min": min(seeds),
        "generator_seeds_max": max(seeds),
        "route_ids_in_order": split["route_ids"],
        "quota_summary": split.get("quota_summary"),
        "independent_of_v3_test": True,
        "files": {rel: lf_sha256(ROOT / rel) for rel in locked_files()},
    }


def verify() -> None:
    lock = load_json(LOCK)
    bad = [rel for rel, digest in lock["files"].items() if lf_sha256(ROOT / rel) != digest]
    if bad:
        raise SystemExit(f"V4 TEST_LOCK mismatch: {bad[:10]}")
    print(f"V4 TEST_LOCK verified: {len(lock['files'])} files")


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
