"""Pinned SHA-256 of the frozen compact corpus. Later experiments must not rewrite these files."""

import hashlib
from pathlib import Path

from data.paths import ROUTES_DIR

PINNED_SHA256 = {
    "corpus.jsonl": "1796d817ff058fe0559021ab79ca97087ef3646c57744c2c883ecc37e85451e0",
    "manifest.csv": "41bb8be3245f8f3b0800db4ba74cb19a2003236b6fbd0dbc1237e7cdfaee104d",
    "corpus_metadata.json": "1ca8725199842f0a8d4a4e6fac2a31931ea3c9aaca833c2d253c6147469d9a96",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracked_corpus_files_are_frozen():
    import json

    for name, expected in PINNED_SHA256.items():
        path = ROUTES_DIR / name
        assert path.is_file(), f"missing frozen corpus file {name}"
        assert _sha256(path) == expected
    metadata = json.loads((ROUTES_DIR / "corpus_metadata.json").read_text(encoding="utf-8"))
    assert metadata["sha256"]["corpus.jsonl"] == PINNED_SHA256["corpus.jsonl"]
    assert metadata["sha256"]["manifest.csv"] == PINNED_SHA256["manifest.csv"]
    assert metadata["fixed_vehicle_cost_formula"] == "F = 2 * n_customers * Dmax + 1"
    assert metadata["fleet_policy"] == "unrestricted_fleet_with_fixed_cost"
    assert metadata["pyvrp_version"] == "0.14.0"
    assert metadata["n_instances"] == 124
    assert metadata["n_routes"] == 258
    assert metadata["seed"] == 42
