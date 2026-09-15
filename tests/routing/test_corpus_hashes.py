"""Pinned SHA-256 of the frozen compact corpus. Later experiments must not rewrite these files."""

import hashlib
from pathlib import Path

from data.paths import ROUTES_DIR, SPLITS_DIR

PINNED_SHA256 = {
    "corpus.jsonl": "1796d817ff058fe0559021ab79ca97087ef3646c57744c2c883ecc37e85451e0",
    "manifest.csv": "41bb8be3245f8f3b0800db4ba74cb19a2003236b6fbd0dbc1237e7cdfaee104d",
    "corpus_metadata.json": "1ca8725199842f0a8d4a4e6fac2a31931ea3c9aaca833c2d253c6147469d9a96",
}

PINNED_SPLIT_SHA256 = {
    "train.json": "3aecd73e9472b8572e899a7e2da3f1f874f0e5a13250f5cd6d76d812002913ce",
    "validation.json": "1379aef7e92c308f90ab78674dec9994db72078c96cb1020ecc49d33eef69af4",
    "test.json": "9bf39fbf1f27aa57146879fb71922b0216f24f1eb37a4fd9a9a43b44b6c4b6d0",
    "split_metadata.json": "4740ab71737bfd9c04d37f258e57bfa4854a526da7c03b218ffe877818128900",
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


def test_tracked_split_files_are_frozen():
    for name, expected in PINNED_SPLIT_SHA256.items():
        path = SPLITS_DIR / name
        assert path.is_file(), f"missing frozen split file {name}"
        assert _sha256(path) == expected
