"""End-to-end checks against the real raw dataset.

The expected counts below were measured from the published files. They are
pinned deliberately: if a future parser change starts dropping nodes or
swallowing a known anomaly, these tests fail rather than quietly agreeing.
"""

import os
from collections import Counter

import pytest
from conftest import hash_tree

from data import Severity, find_missing_variants, scan_dataset
from data.paths import iter_instance_files, resolve_dataset_root

EXPECTED_FILES = 124
EXPECTED_BY_GROUP = {"Small_Network": 108, "Medium_Network": 12, "Large_Network": 4}
EXPECTED_BY_CUSTOMER_COUNT = {5: 36, 10: 36, 15: 36, 25: 12, 50: 4}

#: Small_Network kept the original Schneider load capacities while carrying
#: rescaled demands, so a single demand outruns capacity in these files.
EXPECTED_CAPACITY_WARNINGS = 54
#: Three Very Gentle files break the documented "1.5x Nearly Level" rule.
EXPECTED_VG_MISMATCHES = 3
#: Curb weight is absent from every Small_Network file.
EXPECTED_MISSING_CURB_WEIGHT = 108
#: Medium and Large ship only the Nearly Level variant in this download.
EXPECTED_MISSING_VARIANT_FILES = 32


@pytest.fixture(scope="module")
def scan(raw_root):
    return scan_dataset(raw_root)


def test_every_file_is_discovered(raw_root):
    assert len(list(iter_instance_files(raw_root))) == EXPECTED_FILES


def test_every_file_parses_without_error(scan):
    assert scan.unparsed == []
    assert len(scan.instances) == EXPECTED_FILES
    assert [i.message for i in scan.errors] == []


def test_instances_are_grouped_as_published(scan):
    by_group = Counter(i.metadata.network_group.value for i in scan.instances)
    assert dict(by_group) == EXPECTED_BY_GROUP


def test_customer_counts_match_the_published_sizes(scan):
    by_count = Counter(len(i.customers) for i in scan.instances)
    assert dict(by_count) == EXPECTED_BY_CUSTOMER_COUNT


def test_parsed_counts_agree_with_filenames(scan):
    for instance in scan.instances:
        meta = instance.metadata
        assert len(instance.customers) == meta.declared_customer_count, meta.relative_path
        if meta.declared_station_count is not None:
            assert len(instance.stations) == meta.declared_station_count, meta.relative_path


def test_every_instance_has_exactly_one_depot_and_some_stations(scan):
    for instance in scan.instances:
        assert len(instance.depots) == 1, instance.metadata.relative_path
        assert instance.stations, instance.metadata.relative_path


def test_known_capacity_anomaly_is_reported_and_not_fatal(scan):
    warnings = scan.issues_for("demand_exceeds_capacity")
    assert len(warnings) == EXPECTED_CAPACITY_WARNINGS
    assert all(w.severity is Severity.WARNING for w in warnings)
    # The anomaly is confined to Small_Network.
    assert all(w.relative_path.startswith("Small_Network/") for w in warnings)


def test_known_vg_ratio_anomaly_is_reported(scan):
    mismatches = scan.issues_for("vg_ratio_mismatch")
    assert len(mismatches) == EXPECTED_VG_MISMATCHES
    assert {m.instance_id for m in mismatches} == {
        "r203C10_VG",
        "c208C15_VG",
        "c101C5_VG",
    }


def test_absent_curb_weight_is_confined_to_small_network(scan):
    issues = scan.issues_for("missing_curb_weight")
    assert len(issues) == EXPECTED_MISSING_CURB_WEIGHT
    assert all(i.relative_path.startswith("Small_Network/") for i in issues)

    for instance in scan.instances:
        if instance.metadata.network_group.value == "Small_Network":
            assert instance.vehicle.curb_weight is None
        else:
            assert instance.vehicle.curb_weight == 6350.0


def test_level_variant_altitudes_are_all_zero(scan):
    assert scan.issues_for("level_altitude_non_zero") == []


def test_missing_terrain_variants_are_reported(scan):
    missing = find_missing_variants(scan.instances)
    assert len(missing) == EXPECTED_MISSING_VARIANT_FILES
    # Small_Network is complete; the gap is entirely Medium and Large.
    assert {m["network_group"] for m in missing} == {"Medium_Network", "Large_Network"}
    assert {m["missing_variant"] for m in missing} == {"LEVEL", "VERY_GENTLE"}


def test_scanning_does_not_modify_the_raw_dataset(raw_root):
    before = hash_tree(raw_root)
    scan_dataset(raw_root)
    assert hash_tree(raw_root) == before


def test_dataset_resolves_from_an_unrelated_working_directory(tmp_path, raw_root):
    original = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert resolve_dataset_root() == raw_root
        assert len(list(iter_instance_files())) == EXPECTED_FILES
    finally:
        os.chdir(original)
