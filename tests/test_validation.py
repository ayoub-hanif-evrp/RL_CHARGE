"""Validation tests, driven by hand-built instances rather than real files."""

import pytest
from conftest import DEFAULT_NODES, FOOTER_WITHOUT_CURB_WEIGHT, node_row

from data import Severity, parse_instance, validate_instance


def codes(issues, severity=None):
    return {i.code for i in issues if severity is None or i.severity is severity}


def issues_for(write_instance, **kwargs):
    path, root = write_instance(**kwargs)
    return validate_instance(parse_instance(path, root=root))


def test_a_clean_instance_reports_no_errors(write_instance):
    found = issues_for(write_instance, filename="c101C2_L.txt")
    assert codes(found, Severity.ERROR) == set()


def test_duplicate_node_id_is_an_error(write_instance):
    nodes = [*DEFAULT_NODES, node_row("C1", "c", 1.0, 2.0, 10.0, 0.0, 100.0, 5.0, 0.0)]
    found = issues_for(write_instance, filename="c101C3_L.txt", nodes=nodes)
    assert "duplicate_node_id" in codes(found, Severity.ERROR)


def test_two_depots_is_an_error(write_instance):
    nodes = [*DEFAULT_NODES, node_row("D1", "d", 1.0, 2.0, 0.0, 0.0, 100.0, 0.0, 0.0)]
    found = issues_for(write_instance, filename="c101C2_L.txt", nodes=nodes)
    assert "depot_count" in codes(found, Severity.ERROR)


def test_instance_without_customers_is_an_error(write_instance):
    nodes = [n for n in DEFAULT_NODES if not n.startswith("C")]
    found = issues_for(write_instance, filename="c101C0_L.txt", nodes=nodes)
    assert "no_customers" in codes(found, Severity.ERROR)


def test_instance_without_stations_is_an_error(write_instance):
    nodes = [n for n in DEFAULT_NODES if not n.startswith("S")]
    found = issues_for(write_instance, filename="c101C2_L.txt", nodes=nodes)
    assert "no_stations" in codes(found, Severity.ERROR)


def test_inverted_time_window_is_an_error(write_instance):
    nodes = [*DEFAULT_NODES, node_row("C3", "c", 1.0, 2.0, 10.0, 500.0, 100.0, 5.0, 0.0)]
    found = issues_for(write_instance, filename="c101C3_L.txt", nodes=nodes)
    assert "inverted_time_window" in codes(found, Severity.ERROR)


def test_negative_demand_and_service_time_are_errors(write_instance):
    nodes = [*DEFAULT_NODES, node_row("C3", "c", 1.0, 2.0, -5.0, 0.0, 100.0, -1.0, 0.0)]
    found = issues_for(write_instance, filename="c101C3_L.txt", nodes=nodes)
    assert {"negative_demand", "negative_service_time"} <= codes(found, Severity.ERROR)


def test_id_prefix_disagreeing_with_type_is_an_error(write_instance):
    nodes = [*DEFAULT_NODES, node_row("C7", "f", 1.0, 2.0, 0.0, 0.0, 100.0, 0.0, 0.0)]
    found = issues_for(write_instance, filename="c101C2_L.txt", nodes=nodes)
    assert "id_type_mismatch" in codes(found, Severity.ERROR)


def test_customer_count_disagreeing_with_filename_is_an_error(write_instance):
    # The layout supplies two customers, so a filename claiming five is wrong.
    found = issues_for(write_instance, filename="c101C5_L.txt")
    assert "customer_count_mismatch" in codes(found, Severity.ERROR)


def test_station_count_disagreeing_with_filename_is_an_error(write_instance):
    found = issues_for(
        write_instance,
        filename="c101_2_9_NL.txt",
        network="Large_Network",
        customers="50_Customers",
        terrain="Nearly_Level",
    )
    assert "station_count_mismatch" in codes(found, Severity.ERROR)


def test_demand_above_capacity_is_a_warning_not_an_error(write_instance):
    """Mirrors the published Small_Network inconsistency: it must not be fatal."""
    nodes = [
        *DEFAULT_NODES,
        node_row("C3", "c", 1.0, 2.0, 730.0, 0.0, 100.0, 5.0, 0.0),
    ]
    found = issues_for(write_instance, filename="c101C3_L.txt", nodes=nodes)

    assert "demand_exceeds_capacity" in codes(found, Severity.WARNING)
    assert "demand_exceeds_capacity" not in codes(found, Severity.ERROR)


def test_non_zero_altitude_in_a_level_file_is_a_warning(write_instance):
    nodes = [
        node_row("D0", "d", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
        node_row("S0", "f", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
        node_row("C1", "c", 45.0, 68.0, 182.5, 78.0, 140.0, 90.0, 0.25),
        node_row("C2", "c", 42.0, 66.0, 182.5, 200.0, 300.0, 90.0, 0.0),
    ]
    found = issues_for(write_instance, filename="c101C2_L.txt", nodes=nodes)

    assert "level_altitude_non_zero" in codes(found, Severity.WARNING)
    assert codes(found, Severity.ERROR) == set()


def test_absent_curb_weight_is_reported_as_a_warning(write_instance):
    found = issues_for(
        write_instance, filename="c101C2_L.txt", footer=FOOTER_WITHOUT_CURB_WEIGHT
    )
    assert "missing_curb_weight" in codes(found, Severity.WARNING)


def test_non_zero_altitude_is_allowed_outside_level_variants(write_instance):
    """Altitude is unbounded in the published data, so no range is enforced."""
    nodes = [
        node_row("D0", "d", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
        node_row("S0", "f", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
        node_row("C1", "c", 45.0, 68.0, 182.5, 78.0, 140.0, 90.0, 1.3),
        node_row("C2", "c", 42.0, 66.0, 182.5, 200.0, 300.0, 90.0, -0.928571),
    ]
    found = issues_for(
        write_instance,
        filename="c101C2_NL.txt",
        terrain="Nearly_Level",
        nodes=nodes,
    )
    assert codes(found, Severity.ERROR) == set()


def test_vg_ratio_mismatch_is_detected_across_files(write_instance, tmp_path):
    from data import validate_terrain_consistency

    def nodes_with_customer_altitude(altitude):
        return [
            node_row("D0", "d", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
            node_row("S0", "f", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
            node_row("C1", "c", 45.0, 68.0, 182.5, 78.0, 140.0, 90.0, altitude),
            node_row("C2", "c", 42.0, 66.0, 182.5, 200.0, 300.0, 90.0, 0.0),
        ]

    nl_path, root = write_instance(
        filename="c101C2_NL.txt",
        terrain="Nearly_Level",
        nodes=nodes_with_customer_altitude(0.2),
    )
    # 0.2 * 1.5 would be 0.3; 0.9 breaks the documented relationship.
    vg_path, _ = write_instance(
        filename="c101C2_VG.txt",
        terrain="Very_Gentle",
        nodes=nodes_with_customer_altitude(0.9),
    )

    instances = [parse_instance(nl_path, root=root), parse_instance(vg_path, root=root)]
    assert "vg_ratio_mismatch" in codes(validate_terrain_consistency(instances))


def test_vg_ratio_holding_produces_no_warning(write_instance):
    from data import validate_terrain_consistency

    def nodes_with_customer_altitude(altitude):
        return [
            node_row("D0", "d", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
            node_row("S0", "f", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
            node_row("C1", "c", 45.0, 68.0, 182.5, 78.0, 140.0, 90.0, altitude),
            node_row("C2", "c", 42.0, 66.0, 182.5, 200.0, 300.0, 90.0, 0.0),
        ]

    nl_path, root = write_instance(
        filename="c101C2_NL.txt",
        terrain="Nearly_Level",
        nodes=nodes_with_customer_altitude(0.2),
    )
    vg_path, _ = write_instance(
        filename="c101C2_VG.txt",
        terrain="Very_Gentle",
        nodes=nodes_with_customer_altitude(0.3),
    )

    instances = [parse_instance(nl_path, root=root), parse_instance(vg_path, root=root)]
    assert validate_terrain_consistency(instances) == []


def test_missing_variants_are_reported(write_instance):
    from data import find_missing_variants

    path, root = write_instance(filename="c101C2_NL.txt", terrain="Nearly_Level")
    missing = find_missing_variants([parse_instance(path, root=root)])

    assert {m["missing_variant"] for m in missing} == {"LEVEL", "VERY_GENTLE"}


@pytest.mark.parametrize("severity", list(Severity))
def test_issues_serialise_for_the_summary_file(write_instance, severity):
    found = issues_for(write_instance, filename="c101C5_L.txt")
    for issue in found:
        payload = issue.as_dict()
        assert set(payload) == {
            "severity",
            "code",
            "message",
            "instance_id",
            "relative_path",
        }
        assert payload["severity"] in {s.value for s in Severity}
