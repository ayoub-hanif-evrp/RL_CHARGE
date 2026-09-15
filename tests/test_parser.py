"""Parser tests against real instances and hand-built edge cases."""

import pytest
from conftest import (
    FOOTER_WITH_CURB_WEIGHT,
    FOOTER_WITHOUT_CURB_WEIGHT,
    hash_tree,
    node_row,
)

from data import NetworkGroup, NodeType, TerrainVariant, parse_instance
from data.errors import MalformedInstanceError, MissingFieldError

# One representative instance per network-size group, with the node counts
# taken from the raw files.
REPRESENTATIVE = [
    ("Small_Network/5_Customers/Level/c101C5_L.txt", NetworkGroup.SMALL, 5, 3),
    ("Small_Network/10_Customers/Nearly_Level/c101C10_NL.txt", NetworkGroup.SMALL, 10, 5),
    ("Small_Network/15_Customers/Very_Gentle/c103C15_VG.txt", NetworkGroup.SMALL, 15, 5),
    ("Medium_Network/25_Customers/Nearly_Level/c101C25_NL.txt", NetworkGroup.MEDIUM, 25, 21),
    ("Large_Network/50_Customers/Nearly_Level/c101_50_21_NL.txt", NetworkGroup.LARGE, 50, 21),
]


@pytest.mark.parametrize("relative,group,n_customers,n_stations", REPRESENTATIVE)
def test_representative_instances_classify_nodes(
    raw_root, relative, group, n_customers, n_stations
):
    instance = parse_instance(raw_root / relative, root=raw_root)

    assert instance.metadata.network_group is group
    assert len(instance.customers) == n_customers
    assert len(instance.stations) == n_stations
    assert len(instance.depots) == 1
    assert len(instance.nodes) == n_customers + n_stations + 1

    assert all(n.node_type is NodeType.CUSTOMER for n in instance.customers)
    assert all(n.node_type is NodeType.STATION for n in instance.stations)
    assert instance.depot.node_type is NodeType.DEPOT


def test_fields_are_read_verbatim(raw_root):
    instance = parse_instance(
        raw_root / "Small_Network/5_Customers/Level/c101C5_L.txt", root=raw_root
    )

    depot = instance.depot
    assert depot.string_id == "D0"
    assert depot.coordinates == (40.0, 50.0)

    customer = instance.node_by_id("C30")
    assert customer.node_type is NodeType.CUSTOMER
    assert customer.demand == 182.5
    assert customer.ready_time == 355.0
    assert customer.due_date == 407.0
    assert customer.service_time == 90.0
    assert customer.altitude == 0.0


def test_small_network_footer_has_no_curb_weight(raw_root):
    instance = parse_instance(
        raw_root / "Small_Network/5_Customers/Level/c101C5_L.txt", root=raw_root
    )

    assert instance.vehicle.curb_weight is None
    assert instance.vehicle.has_curb_weight is False
    assert instance.vehicle.tank_capacity == 77.75
    assert instance.vehicle.load_capacity == 200.0


def test_medium_network_footer_carries_curb_weight(raw_root):
    instance = parse_instance(
        raw_root / "Medium_Network/25_Customers/Nearly_Level/c101C25_NL.txt", root=raw_root
    )

    assert instance.vehicle.curb_weight == 6350.0
    assert instance.vehicle.has_curb_weight is True
    assert instance.vehicle.load_capacity == 3650.0


def test_footer_labels_are_preserved_verbatim(raw_root):
    """The two subsets describe the same key with different wording."""
    small = parse_instance(
        raw_root / "Small_Network/5_Customers/Level/c101C5_L.txt", root=raw_root
    )
    medium = parse_instance(
        raw_root / "Medium_Network/25_Customers/Nearly_Level/c101C25_NL.txt", root=raw_root
    )

    assert small.vehicle.label_for("Q") == "Vehicle fuel tank capacity"
    assert medium.vehicle.label_for("Q") == "Battery capacity"
    assert small.vehicle.label_for("r") == "fuel consumption rate"
    assert medium.vehicle.label_for("r") == "Energy consumption rate"


def test_compact_filename_convention(raw_root):
    meta = parse_instance(
        raw_root / "Medium_Network/25_Customers/Nearly_Level/rc103C25_NL.txt", root=raw_root
    ).metadata

    assert meta.base_instance == "rc103"
    assert meta.customer_distribution == "rc"
    assert meta.schedule_type == 1
    assert meta.declared_customer_count == 25
    assert meta.declared_station_count is None
    assert meta.terrain_variant is TerrainVariant.NEARLY_LEVEL


def test_expanded_filename_convention(raw_root):
    meta = parse_instance(
        raw_root / "Large_Network/50_Customers/Nearly_Level/r107_50_21_NL.txt", root=raw_root
    ).metadata

    assert meta.base_instance == "r107"
    assert meta.customer_distribution == "r"
    assert meta.schedule_type == 1
    assert meta.declared_customer_count == 50
    assert meta.declared_station_count == 21


def test_schedule_type_two_is_recognised(raw_root):
    meta = parse_instance(
        raw_root / "Small_Network/10_Customers/Level/c202C10_L.txt", root=raw_root
    ).metadata

    assert meta.schedule_type == 2
    assert meta.customer_distribution == "c"


def test_lowercase_terrain_folder_is_accepted(raw_root):
    """One raw folder is spelled 'Nearly_level' where the rest use '_Level'."""
    instance = parse_instance(
        raw_root / "Small_Network/15_Customers/Nearly_level/c103C15_NL.txt", root=raw_root
    )

    assert instance.metadata.terrain_variant is TerrainVariant.NEARLY_LEVEL
    assert instance.metadata.terrain_folder == "Nearly_level"


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("trailing_newline", [True, False])
def test_line_endings_and_trailing_newline_do_not_change_the_result(
    write_instance, newline, trailing_newline
):
    from conftest import build_instance_text

    text = build_instance_text(trailing_newline=trailing_newline).replace("\n", newline)
    path, root = write_instance(text=text)
    instance = parse_instance(path, root=root)

    assert len(instance.nodes) == 4
    assert instance.node_by_id("C1").demand == 182.5
    assert instance.vehicle.tank_capacity == 77.75


def test_every_node_row_is_kept(write_instance):
    """A node whose fields are all zero must not be dropped as blank."""
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        node_row("C1", "c", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)

    assert len(parse_instance(path, root=root).nodes) == 3


def test_wrong_column_count_is_rejected(write_instance):
    truncated = "C9\t\tc\t\t1.0\t\t2.0\t\t3.0\t\t4.0\t\t5.0\t\t6.0"
    path, root = write_instance(nodes=[*_default_nodes(), truncated])

    with pytest.raises(MalformedInstanceError, match="expected 9 columns, found 8"):
        parse_instance(path, root=root)


def test_non_numeric_field_is_rejected(write_instance):
    bad = node_row("C9", "c", "not-a-number", 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    path, root = write_instance(nodes=[*_default_nodes(), bad])

    with pytest.raises(MalformedInstanceError, match="column 'x' is not numeric"):
        parse_instance(path, root=root)


def test_unknown_node_type_is_rejected(write_instance):
    bad = node_row("X9", "x", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    path, root = write_instance(nodes=[*_default_nodes(), bad])

    with pytest.raises(MalformedInstanceError, match="unknown node type code"):
        parse_instance(path, root=root)


def test_missing_required_footer_key_is_rejected(write_instance):
    footer = [line for line in FOOTER_WITHOUT_CURB_WEIGHT if not line.startswith("Q ")]
    path, root = write_instance(footer=footer)

    with pytest.raises(MissingFieldError, match="missing required key 'Q'"):
        parse_instance(path, root=root)


def test_curb_weight_is_optional_not_defaulted(write_instance):
    without, root_a = write_instance(footer=FOOTER_WITHOUT_CURB_WEIGHT)
    with_curb, root_b = write_instance(
        filename="c101C25_NL.txt",
        network="Medium_Network",
        customers="25_Customers",
        terrain="Nearly_Level",
        footer=FOOTER_WITH_CURB_WEIGHT,
    )

    assert parse_instance(without, root=root_a).vehicle.curb_weight is None
    assert parse_instance(with_curb, root=root_b).vehicle.curb_weight == 6350.0


def test_bad_header_is_rejected(write_instance):
    path, root = write_instance(header="id\ttype\tx\ty")

    with pytest.raises(MalformedInstanceError, match="unexpected header columns"):
        parse_instance(path, root=root)


def test_terrain_folder_disagreeing_with_filename_is_rejected(write_instance):
    path, root = write_instance(filename="c101C5_VG.txt", terrain="Level")

    with pytest.raises(MalformedInstanceError, match="disagrees with filename variant"):
        parse_instance(path, root=root)


def test_unrecognised_filename_is_rejected(write_instance):
    path, root = write_instance(filename="mystery_file.txt")

    with pytest.raises(MalformedInstanceError, match="does not follow any known convention"):
        parse_instance(path, root=root)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(MalformedInstanceError, match="does not exist"):
        parse_instance(tmp_path / "nope.txt")


def test_parsing_never_modifies_the_raw_dataset(raw_root):
    from data import load_dataset

    before = hash_tree(raw_root)
    load_dataset(raw_root)
    assert hash_tree(raw_root) == before


def _default_nodes():
    from conftest import DEFAULT_NODES

    return list(DEFAULT_NODES)
