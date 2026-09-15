"""Shared fixtures for the data-layer tests."""

import hashlib

import pytest

from data.paths import RAW_EVRPTW_GR_DIR

HEADER = (
    "StringID\tType\t\tx\t\ty\t\tdemand\t\tReadyTime\tDueDate\t\tServiceTime\taltitude"
)

#: Footer used by Medium_Network and Large_Network, including curb weight.
FOOTER_WITH_CURB_WEIGHT = [
    "Q Battery capacity /79.69/",
    "C Vehicle load capacity /3650.0/",
    "r Energy consumption rate /1.0/",
    "g inverse recharging rate /3.39/",
    "v average Velocity /1.0/",
    "M Vehicle curb weight /6350.0/",
]

#: Footer used by Small_Network, which omits curb weight entirely.
FOOTER_WITHOUT_CURB_WEIGHT = [
    "Q Vehicle fuel tank capacity /77.75/",
    "C Vehicle load capacity /200.0/",
    "r fuel consumption rate /1.0/",
    "g inverse refueling rate /3.47/",
    "v average Velocity /1.0/",
]


def node_row(string_id, type_code, x, y, demand, ready, due, service, altitude):
    fields = [string_id, type_code, x, y, demand, ready, due, service, altitude]
    return "\t\t".join(str(f) for f in fields)


#: A minimal but valid instance: one depot, one station, two customers.
DEFAULT_NODES = [
    node_row("D0", "d", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
    node_row("S0", "f", 40.0, 50.0, 0.0, 0.0, 1236.0, 0.0, 0.0),
    node_row("C1", "c", 45.0, 68.0, 182.5, 78.0, 140.0, 90.0, 0.0),
    node_row("C2", "c", 42.0, 66.0, 365.0, 200.0, 300.0, 90.0, 0.0),
]


def build_instance_text(nodes=None, footer=None, header=HEADER, trailing_newline=True):
    """Assemble raw instance text from parts, mirroring the real layout."""
    nodes = DEFAULT_NODES if nodes is None else nodes
    footer = FOOTER_WITHOUT_CURB_WEIGHT if footer is None else footer
    lines = [header, *nodes, "", *footer]
    text = "\n".join(lines)
    return text + "\n" if trailing_newline else text


@pytest.fixture
def write_instance(tmp_path):
    """Write a synthetic instance into a realistic directory layout.

    Returns a callable giving back ``(path, root)`` so the parser resolves
    metadata from the directory structure exactly as it does for real data.
    """

    def _write(
        filename="c101C5_L.txt",
        text=None,
        network="Small_Network",
        customers="5_Customers",
        terrain="Level",
        **kwargs,
    ):
        root = tmp_path / "EVRPTW_GR"
        directory = root / network / customers / terrain
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(
            build_instance_text(**kwargs) if text is None else text, encoding="utf-8"
        )
        return path, root

    return _write


@pytest.fixture(scope="session")
def raw_root():
    if not RAW_EVRPTW_GR_DIR.is_dir():
        pytest.skip(f"raw dataset not available at {RAW_EVRPTW_GR_DIR}")
    return RAW_EVRPTW_GR_DIR


def hash_tree(root):
    """Content hash of every file under ``root``, used to prove immutability."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()
