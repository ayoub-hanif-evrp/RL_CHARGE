"""Typed representation of an EVRPTW-GR instance.

All values are stored exactly as they appear in the raw files. No unit is
converted, and no quantity is derived: distances are not turned into travel
times or energy, and no state of charge, slope, or consumption figure is
computed here.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class NodeType(Enum):
    """Node roles, keyed by the single-letter code in the ``Type`` column."""

    DEPOT = "d"
    STATION = "f"
    CUSTOMER = "c"

    @classmethod
    def from_code(cls, code):
        try:
            return cls(code)
        except ValueError:
            raise ValueError(f"unknown node type code {code!r}") from None


class TerrainVariant(Enum):
    """Terrain variants, keyed by the filename suffix."""

    LEVEL = "L"
    NEARLY_LEVEL = "NL"
    VERY_GENTLE = "VG"


class NetworkGroup(Enum):
    """Top-level dataset subsets."""

    SMALL = "Small_Network"
    MEDIUM = "Medium_Network"
    LARGE = "Large_Network"


#: Expected ``StringID`` prefix for each node type.
NODE_ID_PREFIX = {NodeType.DEPOT: "D", NodeType.STATION: "S", NodeType.CUSTOMER: "C"}

#: The nine raw column names, in file order.
COLUMN_NAMES = (
    "StringID",
    "Type",
    "x",
    "y",
    "demand",
    "ReadyTime",
    "DueDate",
    "ServiceTime",
    "altitude",
)


@dataclass(frozen=True)
class Node:
    """One row of an instance file, in raw units."""

    string_id: str
    node_type: NodeType
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float
    altitude: float
    line_number: int

    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class VehicleParameter:
    """A single footer entry, keeping the raw label alongside the value.

    The label wording differs between subsets -- Small_Network says "Vehicle
    fuel tank capacity" where Medium and Large say "Battery capacity" -- so the
    original text is retained as evidence rather than normalised away.
    """

    key: str
    label: str
    value: float


@dataclass(frozen=True)
class VehicleParameters:
    """The footer block of an instance file.

    ``curb_weight`` is ``None`` for the 108 Small_Network files, which omit the
    ``M`` entry entirely. It is never filled in with a substitute value.
    """

    tank_capacity: float
    load_capacity: float
    consumption_rate: float
    inverse_refueling_rate: float
    average_velocity: float
    curb_weight: Optional[float]
    raw: Tuple[VehicleParameter, ...]

    @property
    def has_curb_weight(self) -> bool:
        return self.curb_weight is not None

    def label_for(self, key) -> Optional[str]:
        """Return the raw label text recorded for a footer key."""
        for parameter in self.raw:
            if parameter.key == key:
                return parameter.label
        return None


@dataclass(frozen=True)
class InstanceMetadata:
    """Provenance recovered from an instance's path and filename."""

    instance_id: str
    path: Path
    relative_path: str
    network_group: NetworkGroup
    customer_folder: str
    terrain_variant: TerrainVariant
    #: Schneider identifier, e.g. "c101".
    base_instance: str
    #: Geographic distribution of customers: "c", "r", or "rc".
    customer_distribution: str
    #: Schneider problem type: 1 is a short horizon, 2 a long one.
    schedule_type: int
    #: Customer count encoded in the filename, kept separate from the parsed
    #: count so the two can be cross-checked.
    declared_customer_count: int
    #: Station count encoded in the filename. Only the Large_Network naming
    #: convention carries this, so it is ``None`` elsewhere.
    declared_station_count: Optional[int]
    #: Terrain folder name as spelled on disk. Casing is inconsistent in the
    #: raw dataset ("Nearly_level" in one folder, "Nearly_Level" elsewhere).
    terrain_folder: str


@dataclass(frozen=True)
class EVRPTWGRInstance:
    """A fully parsed instance file."""

    metadata: InstanceMetadata
    nodes: Tuple[Node, ...]
    vehicle: VehicleParameters

    @property
    def depots(self) -> Tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.node_type is NodeType.DEPOT)

    @property
    def stations(self) -> Tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.node_type is NodeType.STATION)

    @property
    def customers(self) -> Tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.node_type is NodeType.CUSTOMER)

    @property
    def depot(self) -> Node:
        """The single depot.

        Raises if the instance does not have exactly one, rather than silently
        picking the first.
        """
        depots = self.depots
        if len(depots) != 1:
            raise ValueError(
                f"{self.metadata.relative_path}: expected exactly 1 depot, found {len(depots)}"
            )
        return depots[0]

    def node_by_id(self, string_id) -> Node:
        for node in self.nodes:
            if node.string_id == string_id:
                return node
        raise KeyError(f"{self.metadata.relative_path}: no node with StringID {string_id!r}")
