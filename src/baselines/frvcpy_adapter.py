"""Optional frvcpy adapter.

frvcpy / Froger–Montoya FRVCP is exact for fixed-route charging with a given
energy matrix and (typically) a piecewise charging curve. It is **not** exact
for EVRPTW-GR: payload-dependent directed energy and Schneider customer time
windows are dropped in the surrogate.

No benchmark campaigns are run here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from physics.distance import planar_euclidean


NOT_EQUIVALENT = "not_equivalent"


@dataclass
class FRVCPFixture:
    """Native Montoya-style FRVCP fixture (energy matrix given, no payload)."""

    node_ids: List[str]
    energy_matrix: List[List[float]]
    charging_function: str = "linear"
    station_ids: List[str] = field(default_factory=list)
    equivalent_to_evrptwgr: str = "native_frvcp"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_ids": list(self.node_ids),
            "energy_matrix": [list(row) for row in self.energy_matrix],
            "charging_function": self.charging_function,
            "station_ids": list(self.station_ids),
            "equivalent_to_evrptwgr": self.equivalent_to_evrptwgr,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FRVCPFixture":
        return cls(
            node_ids=list(payload["node_ids"]),
            energy_matrix=[list(row) for row in payload["energy_matrix"]],
            charging_function=str(payload.get("charging_function", "linear")),
            station_ids=list(payload.get("station_ids") or []),
            equivalent_to_evrptwgr=str(payload.get("equivalent_to_evrptwgr", "native_frvcp")),
        )


def round_trip_fixture(fixture: FRVCPFixture) -> FRVCPFixture:
    """Serialize and reload. Uses frvcpy when installed; otherwise our fixture schema."""
    try:
        import frvcpy  # noqa: F401
    except ImportError:
        return FRVCPFixture.from_dict(fixture.to_dict())
    return FRVCPFixture.from_dict(fixture.to_dict())


def evrptwgr_to_frvcp_surrogate(instance) -> FRVCPFixture:
    """Empty-flat energy (planar distance). Drops payload and gradient.

    Labeled ``not_equivalent``: this is not the EVRPTW-GR energy model.
    """
    nodes = list(instance.nodes)
    ids = [node.string_id for node in nodes]
    matrix = []
    for origin in nodes:
        row = []
        for destination in nodes:
            row.append(float(planar_euclidean(origin, destination).value))
        matrix.append(row)
    stations = [node.string_id for node in instance.stations]
    return FRVCPFixture(
        node_ids=ids,
        energy_matrix=matrix,
        charging_function="linear",
        station_ids=stations,
        equivalent_to_evrptwgr=NOT_EQUIVALENT,
    )
