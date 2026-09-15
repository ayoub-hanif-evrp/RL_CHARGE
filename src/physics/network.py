"""Directed arc table with cached static geometry and travel time.

Energy is not cached here because it depends on payload. Distance is
symmetric; energy is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from data.models import EVRPTWGRInstance, Node
from domain.quantities import Altitude, Distance, TravelTime

from .distance import planar_euclidean
from .travel_time import travel_time as compute_travel_time


@dataclass(frozen=True)
class DirectedArc:
    from_id: str
    to_id: str
    distance: Distance
    delta_altitude: Altitude
    angle_rad: float
    angle_deg: float
    gradient_percent: float
    travel_time: TravelTime

    @property
    def is_zero_length(self) -> bool:
        return self.distance.value == 0.0


class DirectedArcNetwork:
    """All ordered pairs of instance nodes."""

    def __init__(self, instance: EVRPTWGRInstance, average_velocity: float):
        self.instance = instance
        self.average_velocity = float(average_velocity)
        self._nodes: Dict[str, Node] = {node.string_id: node for node in instance.nodes}
        if len(self._nodes) != len(instance.nodes):
            raise ValueError(f"{instance.metadata.relative_path}: duplicate StringID")
        self._arcs: Dict[Tuple[str, str], DirectedArc] = {}
        ids = [node.string_id for node in instance.nodes]
        for origin in instance.nodes:
            for destination in instance.nodes:
                self._arcs[(origin.string_id, destination.string_id)] = _build_arc(
                    origin, destination, self.average_velocity
                )
        self.node_ids: Tuple[str, ...] = tuple(ids)

    def node(self, string_id: str) -> Node:
        try:
            return self._nodes[string_id]
        except KeyError as exc:
            raise KeyError(
                f"{self.instance.metadata.relative_path}: unknown node {string_id!r}"
            ) from exc

    def arc(self, from_id: str, to_id: str) -> DirectedArc:
        try:
            return self._arcs[(from_id, to_id)]
        except KeyError as exc:
            raise KeyError(
                f"{self.instance.metadata.relative_path}: no arc {from_id!r} -> {to_id!r}"
            ) from exc

    def arcs(self) -> Iterable[DirectedArc]:
        return self._arcs.values()


def _build_arc(origin: Node, destination: Node, average_velocity: float) -> DirectedArc:
    distance = planar_euclidean(origin, destination)
    delta_h = destination.altitude - origin.altitude
    if distance.value == 0.0:
        angle_rad = 0.0
        angle_deg = 0.0
        gradient_percent = 0.0
    else:
        # Official Model 2: degrees(arctan(Δh / d_planar)), then sin/cos of that.
        angle_rad = math.atan(delta_h / distance.value)
        angle_deg = math.degrees(angle_rad)
        gradient_percent = 100.0 * delta_h / distance.value
    return DirectedArc(
        from_id=origin.string_id,
        to_id=destination.string_id,
        distance=distance,
        delta_altitude=Altitude(delta_h),
        angle_rad=angle_rad,
        angle_deg=angle_deg,
        gradient_percent=gradient_percent,
        travel_time=compute_travel_time(distance, average_velocity),
    )
