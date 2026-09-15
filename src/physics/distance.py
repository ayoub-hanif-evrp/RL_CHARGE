"""Planar Euclidean distance in instance coordinate units.

Distance is horizontal ``hypot(Δx, Δy)``. It is **not** the 3-D slope length
``hypot(d, Δh)``. Energy uses the road angle separately.
"""

from __future__ import annotations

import math

from data.models import Node
from domain.quantities import Distance


def planar_euclidean(from_node: Node, to_node: Node) -> Distance:
    return planar_euclidean_xy(from_node.x, from_node.y, to_node.x, to_node.y)


def planar_euclidean_xy(x1: float, y1: float, x2: float, y2: float) -> Distance:
    return Distance(math.hypot(x2 - x1, y2 - y1))
