"""Hierarchical episode sampling: parent -> canonical route -> terrain.

Declared in configs/experiments/sampling.toml. Avoids overweighting
Level / Nearly Level / Very Gentle siblings of the same frozen sequence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from routing.fixed_route import FrozenRoute


@dataclass(frozen=True)
class SampledEpisode:
    route: FrozenRoute
    parent: str
    vehicle_index: int
    terrain: str


class HierarchicalSampler:
    """Sample uniformly over train parents, then vehicle index, then terrain."""

    name = "parent_then_route_then_terrain"

    def __init__(self, routes: Sequence[FrozenRoute], seed: int = 42):
        self.rng = np.random.default_rng(int(seed))
        by_parent: Dict[str, List[FrozenRoute]] = defaultdict(list)
        for route in routes:
            by_parent[route.base_instance].append(route)
        self.parents: Tuple[str, ...] = tuple(sorted(by_parent))
        if not self.parents:
            raise ValueError("HierarchicalSampler requires at least one route")
        self._vehicles: Dict[str, Tuple[int, ...]] = {}
        self._by_parent_vehicle_terrain: Dict[Tuple[str, int, str], FrozenRoute] = {}
        self._terrains: Dict[Tuple[str, int], Tuple[str, ...]] = {}
        for parent, group in by_parent.items():
            vehicles = sorted({route.vehicle_index for route in group})
            self._vehicles[parent] = tuple(vehicles)
            for route in group:
                key = (parent, route.vehicle_index, route.terrain_variant)
                self._by_parent_vehicle_terrain[key] = route
                tkey = (parent, route.vehicle_index)
                terrains = set(self._terrains.get(tkey, ()))
                terrains.add(route.terrain_variant)
                self._terrains[tkey] = tuple(sorted(terrains))

    def sample(self) -> SampledEpisode:
        parent = str(self.rng.choice(self.parents))
        vehicles = self._vehicles[parent]
        vehicle = int(self.rng.choice(vehicles))
        terrains = self._terrains[(parent, vehicle)]
        terrain = str(self.rng.choice(terrains))
        route = self._by_parent_vehicle_terrain[(parent, vehicle, terrain)]
        return SampledEpisode(
            route=route, parent=parent, vehicle_index=vehicle, terrain=terrain
        )
