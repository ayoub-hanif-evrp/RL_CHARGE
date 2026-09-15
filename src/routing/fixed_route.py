"""Immutable frozen customer route. Charging visits are method output, not part of this object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


CHARGING_FEASIBILITY_UNVERIFIED = "unverified"


@dataclass(frozen=True)
class FrozenRoute:
    route_id: str
    source_dataset: str
    doi: str
    raw_instance_id: str
    relative_path: str
    network_group: str
    terrain_variant: str
    customer_distribution: str
    schedule_type: int
    generator: str
    generator_version: str
    seed: int
    stop: str
    n_iterations: int
    config_hash: str
    physics_profile: str
    capacity_policy: str
    distance_scale: int
    demand_scale: int
    rounding_policy: str
    routing_problem: str
    python: str
    os_name: str
    arch: str
    instance_sha256: str
    vehicle_index: int
    depot_id: str
    customer_ids: Tuple[str, ...]
    route_demand: float
    route_distance: float
    route_duration_lower_bound: float
    n_customers: int
    routing_feasible: bool
    charging_feasibility_status: str
    generation_runtime_s: float
    routing_objective: float
    unassigned_customer_ids: Tuple[str, ...] = ()
    demand_mapping: str = "pickup"
    fleet_policy: str = "one_vehicle_per_customer"

    def __post_init__(self) -> None:
        object.__setattr__(self, "customer_ids", tuple(self.customer_ids))
        object.__setattr__(self, "unassigned_customer_ids", tuple(self.unassigned_customer_ids))
        if self.n_customers != len(self.customer_ids):
            object.__setattr__(self, "n_customers", len(self.customer_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "capacity_policy": self.capacity_policy,
            "charging_feasibility_status": self.charging_feasibility_status,
            "config_hash": self.config_hash,
            "customer_distribution": self.customer_distribution,
            "customer_ids": list(self.customer_ids),
            "demand_mapping": self.demand_mapping,
            "demand_scale": self.demand_scale,
            "depot_id": self.depot_id,
            "distance_scale": self.distance_scale,
            "doi": self.doi,
            "fleet_policy": self.fleet_policy,
            "generation_runtime_s": self.generation_runtime_s,
            "generator": self.generator,
            "generator_version": self.generator_version,
            "instance_sha256": self.instance_sha256,
            "n_customers": self.n_customers,
            "n_iterations": self.n_iterations,
            "network_group": self.network_group,
            "os_name": self.os_name,
            "physics_profile": self.physics_profile,
            "python": self.python,
            "raw_instance_id": self.raw_instance_id,
            "relative_path": self.relative_path,
            "rounding_policy": self.rounding_policy,
            "route_demand": self.route_demand,
            "route_distance": self.route_distance,
            "route_duration_lower_bound": self.route_duration_lower_bound,
            "route_id": self.route_id,
            "routing_feasible": self.routing_feasible,
            "routing_objective": self.routing_objective,
            "routing_problem": self.routing_problem,
            "schedule_type": self.schedule_type,
            "seed": self.seed,
            "source_dataset": self.source_dataset,
            "stop": self.stop,
            "terrain_variant": self.terrain_variant,
            "unassigned_customer_ids": list(self.unassigned_customer_ids),
            "vehicle_index": self.vehicle_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrozenRoute":
        payload = dict(data)
        payload["customer_ids"] = tuple(payload["customer_ids"])
        payload["unassigned_customer_ids"] = tuple(payload.get("unassigned_customer_ids") or ())
        return cls(**payload)
