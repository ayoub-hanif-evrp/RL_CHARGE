"""Immutable frozen customer route. Charging visits are method output, not part of this object."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional, Tuple


CHARGING_FEASIBILITY_UNVERIFIED = "unverified"
FLEET_POLICY_UNRESTRICTED = "unrestricted_fleet_with_fixed_cost"


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
    fleet_policy: str = FLEET_POLICY_UNRESTRICTED
    base_instance: str = ""
    customer_folder: str = ""
    n_vehicles: int = 1
    total_distance: float = 0.0
    fixed_vehicle_cost: int = 0
    lexicographic_objective: float = 0.0
    route_source_instance_id: str = ""
    terrain_reuse: bool = False
    routing_tw_feasible: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "customer_ids", tuple(self.customer_ids))
        object.__setattr__(self, "unassigned_customer_ids", tuple(self.unassigned_customer_ids))
        if self.n_customers != len(self.customer_ids):
            object.__setattr__(self, "n_customers", len(self.customer_ids))

    def to_dict(self) -> dict[str, Any]:
        payload = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"customer_ids", "unassigned_customer_ids"}:
                payload[field.name] = list(value)
            else:
                payload[field.name] = value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrozenRoute":
        names = {field.name for field in fields(cls)}
        payload = {key: value for key, value in data.items() if key in names}
        payload["customer_ids"] = tuple(payload.get("customer_ids") or ())
        payload["unassigned_customer_ids"] = tuple(payload.get("unassigned_customer_ids") or ())
        return cls(**payload)
