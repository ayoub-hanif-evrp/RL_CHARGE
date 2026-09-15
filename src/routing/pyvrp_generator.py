"""Customer-only pickup-VRPTW via PyVRP.

Charging stations, battery, altitude, and energy are omitted. Terrain variants
of the same Solomon base therefore produce identical integer matrices.
"""

from __future__ import annotations

import hashlib
import platform
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from data.models import EVRPTWGRInstance, Node
from physics.distance import planar_euclidean
from physics.parameters import PhysicsProfile
from physics.travel_time import travel_time

from .config import PyVRPConfig
from .fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute

DATASET_NAME = "EVRPTW-GR"
DATASET_DOI = "10.17632/srfdbp2twv.1"


@dataclass(frozen=True)
class IntegerRoutingMatrices:
    location_ids: Tuple[str, ...]
    distance: np.ndarray
    duration: np.ndarray
    pickup: Tuple[int, ...]
    tw_early: Tuple[int, ...]
    tw_late: Tuple[int, ...]
    service: Tuple[int, ...]
    capacity: int
    distance_scale: int
    demand_scale: int


def scale_int(value: float, scale: int) -> int:
    return int(np.round(value * scale))


def build_customer_matrices(
    instance: EVRPTWGRInstance,
    profile: PhysicsProfile,
    config: PyVRPConfig,
) -> IntegerRoutingMatrices:
    depot = instance.depot
    customers = list(instance.customers)
    locations: List[Node] = [depot, *customers]
    n = len(locations)
    distance = np.zeros((n, n), dtype=np.int64)
    duration = np.zeros((n, n), dtype=np.int64)
    for i, origin in enumerate(locations):
        for j, destination in enumerate(locations):
            if i == j:
                continue
            d = planar_euclidean(origin, destination)
            t = travel_time(d, profile.average_velocity)
            distance[i, j] = scale_int(d.value, config.distance_scale)
            duration[i, j] = scale_int(t.value, config.distance_scale)
    pickups = tuple(scale_int(node.demand, config.demand_scale) for node in customers)
    tw_early = tuple(
        scale_int(node.ready_time, config.distance_scale) for node in locations
    )
    tw_late = tuple(scale_int(node.due_date, config.distance_scale) for node in locations)
    service = tuple(
        scale_int(node.service_time, config.distance_scale) for node in locations
    )
    capacity = scale_int(profile.payload_capacity_kg, config.demand_scale)
    return IntegerRoutingMatrices(
        location_ids=tuple(node.string_id for node in locations),
        distance=distance,
        duration=duration,
        pickup=pickups,
        tw_early=tw_early,
        tw_late=tw_late,
        service=service,
        capacity=capacity,
        distance_scale=config.distance_scale,
        demand_scale=config.demand_scale,
    )


def instance_sha256(instance: EVRPTWGRInstance) -> str:
    return hashlib.sha256(instance.metadata.path.read_bytes()).hexdigest()


def _unscaled(value: int, scale: int) -> float:
    return float(value) / float(scale)


class PyVRPGenerator:
    def __init__(self, config: PyVRPConfig, max_iterations_override: Optional[int] = None):
        self.config = config
        self.max_iterations_override = max_iterations_override
        try:
            import pyvrp  # noqa: F401
        except ImportError as exc:
            raise ImportError("pyvrp is required for route generation") from exc

    @property
    def version(self) -> str:
        import importlib.metadata

        return importlib.metadata.version("pyvrp")

    def generate(
        self, instance: EVRPTWGRInstance, profile: PhysicsProfile
    ) -> List[FrozenRoute]:
        from pyvrp.stop import MaxIterations

        matrices = build_customer_matrices(instance, profile, self.config)
        model = self._build_model(matrices)
        iterations = self.max_iterations_override
        if iterations is None:
            iterations = self.config.iterations_for(instance.metadata.network_group)
        started = time.perf_counter()
        result = model.solve(
            stop=MaxIterations(iterations),
            seed=self.config.seed,
            display=self.config.display,
        )
        runtime = time.perf_counter() - started
        return self._routes_from_result(
            instance, profile, matrices, result, iterations, runtime
        )

    def _build_model(self, matrices: IntegerRoutingMatrices):
        from pyvrp import Model

        model = Model()
        locations = []
        for string_id in matrices.location_ids:
            # Coordinates are unused for travel: we add explicit integer edges.
            locations.append(model.add_location(0, 0, name=string_id))
        depot = model.add_depot(
            locations[0],
            tw_early=int(matrices.tw_early[0]),
            tw_late=int(matrices.tw_late[0]),
            service_duration=int(matrices.service[0]),
            name=matrices.location_ids[0],
        )
        for index, pickup in enumerate(matrices.pickup, start=1):
            model.add_client(
                locations[index],
                pickup=[int(pickup)],
                delivery=[],
                service_duration=int(matrices.service[index]),
                tw_early=int(matrices.tw_early[index]),
                tw_late=int(matrices.tw_late[index]),
                name=matrices.location_ids[index],
            )
        n_clients = len(matrices.pickup)
        if self.config.fleet_policy != "one_vehicle_per_customer":
            raise ValueError(f"unsupported fleet_policy {self.config.fleet_policy!r}")
        model.add_vehicle_type(
            num_available=max(n_clients, 1),
            capacity=[int(matrices.capacity)],
            tw_early=int(matrices.tw_early[0]),
            tw_late=int(matrices.tw_late[0]),
            start_depot=depot,
            end_depot=depot,
        )
        for i, frm in enumerate(locations):
            for j, to in enumerate(locations):
                if i == j:
                    continue
                model.add_edge(
                    frm,
                    to,
                    distance=int(matrices.distance[i, j]),
                    duration=int(matrices.duration[i, j]),
                )
        return model

    def _routes_from_result(
        self,
        instance: EVRPTWGRInstance,
        profile: PhysicsProfile,
        matrices: IntegerRoutingMatrices,
        result,
        iterations: int,
        runtime: float,
    ) -> List[FrozenRoute]:
        solution = result.best
        id_by_index = {i: sid for i, sid in enumerate(matrices.location_ids)}
        depot_id = matrices.location_ids[0]
        assigned_ids: List[str] = []
        built: List[FrozenRoute] = []
        objective = _unscaled(_solution_objective(result, solution), self.config.distance_scale)
        for vehicle_index, route in enumerate(solution.routes()):
            customer_ids = _extract_customers(route, id_by_index, depot_id)
            assigned_ids.extend(customer_ids)
            demand = sum(instance.node_by_id(cid).demand for cid in customer_ids)
            built.append(
                FrozenRoute(
                    route_id=f"{instance.metadata.instance_id}__v{vehicle_index}",
                    source_dataset=DATASET_NAME,
                    doi=DATASET_DOI,
                    raw_instance_id=instance.metadata.instance_id,
                    relative_path=instance.metadata.relative_path,
                    network_group=instance.metadata.network_group.value,
                    terrain_variant=instance.metadata.terrain_variant.value,
                    customer_distribution=instance.metadata.customer_distribution,
                    schedule_type=instance.metadata.schedule_type,
                    generator=self.config.generator,
                    generator_version=self.version,
                    seed=self.config.seed,
                    stop=self.config.stop,
                    n_iterations=iterations,
                    config_hash=self.config.fingerprint(),
                    physics_profile=profile.name,
                    capacity_policy=(
                        "official_3650kg"
                        if not profile.use_instance_payload_capacity
                        else "instance_C"
                    ),
                    distance_scale=self.config.distance_scale,
                    demand_scale=self.config.demand_scale,
                    rounding_policy=self.config.rounding_policy,
                    routing_problem=self.config.routing_problem,
                    python=sys.version.split()[0],
                    os_name=platform.system(),
                    arch=platform.machine(),
                    instance_sha256=instance_sha256(instance),
                    vehicle_index=vehicle_index,
                    depot_id=instance.depot.string_id,
                    customer_ids=tuple(customer_ids),
                    route_demand=float(demand),
                    route_distance=_unscaled(
                        _route_distance(route), self.config.distance_scale
                    ),
                    route_duration_lower_bound=_unscaled(
                        _route_duration(route), self.config.distance_scale
                    ),
                    n_customers=len(customer_ids),
                    routing_feasible=bool(
                        _route_feasible(route, solution)
                        and demand <= profile.payload_capacity_kg + 1e-9
                    ),
                    charging_feasibility_status=CHARGING_FEASIBILITY_UNVERIFIED,
                    generation_runtime_s=float(runtime),
                    routing_objective=objective,
                    unassigned_customer_ids=(),
                    demand_mapping=self.config.demand_mapping,
                    fleet_policy=self.config.fleet_policy,
                )
            )
        required = [node.string_id for node in instance.customers]
        unassigned = tuple(cid for cid in required if cid not in assigned_ids)
        return [
            FrozenRoute.from_dict(
                {**route.to_dict(), "unassigned_customer_ids": list(unassigned)}
            )
            for route in built
        ]


def _extract_customers(route, id_by_index, depot_id: str) -> List[str]:
    customers = []
    for item in _route_visits(route):
        if hasattr(item, "is_client"):
            is_client = item.is_client() if callable(item.is_client) else item.is_client
            if not is_client:
                continue
            # ScheduledActivity.idx is the 0-based client index, not the location index.
            location_index = int(item.idx) + 1
            node_id = id_by_index[location_index]
            if node_id != depot_id:
                customers.append(node_id)
            continue
        if hasattr(item, "location"):
            index = int(item.location)
        else:
            index = int(item)
        node_id = id_by_index.get(index)
        if node_id is None or node_id == depot_id:
            continue
        customers.append(node_id)
    return customers


def _route_visits(route) -> Sequence:
    if hasattr(route, "schedule"):
        schedule = route.schedule()
        if schedule is not None:
            return list(schedule)
    if hasattr(route, "visits"):
        return list(route.visits())
    return list(route)


def _route_distance(route) -> int:
    if hasattr(route, "distance"):
        value = route.distance()
        return int(value[0] if isinstance(value, (list, tuple)) else value)
    return 0


def _route_duration(route) -> int:
    if hasattr(route, "duration"):
        value = route.duration()
        return int(value[0] if isinstance(value, (list, tuple)) else value)
    return 0


def _route_feasible(route, solution) -> bool:
    if hasattr(route, "is_feasible"):
        return bool(route.is_feasible())
    if hasattr(solution, "is_feasible"):
        return bool(solution.is_feasible())
    return True


def _solution_objective(result, solution) -> int:
    if hasattr(result, "cost"):
        try:
            return int(result.cost())
        except TypeError:
            pass
    if hasattr(solution, "distance"):
        value = solution.distance()
        return int(value[0] if isinstance(value, (list, tuple)) else value)
    return 0
