"""State features for Hybrid PPO. No KNN, GCN, CNN, or VehicleCNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from simulation.shield import evaluate_shield, soc_interval_for_station
from simulation.simulator import FixedRouteSimulator

from .normalization import Normalizer


GLOBAL_DIM = 8
NEXT_DIM = 13
CUSTOMER_DIM = 11
STATION_DIM = 12


@dataclass
class FeatureBundle:
    global_features: np.ndarray
    next_features: np.ndarray
    remaining: np.ndarray
    remaining_mask: np.ndarray
    stations: np.ndarray
    station_mask: np.ndarray
    discrete_mask: np.ndarray
    station_ids: Tuple[str, ...]

    def as_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "global": self.global_features,
            "next": self.next_features,
            "remaining": self.remaining,
            "stations": self.stations,
        }


def extract_features(
    simulator: FixedRouteSimulator,
    normalizer: Optional[Normalizer] = None,
    *,
    use_remaining_route: bool = True,
    use_terrain_load_features: bool = True,
    soc_interval: str = "continuation_to_max",
) -> FeatureBundle:
    instance = simulator.instance
    state = simulator.state
    horizon = max(simulator.horizon, 1e-9)
    cap = max(simulator.profile.payload_capacity_kg, 1e-9)
    n_customers = max(len(simulator.customer_ids), 1)
    current = simulator.network.node(state.current_node_id)
    next_id = simulator.next_frozen_node_id()
    next_node = simulator.network.node(next_id)
    global_features = np.array(
        [
            state.soc.value,
            state.time.value / horizon,
            state.payload.value / cap,
            state.next_customer_index / n_customers,
            _slack(next_node, state.time.value) / horizon,
            current.altitude,
            len(simulator.remaining_customer_ids()) / n_customers,
            1.0 if not state.completed else 0.0,
        ],
        dtype=np.float64,
    )
    next_features = _node_features(simulator, next_node, include_depot_flag=True)
    remaining_ids = simulator.remaining_customer_ids()
    if remaining_ids:
        remaining = np.stack(
            [
                _customer_features(simulator, instance.node_by_id(cid), order)
                for order, cid in enumerate(remaining_ids)
            ]
        )
        remaining_mask = np.ones((len(remaining_ids),), dtype=np.float64)
    else:
        remaining = np.zeros((1, CUSTOMER_DIM), dtype=np.float64)
        remaining_mask = np.ones((1,), dtype=np.float64)
    if not use_remaining_route:
        remaining = np.zeros((1, CUSTOMER_DIM), dtype=np.float64)
        remaining_mask = np.ones((1,), dtype=np.float64)
    decision = evaluate_shield(simulator)
    station_rows = []
    station_mask = []
    for i, station_id in enumerate(decision.station_ids):
        station_rows.append(
            _station_features(simulator, station_id, next_id, soc_interval=soc_interval)
        )
        station_mask.append(1.0 if decision.mask[1 + i] else 0.0)
    if station_rows:
        stations = np.stack(station_rows)
        station_mask_arr = np.asarray(station_mask, dtype=np.float64)
    else:
        stations = np.zeros((1, STATION_DIM), dtype=np.float64)
        station_mask_arr = np.zeros((1,), dtype=np.float64)
    if normalizer is not None and normalizer.fitted:
        global_features = normalizer.transform("global", global_features)
        next_features = normalizer.transform("next", next_features)
        remaining = normalizer.transform("remaining", remaining)
        stations = normalizer.transform("stations", stations)
    if not use_terrain_load_features:
        global_features = _hide_global_terrain_load(global_features)
        next_features = _hide_next_terrain_load(next_features)
        remaining = _hide_remaining_terrain_load(remaining)
        stations = _hide_station_terrain_load(stations)
    return FeatureBundle(
        global_features=np.asarray(global_features, dtype=np.float32),
        next_features=np.asarray(next_features, dtype=np.float32),
        remaining=np.asarray(remaining, dtype=np.float32),
        remaining_mask=remaining_mask.astype(np.float32),
        stations=np.asarray(stations, dtype=np.float32),
        station_mask=station_mask_arr.astype(np.float32),
        discrete_mask=np.asarray(decision.mask, dtype=np.bool_),
        station_ids=decision.station_ids,
    )


def _node_features(simulator, node, include_depot_flag: bool) -> np.ndarray:
    current = simulator.state.current_node_id
    arc = simulator.network.arc(current, node.string_id)
    energy = simulator.energy_model.energy_for_arc(
        current, node.string_id, simulator.state.payload
    )
    slack = _slack(node, simulator.state.time.value)
    flags = [
        1.0 if node.string_id == simulator.depot_id else 0.0,
        1.0 if node.string_id != simulator.depot_id else 0.0,
    ]
    return np.array(
        [
            node.x - simulator.network.node(current).x,
            node.y - simulator.network.node(current).y,
            arc.distance.value,
            arc.travel_time.value,
            energy.net_energy.value,
            arc.gradient_percent,
            node.ready_time,
            node.due_date,
            node.service_time,
            node.demand,
            slack,
            *flags,
        ],
        dtype=np.float64,
    )


def _customer_features(simulator, node, order: int) -> np.ndarray:
    current = simulator.state.current_node_id
    arc = simulator.network.arc(current, node.string_id)
    energy = simulator.energy_model.energy_for_arc(
        current, node.string_id, simulator.state.payload
    )
    return np.array(
        [
            node.x - simulator.network.node(current).x,
            node.y - simulator.network.node(current).y,
            arc.distance.value,
            arc.travel_time.value,
            energy.net_energy.value,
            node.ready_time,
            node.due_date,
            node.service_time,
            node.demand,
            _slack(node, simulator.state.time.value),
            float(order),
        ],
        dtype=np.float64,
    )


def _station_features(
    simulator, station_id: str, next_id: str, *, soc_interval: str = "continuation_to_max"
) -> np.ndarray:
    current = simulator.state.current_node_id
    station = simulator.network.node(station_id)
    arc_cf = simulator.network.arc(current, station_id)
    arc_fn = simulator.network.arc(station_id, next_id)
    arc_cn = simulator.network.arc(current, next_id)
    energy_cf = simulator.energy_model.energy_for_arc(
        current, station_id, simulator.state.payload
    )
    energy_fn = simulator.energy_model.energy_for_arc(
        station_id, next_id, simulator.state.payload
    )
    interval = soc_interval_for_station(simulator, station_id, mode=soc_interval)
    detour = arc_cf.travel_time.value + arc_fn.travel_time.value - arc_cn.travel_time.value
    arrival = interval.soc_lower
    next_node = simulator.network.node(next_id)
    return np.array(
        [
            arc_cf.distance.value,
            arc_cf.travel_time.value,
            energy_cf.net_energy.value,
            arc_fn.distance.value,
            arc_fn.travel_time.value,
            energy_fn.net_energy.value,
            arrival,
            station.altitude,
            detour,
            interval.soc_lower,
            interval.soc_upper,
            _slack(next_node, simulator.state.time.value + arc_cf.travel_time.value),
        ],
        dtype=np.float64,
    )


def _slack(node, now: float) -> float:
    return float(node.due_date - now)


def _hide_global_terrain_load(vec: np.ndarray) -> np.ndarray:
    out = np.array(vec, copy=True, dtype=np.float64)
    out[..., 2] = 0.0  # payload/capacity
    out[..., 5] = 0.0  # altitude
    return out


def _hide_next_terrain_load(vec: np.ndarray) -> np.ndarray:
    out = np.array(vec, copy=True, dtype=np.float64)
    out[..., 5] = 0.0  # gradient
    out[..., 9] = 0.0  # demand
    return out


def _hide_remaining_terrain_load(vec: np.ndarray) -> np.ndarray:
    out = np.array(vec, copy=True, dtype=np.float64)
    out[..., 8] = 0.0  # demand
    return out


def _hide_station_terrain_load(vec: np.ndarray) -> np.ndarray:
    out = np.array(vec, copy=True, dtype=np.float64)
    out[..., 7] = 0.0  # station altitude
    return out
