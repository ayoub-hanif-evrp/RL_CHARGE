"""Optimistic time-window cap on charging delay at one station.

The bound is a necessary condition only. It does not choose the action and
does not claim that every SOC inside the interval completes the route.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulation.simulator import FixedRouteSimulator


@dataclass(frozen=True)
class TimeSocCap:
    """``soc_upper is None`` means even zero extra charging delay misses a due date."""

    applicable: bool
    soc_upper: float | None
    max_delay: float


def _arrival_clock_and_soc(simulator: FixedRouteSimulator, station_id: str) -> tuple[float, float] | None:
    if simulator.state.current_node_id == station_id:
        return float(simulator.state.time.value), float(simulator.state.soc.value)
    battery = simulator.battery_model.state_from_soc(simulator.state.soc.value)
    check = simulator.feasibility.can_reach(
        simulator.state.current_node_id,
        station_id,
        simulator.state.payload,
        battery,
    )
    if not check.battery_feasible:
        return None
    applied = simulator.battery_model.apply_arc_energy(battery, check.energy.net_energy)
    travel = simulator.network.arc(simulator.state.current_node_id, station_id).travel_time.value
    return float(simulator.state.time.value) + float(travel), float(applied.after.soc.value)


def max_station_delay(simulator: FixedRouteSimulator, station_id: str, t_arrive: float) -> float:
    """Latest extra delay at ``station_id`` before an optimistic mandatory schedule breaks.

    The schedule after departure uses direct travel along the remaining frozen
    customers and the depot, allows waiting for ready times, and ignores future
    charging and future station detours. Returns a negative value when no
    non-negative delay works.
    """
    network = simulator.network
    remaining = list(simulator.remaining_customer_ids())
    sequence = remaining + [simulator.depot_id]
    latest_arrival = [0.0] * len(sequence)
    depot = network.node(simulator.depot_id)
    latest_arrival[-1] = float(depot.due_date)
    for idx in range(len(sequence) - 2, -1, -1):
        node = network.node(sequence[idx])
        travel = float(network.arc(sequence[idx], sequence[idx + 1]).travel_time.value)
        latest_depart = latest_arrival[idx + 1] - travel
        latest_service_start = min(float(node.due_date), latest_depart - float(node.service_time))
        if latest_service_start + 1e-9 < float(node.ready_time):
            return float("-inf")
        latest_arrival[idx] = latest_service_start
    first = sequence[0]
    travel_out = float(network.arc(station_id, first).travel_time.value)
    latest_depart_station = latest_arrival[0] - travel_out
    return float(latest_depart_station - t_arrive)


def time_soc_cap(simulator: FixedRouteSimulator, station_id: str) -> TimeSocCap:
    """Maximum departure SOC at ``station_id`` allowed by the optimistic slack."""
    arrived = _arrival_clock_and_soc(simulator, station_id)
    if arrived is None:
        return TimeSocCap(applicable=False, soc_upper=None, max_delay=float("-inf"))
    t_arrive, arrival_soc = arrived
    delay = max_station_delay(simulator, station_id, t_arrive)
    battery = simulator.battery_model.state_from_soc(arrival_soc)
    max_soc = float(battery.max_soc_fraction)
    if delay < -1e-9:
        return TimeSocCap(applicable=True, soc_upper=None, max_delay=delay)
    g = float(simulator.profile.inverse_refueling_rate)
    if g <= 0.0:
        upper = max_soc
    else:
        max_extra_energy = max(0.0, delay) / g
        upper = min(max_soc, (float(battery.energy.value) + max_extra_energy) / float(battery.capacity.value))
    upper = min(max_soc, float(upper))
    if upper + 1e-9 < arrival_soc:
        return TimeSocCap(applicable=True, soc_upper=None, max_delay=delay)
    return TimeSocCap(applicable=True, soc_upper=upper, max_delay=delay)
