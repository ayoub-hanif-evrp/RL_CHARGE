"""Replay an official EVRPTW-GR tour under charge-to-max and record the trace."""

from __future__ import annotations

from typing import Optional

from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from simulation.actions import ChargeAction, ContinueAction

from exact.charging_certificate import replay_action_trace


def index_maps(instance) -> tuple[dict, int, int]:
    customers = list(instance.customers)
    stations = list(instance.stations)
    nc = len(customers)
    ns = len(stations)
    dummy = nc + ns + 1
    index_to_id = {0: instance.depot.string_id, dummy: instance.depot.string_id}
    for offset, node in enumerate(customers, start=1):
        index_to_id[offset] = node.string_id
    for offset, node in enumerate(stations, start=1):
        index_to_id[nc + offset] = node.string_id
    return index_to_id, nc, dummy


def charge_to_max_actions(instance, tour: list[int]) -> Optional[tuple]:
    index_to_id, nc, dummy = index_maps(instance)
    customers = tuple(index_to_id[i] for i in tour if 1 <= i <= nc)
    if not customers:
        return None
    actions = []
    profile = PhysicsProfile.from_instance(instance)
    from simulation.simulator import FixedRouteSimulator

    sim = FixedRouteSimulator(
        instance, customers, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    for idx in tour:
        if idx in (0, dummy) or sim.state.completed:
            continue
        node_id = index_to_id.get(idx)
        if node_id is None:
            return None
        if 1 <= idx <= nc:
            actions.append(("CONTINUE",))
            result = sim.step(ContinueAction())
        else:
            actions.append(("CHARGE", node_id, 1.0))
            result = sim.step(ChargeAction(node_id, 1.0))
        if not result.feasible:
            return None
    if not sim.state.completed:
        actions.append(("CONTINUE",))
        result = sim.step(ContinueAction())
        if not result.feasible or not sim.state.completed:
            return None
    return tuple(actions)


def certified_charge_to_max_trace(instance, tour: list[int]):
    customers_actions = charge_to_max_actions(instance, tour)
    if customers_actions is None:
        return None
    index_to_id, nc, _dummy = index_maps(instance)
    customers = tuple(index_to_id[i] for i in tour if 1 <= i <= nc)
    trace = replay_action_trace(instance, customers, customers_actions)
    if trace is None:
        raise AssertionError("official charge-to-max replay did not complete on a fresh simulator")
    return customers, customers_actions, trace
