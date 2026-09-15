"""Greedy / lookahead policies on a thin native-FRVCP graph.

Not EVRPTW-GR. Energy and time come from the published matrices; charging
uses the instance piecewise breakpoints. Used only for FRVCP optimality gaps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def _energy(instance: dict, i: int, j: int) -> float:
    return float(instance["energy_matrix"][i][j])


def _time(instance: dict, i: int, j: int) -> float:
    return float(instance["time_matrix"][i][j])


def _process(instance: dict, i: int) -> float:
    times = instance.get("process_times") or [0.0] * len(instance["energy_matrix"])
    return float(times[i])


def _cs_nodes(instance: dict) -> List[int]:
    return [int(row["node_id"]) for row in instance["css"]]


def _charge_time(instance: dict, cs_node: int, q_before: float, q_after: float) -> float:
    cs_type = next(int(row["cs_type"]) for row in instance["css"] if int(row["node_id"]) == cs_node)
    curve = next(bp for bp in instance["breakpoints_by_type"] if int(bp["cs_type"]) == cs_type)
    return _interp(q_after, curve["charge"], curve["time"]) - _interp(q_before, curve["charge"], curve["time"])


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    x = min(max(float(x), float(xs[0])), float(xs[-1]))
    for left, right, y_left, y_right in zip(xs, xs[1:], ys, ys[1:]):
        if x <= right or right == xs[-1]:
            if right == left:
                return float(y_left)
            span = (x - left) / (right - left)
            return float(y_left) + span * (float(y_right) - float(y_left))
    return float(ys[-1])


def greedy_frvcp(
    instance: dict,
    route: Sequence[int],
    q_init: float,
    *,
    full_charge: bool = False,
) -> dict:
    max_q = float(instance["max_q"])
    t_max = float(instance.get("t_max") or 1e9)
    stations = _cs_nodes(instance)
    time = 0.0
    q = float(q_init)
    node = int(route[0])
    visits = 0
    insertions: List[int] = [node]
    for nxt in list(route[1:]):
        nxt = int(nxt)
        need = _energy(instance, node, nxt)
        if q + 1e-9 < need:
            best = None
            for cs in stations:
                e1 = _energy(instance, node, cs)
                e2 = _energy(instance, cs, nxt)
                if q + 1e-9 < e1:
                    continue
                q_arr = q - e1
                target = max_q if full_charge else min(max_q, e2)
                if target < q_arr:
                    target = q_arr
                if q_arr + (target - q_arr) + 1e-9 < e2:
                    continue
                dt = _time(instance, node, cs) + _charge_time(instance, cs, q_arr, target) + _time(instance, cs, nxt)
                if best is None or dt < best[0]:
                    best = (dt, cs, target, e1, e2)
            if best is None:
                return {
                    "feasible": False,
                    "duration": None,
                    "n_station_visits": visits,
                    "route": insertions,
                    "reason": "no_cs_insertion",
                }
            dt, cs, target, e1, e2 = best
            time += _time(instance, node, cs)
            q -= e1
            time += _charge_time(instance, cs, q, target)
            q = target
            visits += 1
            insertions.append(cs)
            time += _time(instance, cs, nxt) + _process(instance, nxt)
            q -= e2
            node = nxt
            insertions.append(nxt)
        else:
            time += _time(instance, node, nxt) + _process(instance, nxt)
            q -= need
            node = nxt
            insertions.append(nxt)
        if time > t_max + 1e-9:
            return {
                "feasible": False,
                "duration": time,
                "n_station_visits": visits,
                "route": insertions,
                "reason": "t_max",
            }
    return {
        "feasible": True,
        "duration": time,
        "n_station_visits": visits,
        "route": insertions,
        "reason": None,
    }
