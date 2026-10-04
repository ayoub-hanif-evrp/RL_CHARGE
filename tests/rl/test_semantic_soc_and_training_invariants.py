"""Semantic SOC interval / mapping tests and training-data invariants.

Documents correct shield semantics. Frozen feature duplication is covered in
test_frozen_station_features.py and must not silently change.
"""

from __future__ import annotations

import inspect

import numpy as np

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.train_loop import fit_normalizer
from simulation.shield import (
    ARRIVAL_TO_MAX,
    CONTINUATION_TO_MAX,
    _arrival_soc,
    map_u_to_target_soc,
    soc_interval_for_station,
)
from simulation.simulator import FixedRouteSimulator
from simulation.time_envelope import time_soc_cap


def _sim(write_instance, nodes, customers=("C1",)):
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    return FixedRouteSimulator(
        instance, customers, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )


def test_soc_target_mapping_u_grid(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 40.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 90.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    interval = soc_interval_for_station(sim, "S0", mode=CONTINUATION_TO_MAX)
    for u in (0.0, 0.5, 1.0):
        expected = interval.soc_lower + u * (interval.soc_upper - interval.soc_lower)
        got = map_u_to_target_soc(sim, "S0", u, mode=CONTINUATION_TO_MAX)
        # map_u_to_target_soc also clamps to arrival SOC
        arrival = _arrival_soc(sim, "S0")
        assert abs(got - max(expected, arrival)) < 1e-9


def test_arrival_soc_ne_continuation_lower(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 40.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 90.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    arrival = _arrival_soc(sim, "S0")
    cont = soc_interval_for_station(sim, "S0", mode=CONTINUATION_TO_MAX)
    a2 = soc_interval_for_station(sim, "S0", mode=ARRIVAL_TO_MAX)
    assert abs(a2.soc_lower - arrival) < 1e-9
    assert cont.soc_lower > arrival + 1e-6


def test_time_aware_upper_bound_can_tighten(write_instance):
    # Tight due date on next customer can shrink soc_upper when time_aware=True.
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 10.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 20.0, 0.0, 1.0, 0.0, 30.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    sim.time_aware_envelope = True
    loose = soc_interval_for_station(sim, "S0", mode=CONTINUATION_TO_MAX, time_aware=False)
    tight = soc_interval_for_station(sim, "S0", mode=CONTINUATION_TO_MAX, time_aware=True)
    cap = time_soc_cap(sim, "S0")
    if cap.applicable and cap.soc_upper is not None:
        assert tight.soc_upper <= loose.soc_upper + 1e-9
        assert tight.soc_upper <= float(cap.soc_upper) + 1e-9
    else:
        # Still assert API: time-aware call returns a finite interval object.
        assert tight.soc_upper >= tight.soc_lower - 1e-8


def test_fit_normalizer_source_forbids_val_test():
    source = inspect.getsource(fit_normalizer)
    assert "Never val/test" in (fit_normalizer.__doc__ or "")
    assert "val_routes" not in source
    assert "test_routes" not in source


def test_return_scale_must_be_positive_global_constant():
    from dataclasses import replace

    from rl.ppo import HybridPPO, PPOConfig

    cfg = PPOConfig.from_toml()
    agent = HybridPPO(cfg)
    assert abs(agent.scale_reward(-10.0) - (-10.0 / float(cfg.return_scale))) < 1e-12
    agent.config = replace(agent.config, return_scale=10.0)
    assert abs(agent.scale_reward(-10.0) - (-1.0)) < 1e-12
    agent.config = replace(agent.config, return_scale=0.0)
    try:
        agent.scale_reward(-1.0)
        raised = False
    except ValueError:
        raised = True
    assert raised
