"""Provenance + semantic tests for frozen V3 station features.

STATION_DIM=12 layout (frozen; do not silently change):
  0 dist current→station
  1 time current→station
  2 energy current→station
  3 dist station→next
  4 time station→next
  5 energy station→next
  6 arrival (FROZEN: copy of interval.soc_lower — duplicate of index 9)
  7 station altitude
  8 detour time
  9 continuation / interval SOC lower
 10 interval SOC upper
 11 slack to next after travel to station

These tests protect provenance. They do NOT endorse the duplication as desirable.
"""

from __future__ import annotations

import inspect

import numpy as np

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.features import STATION_DIM, _station_features, extract_features
from simulation.shield import (
    ARRIVAL_TO_MAX,
    CONTINUATION_TO_MAX,
    _arrival_soc,
    continuation_departure_soc,
    soc_interval_for_station,
)
from simulation.simulator import FixedRouteSimulator


def _sim(write_instance, nodes, customers=("C1",)):
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    return FixedRouteSimulator(
        instance, customers, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )


def test_frozen_station_feature_duplicate_is_documented(write_instance):
    """Provenance: indices 6 and 9 remain identical copies of interval.soc_lower."""
    source = inspect.getsource(_station_features)
    assert "arrival = interval.soc_lower" in source
    assert STATION_DIM == 12

    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 2.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 5.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    feats = extract_features(sim, soc_interval=CONTINUATION_TO_MAX)
    assert feats.stations.shape[-1] == STATION_DIM
    row = feats.stations[feats.station_ids.index("S1")]
    interval = soc_interval_for_station(sim, "S1", mode=CONTINUATION_TO_MAX)
    # extract_features casts stations to float32
    np.testing.assert_allclose(row[6], interval.soc_lower, rtol=0, atol=1e-6)
    np.testing.assert_allclose(row[9], interval.soc_lower, rtol=0, atol=1e-6)
    np.testing.assert_allclose(row[6], row[9], rtol=0, atol=1e-6)


def test_arrival_soc_can_differ_from_continuation_lower(write_instance):
    """Semantic diagnostic: physical arrival SOC ≠ continuation lower bound."""
    # Same geometry as tests/simulation/test_shield.py::test_full_vs_a2_intervals_differ
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 40.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 90.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    arrival = float(_arrival_soc(sim, "S0"))
    bound = continuation_departure_soc(sim, "S0")
    assert bound is not None
    cont_lower = max(arrival, float(bound))
    assert cont_lower > arrival + 1e-6

    interval_cont = soc_interval_for_station(sim, "S0", mode=CONTINUATION_TO_MAX)
    interval_arr = soc_interval_for_station(sim, "S0", mode=ARRIVAL_TO_MAX)
    assert abs(interval_cont.soc_lower - cont_lower) < 1e-9
    assert abs(interval_arr.soc_lower - arrival) < 1e-9

    # Frozen feature vector still duplicates continuation lower into index 6
    # (not the distinct physical arrival SOC).
    row = _station_features(sim, "S0", "C1", soc_interval=CONTINUATION_TO_MAX)
    assert abs(float(row[6]) - float(interval_cont.soc_lower)) < 1e-9
    assert abs(float(row[6]) - arrival) > 1e-6


def test_station_feature_geometry_and_semantics(write_instance):
    """Documented 12-position semantics (geometry / energy / detour / slack)."""
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 100.0),
        node_row("S1", "f", 3.0, 4.0, 0.0, 0.0, 10_000.0, 0.0, 50.0),
        node_row("C1", "c", 6.0, 0.0, 10.0, 5.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    row = _station_features(sim, "S1", "C1", soc_interval=CONTINUATION_TO_MAX)
    assert row.shape == (STATION_DIM,)

    arc_cf = sim.network.arc("D0", "S1")
    arc_fn = sim.network.arc("S1", "C1")
    arc_cn = sim.network.arc("D0", "C1")
    energy_cf = sim.energy_model.energy_for_arc("D0", "S1", sim.state.payload)
    energy_fn = sim.energy_model.energy_for_arc("S1", "C1", sim.state.payload)
    interval = soc_interval_for_station(sim, "S1", mode=CONTINUATION_TO_MAX)
    next_node = sim.network.node("C1")
    detour = arc_cf.travel_time.value + arc_fn.travel_time.value - arc_cn.travel_time.value
    slack = float(next_node.due_date - (sim.state.time.value + arc_cf.travel_time.value))

    np.testing.assert_allclose(row[0], arc_cf.distance.value, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[1], arc_cf.travel_time.value, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[2], energy_cf.net_energy.value, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[3], arc_fn.distance.value, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[4], arc_fn.travel_time.value, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[5], energy_fn.net_energy.value, rtol=0, atol=1e-9)
    # Frozen duplication (documented; not desirable)
    np.testing.assert_allclose(row[6], interval.soc_lower, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[7], 50.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[8], detour, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[9], interval.soc_lower, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[10], interval.soc_upper, rtol=0, atol=1e-9)
    np.testing.assert_allclose(row[11], slack, rtol=0, atol=1e-9)


def test_frozen_beta_softplus_plus_one_provenance():
    """Provenance: Beta uses softplus(+)+1 so alpha,beta > 1 (not a desirability claim)."""
    from rl.policy import HybridPolicy

    source = inspect.getsource(HybridPolicy.forward)
    assert "softplus" in source
    assert "+ 1.0" in source
    act = inspect.getsource(HybridPolicy.act)
    assert "alpha_sel / (alpha_sel + beta_sel)" in act
    assert "1e-4" in act
