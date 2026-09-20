"""Station attention uses the shield legality mask."""

import numpy as np
import torch

from conftest import node_row
from data.parser import parse_instance
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.encoder import HybridEncoder
from rl.features import extract_features
from rl.policy import features_to_batch
from simulation.actions import ChargeAction
from simulation.simulator import FixedRouteSimulator


def _sim(write_instance, nodes, customers=("C1",)):
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    return FixedRouteSimulator(
        instance, customers, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )


def test_station_attention_mask_follows_shield_legality(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S2", "f", 2.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 3.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    before = extract_features(sim)
    assert before.station_mask.shape[0] == len(before.station_ids)
    assert before.discrete_mask.shape[0] == 1 + len(before.station_ids)
    np.testing.assert_array_equal(before.station_mask, before.discrete_mask[1:].astype(np.float32))
    assert sim.step(ChargeAction("S1", 1.0)).feasible
    after = extract_features(sim)
    idx = after.station_ids.index("S1")
    assert after.station_mask[idx] == 0.0
    assert after.discrete_mask[1 + idx] is False or after.discrete_mask[1 + idx] == False
    assert after.station_mask[after.station_ids.index("S2")] == 1.0
    assert not np.array_equal(before.station_mask, after.station_mask)


def test_encoder_is_safe_when_no_station_is_legal(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S0", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 200.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes)
    assert sim.step(ChargeAction("S0", 1.0)).feasible
    feats = extract_features(sim)
    assert np.all(feats.station_mask == 0.0)
    encoder = HybridEncoder(d_model=16, n_heads=4, n_layers=1)
    encoder.eval()
    with torch.no_grad():
        out = encoder(features_to_batch(feats))
    assert torch.isfinite(out["h"]).all()
    assert torch.isfinite(out["station_embed"]).all()


def test_illegal_stations_receive_no_attention_mass(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S1", "f", 1.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("S2", "f", 2.0, 0.0, 0.0, 0.0, 10_000.0, 0.0, 0.0),
        node_row("C1", "c", 3.0, 0.0, 10.0, 0.0, 10_000.0, 1.0, 0.0),
    ]
    sim = _sim(write_instance, nodes, ("C1",))
    assert sim.step(ChargeAction("S1", 1.0)).feasible
    feats = extract_features(sim)
    idx = feats.station_ids.index("S1")
    assert feats.station_mask[idx] == 0.0
    encoder = HybridEncoder(d_model=16, n_heads=4, n_layers=1)
    encoder.eval()
    batch = features_to_batch(feats)
    with torch.no_grad():
        h1 = encoder(batch)["h"]
        batch["stations"] = batch["stations"].clone()
        batch["stations"][:, idx] = 999.0
        h2 = encoder(batch)["h"]
    assert torch.allclose(h1, h2)


def test_fit_normalizer_never_loads_val_or_test():
    import inspect

    from rl import train_loop

    source = inspect.getsource(train_loop.fit_normalizer)
    assert "load_split_routes" not in source
    assert "val_routes" not in source
    assert "test_routes" not in source
    assert train_loop.fit_normalizer.__doc__ is not None
    assert "Never val/test" in train_loop.fit_normalizer.__doc__
