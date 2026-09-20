"""Double DQN: online selects, target evaluates; station-conditioned charge head."""

import inspect
import math
import tomllib

import pytest
import torch

from baselines.ddqn_train import train_ddqn
from baselines.legacy_ddqn import LegacyTwoStageDDQN, double_dqn_next_value
from data.paths import REPO_ROOT
from experiments.dataset import load_split_payload, load_split_routes
from rl.train_loop import fit_normalizer as real_fit_normalizer


def test_ddqn_target_frozen_and_not_in_optimizer():
    agent = LegacyTwoStageDDQN(d_model=16)
    agent.sync_target()
    for parameter in agent.target.parameters():
        assert parameter.requires_grad is False
    assert agent.target_in_optimizer() is False
    online_ids = {id(p) for p in agent.online.parameters()}
    opt_ids = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
    assert opt_ids == online_ids


def test_double_dqn_uses_online_selection_and_target_evaluation():
    # Actions: CONTINUE, S0, S1. Charge levels: 6.
    online_disc = torch.tensor([[0.1, 5.0, 1.0]])
    target_disc = torch.tensor([[10.0, 0.5, 100.0]])
    online_soc = torch.zeros(1, 2, 6)
    online_soc[0, 0, 5] = 9.0
    online_soc[0, 1, 0] = 3.0
    target_soc = torch.zeros(1, 2, 6)
    target_soc[0, 0, 5] = 1.5
    target_soc[0, 1, :] = 50.0
    value = double_dqn_next_value(online_disc, online_soc, target_disc, target_soc)
    # online picks S0 (index 1); online charge level 5; target Q_disc[S0]+Q_soc[S0,5]
    assert float(value.item()) == pytest.approx(0.5 + 1.5)
    greedy_target = float(target_disc.max().item() + target_soc.max().item())
    assert float(value.item()) != pytest.approx(greedy_target)


def test_double_dqn_continue_has_no_charge_term():
    online_disc = torch.tensor([[9.0, 1.0, 1.0]])
    target_disc = torch.tensor([[2.25, 8.0, 8.0]])
    online_soc = torch.ones(1, 2, 6)
    target_soc = torch.full((1, 2, 6), 99.0)
    value = double_dqn_next_value(online_disc, online_soc, target_disc, target_soc)
    assert float(value.item()) == pytest.approx(2.25)


def test_soc_head_is_station_conditioned():
    agent = LegacyTwoStageDDQN(d_model=16)
    n_stations = next(p for n, p in agent.online.named_parameters() if "soc_head.0.weight" in n)
    assert n_stations.shape[1] == 2 * 16


def test_ddqn_paper_budget_matches_ppo_transitions():
    with (REPO_ROOT / "configs" / "rl" / "legacy_ddqn.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    assert int(raw["gradient_steps"]) == 400 * 256
    assert int(raw["val_interval"]) == 10 * 256
    assert int(raw["early_stopping_patience"]) == 20


def test_ddqn_selection_and_normalizer_are_parent_balanced_train_only():
    source = inspect.getsource(train_ddqn)
    assert "fit_normalizer(train_routes" in source
    assert "parent_balanced_feasibility" in source
    assert "parent_balanced_completion_all" in source
    assert "split_used_for_learning=\"train\"" in source or "split_used_for_learning" in source
    assert 'val["parent_balanced_feasibility"]' in source
    assert "load_split_routes(\"test\"" not in source
    assert "load_split_routes('test'" not in source


def test_ddqn_tiny_train_never_fits_val_or_test(tmp_path, monkeypatch):
    seen = {}

    def wrapped(routes, ablation, **kwargs):
        seen["ids"] = [route.route_id for route in routes]
        seen["parents"] = [route.raw_instance_id for route in routes]
        return real_fit_normalizer(routes, ablation, **kwargs)

    monkeypatch.setattr("baselines.ddqn_train.fit_normalizer", wrapped)
    train_routes = load_split_routes("train", max_customers=5)[:1]
    val_routes = load_split_routes("validation", max_customers=5)[:1]
    assert train_routes and val_routes
    test_parents = set(load_split_payload("test")["instance_ids"])
    manifest = train_ddqn(
        train_routes=train_routes,
        val_routes=val_routes,
        config_seed=0,
        gradient_steps=4,
        batch_size=2,
        target_sync=2,
        replay_size=16,
        d_model=16,
        out_dir=tmp_path,
        val_interval=2,
        early_stopping_patience=5,
    )
    assert seen["ids"] == [train_routes[0].route_id]
    assert val_routes[0].route_id not in seen["ids"]
    assert not set(seen["parents"]) & test_parents
    prov = manifest["normalizer_provenance"]
    assert prov["split"] == "train"
    assert prov["n_routes"] == 1
    assert manifest["checkpoint_selection"] == "parent_balanced_lexicographic"
    assert "best_val_feasibility" in manifest
    assert "best_val_route_weighted_feasibility" in manifest
    if manifest.get("last_loss") is not None:
        assert math.isfinite(float(manifest["last_loss"]))
