"""Double DQN: online selects, target evaluates; station-conditioned charge head."""

import pytest
import torch

from baselines.legacy_ddqn import LegacyTwoStageDDQN, double_dqn_next_value


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
