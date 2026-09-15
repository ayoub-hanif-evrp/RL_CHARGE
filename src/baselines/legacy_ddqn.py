"""Legacy two-stage DDQN *method* on the new simulator.

This is not the old environment. Features and encoder are the Part 3A ones.
One optimizer. Target encoder and both target heads are copied and frozen.
SOC levels are the historical six-level grid 0.5 ... 1.0.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import torch
from torch import nn

from rl.encoder import HybridEncoder
from rl.features import extract_features
from simulation.simulator import FixedRouteSimulator

from .common import BaselineResult, run_discrete_policy

SOC_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class TwoHeadQ(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.encoder = HybridEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.station_head = nn.Linear(d_model, 1)
        self.continue_head = nn.Linear(d_model, 1)
        self.soc_head = nn.Linear(d_model, len(SOC_LEVELS))

    def q_values(self, features) -> tuple[torch.Tensor, torch.Tensor]:
        from rl.policy import features_to_batch

        batch = features_to_batch(features)
        encoded = self.encoder(batch)
        continue_q = self.continue_head(encoded["h"])
        station_q = self.station_head(encoded["station_embed"]).squeeze(-1)
        q_discrete = torch.cat([continue_q, station_q], dim=-1)
        q_soc = self.soc_head(encoded["h"])
        mask = batch["discrete_mask"]
        if mask.size(-1) != q_discrete.size(-1):
            if mask.size(-1) < q_discrete.size(-1):
                extra = torch.zeros(
                    mask.size(0),
                    q_discrete.size(-1) - mask.size(-1),
                    dtype=torch.bool,
                    device=q_discrete.device,
                )
                mask = torch.cat([mask, extra], dim=-1)
            else:
                mask = mask[:, : q_discrete.size(-1)]
        q_discrete = q_discrete.masked_fill(~mask, -1e9)
        return q_discrete, q_soc


class LegacyTwoStageDDQN:
    name = "LegacyTwoStageDDQN"

    def __init__(self, d_model: int = 64, lr: float = 1e-3, gamma: float = 1.0):
        self.online = TwoHeadQ(d_model=d_model)
        self.target = deepcopy(self.online)
        self._freeze_target()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = gamma
        self.soc_levels = SOC_LEVELS

    def _freeze_target(self) -> None:
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad = False

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self._freeze_target()

    def target_in_optimizer(self) -> bool:
        opt_ids = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        return any(id(p) in opt_ids for p in self.target.parameters())

    def choose(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> tuple[int, float]:
        features = extract_features(simulator)
        with torch.no_grad():
            q_discrete, q_soc = self.online.q_values(features)
        discrete = int(torch.argmax(q_discrete, dim=-1).item())
        if discrete == 0:
            return 0, 0.0
        level = int(torch.argmax(q_soc, dim=-1).item())
        target_soc = self.soc_levels[level]
        from simulation.shield import soc_interval_for_station, station_ids_of

        station_id = station_ids_of(simulator)[discrete - 1]
        interval = soc_interval_for_station(simulator, station_id)
        span = max(interval.soc_upper - interval.soc_lower, 1e-12)
        u = (min(interval.soc_upper, max(interval.soc_lower, target_soc)) - interval.soc_lower) / span
        return discrete, float(u)

    def __call__(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> BaselineResult:
        return run_discrete_policy(simulator, lambda sim: self.choose(sim, eval_mode=eval_mode))
