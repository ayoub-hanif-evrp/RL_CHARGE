"""Legacy two-stage Double DQN *method* on the new simulator.

This is not the old environment. Features and encoder are the Part 3A ones.
One optimizer. Target encoder and both target heads are copied and frozen.
SOC levels are the historical six-level grid 0.5 ... 1.0.

Double DQN: the online network selects the next discrete action (and, for a
station, the charge level at that station). The frozen target network evaluates
that exact pair. CONTINUE has no charge-level Q term.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from rl.encoder import HybridEncoder
from rl.features import extract_features
from simulation.simulator import FixedRouteSimulator

from .common import BaselineResult, run_discrete_policy

SOC_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def double_dqn_next_value(
    online_disc: torch.Tensor,
    online_soc: torch.Tensor,
    target_disc: torch.Tensor,
    target_soc: torch.Tensor,
) -> torch.Tensor:
    """Online selects; target evaluates. ``online_soc`` is [B, n_stations, L]."""
    next_a = online_disc.argmax(dim=-1)
    gathered_disc = target_disc.gather(1, next_a.unsqueeze(-1)).squeeze(-1)
    is_station = next_a > 0
    station_idx = (next_a - 1).clamp(min=0)
    batch = torch.arange(online_disc.size(0), device=online_disc.device)
    soc_for_station = online_soc[batch, station_idx]
    next_level = soc_for_station.argmax(dim=-1)
    target_soc_for_station = target_soc[batch, station_idx]
    gathered_soc = target_soc_for_station.gather(1, next_level.unsqueeze(-1)).squeeze(-1)
    return gathered_disc + torch.where(is_station, gathered_soc, torch.zeros_like(gathered_soc))


class TwoHeadQ(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.encoder = HybridEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.station_head = nn.Linear(d_model, 1)
        self.continue_head = nn.Linear(d_model, 1)
        self.soc_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, len(SOC_LEVELS)),
        )

    def q_values(self, features) -> tuple[torch.Tensor, torch.Tensor]:
        from rl.policy import features_to_batch

        batch = features_to_batch(features)
        encoded = self.encoder(batch)
        h = encoded["h"]
        station_embed = encoded["station_embed"]
        continue_q = self.continue_head(h)
        station_q = self.station_head(station_embed).squeeze(-1)
        q_discrete = torch.cat([continue_q, station_q], dim=-1)
        soc_in = torch.cat(
            [h.unsqueeze(1).expand(-1, station_embed.size(1), -1), station_embed], dim=-1
        )
        q_soc = self.soc_head(soc_in)
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
        self.normalizer = None

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
        features = extract_features(
            simulator, getattr(self, "normalizer", None), soc_interval="continuation_to_max"
        )
        with torch.no_grad():
            q_discrete, q_soc = self.online.q_values(features)
        discrete = int(torch.argmax(q_discrete, dim=-1).item())
        if discrete == 0:
            return 0, 0.0
        level = int(torch.argmax(q_soc[0, discrete - 1], dim=-1).item())
        target_soc = self.soc_levels[level]
        from simulation.shield import soc_interval_for_station, station_ids_of

        station_id = station_ids_of(simulator)[discrete - 1]
        interval = soc_interval_for_station(simulator, station_id)
        span = max(interval.soc_upper - interval.soc_lower, 1e-12)
        u = (min(interval.soc_upper, max(interval.soc_lower, target_soc)) - interval.soc_lower) / span
        return discrete, float(u)

    def __call__(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> BaselineResult:
        return run_discrete_policy(simulator, lambda sim: self.choose(sim, eval_mode=eval_mode))


def load_ddqn_agent(path: Path) -> LegacyTwoStageDDQN:
    """Restore online weights and the TRAIN-only normalizer. TEST is never involved."""
    from rl.normalization import Normalizer

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    agent = LegacyTwoStageDDQN(d_model=int(payload.get("d_model", 64)))
    agent.online.load_state_dict(payload["online"])
    agent.sync_target()
    if payload.get("normalizer"):
        agent.normalizer = Normalizer.from_state_dict(payload["normalizer"])
    return agent
