"""Hybrid PPO trainer. One optimizer owns the shared encoder. No paper-scale training here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tomllib
import torch
from torch import nn

from data.paths import REPO_ROOT

from .ablation import AblationConfig
from .buffers import RolloutBuffer, Transition
from .env import ShieldedRouteEnv
from .policy import HybridPolicy
from .seed import seed_everything


RL_CONFIG_DIR = REPO_ROOT / "configs" / "rl"


@dataclass
class PPOConfig:
    learning_rate: float
    rollout_steps: int
    minibatch_size: int
    update_epochs: int
    clip_eps: float
    gamma: float
    gae_lambda: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    budget_updates: int
    eval_interval: int
    early_stopping_patience: int
    seed: int
    u_eps: float = 1e-4

    @classmethod
    def from_toml(cls, path: Path | None = None) -> "PPOConfig":
        path = path or (RL_CONFIG_DIR / "hybrid_ppo.toml")
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        return cls(
            learning_rate=float(raw["learning_rate"]),
            rollout_steps=int(raw["rollout_steps"]),
            minibatch_size=int(raw["minibatch_size"]),
            update_epochs=int(raw["update_epochs"]),
            clip_eps=float(raw["clip_eps"]),
            gamma=float(raw["gamma"]),
            gae_lambda=float(raw["gae_lambda"]),
            entropy_coef=float(raw["entropy_coef"]),
            value_coef=float(raw["value_coef"]),
            max_grad_norm=float(raw["max_grad_norm"]),
            d_model=int(raw["d_model"]),
            n_heads=int(raw["n_heads"]),
            n_layers=int(raw["n_layers"]),
            dropout=float(raw["dropout"]),
            budget_updates=int(raw["budget_updates"]),
            eval_interval=int(raw["eval_interval"]),
            early_stopping_patience=int(raw["early_stopping_patience"]),
            seed=int(raw["seed"]),
            u_eps=float(raw.get("u_eps", 1e-4)),
        )


class HybridPPO:
    def __init__(
        self,
        config: PPOConfig,
        device: str = "cpu",
        ablation: Optional[AblationConfig] = None,
    ):
        self.config = config
        self.ablation = ablation or AblationConfig()
        self.device = torch.device(device)
        seed_everything(config.seed)
        self.policy = HybridPolicy(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            ablation=self.ablation,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(config.seed)

    def collect(self, env: ShieldedRouteEnv, n_steps: Optional[int] = None) -> RolloutBuffer:
        steps = n_steps or self.config.rollout_steps
        buffer = RolloutBuffer(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        features = env.reset()
        for _ in range(steps):
            with torch.no_grad():
                output = self.policy.act(features, eval_mode=False)
            discrete = int(output.discrete_index.item())
            u = float(output.u.item())
            info = env.step(discrete, u)
            buffer.add(
                Transition(
                    discrete_index=discrete,
                    u=u,
                    log_prob=float(output.log_prob.item()),
                    value=float(output.value.item()),
                    reward=info.reward,
                    done=info.done,
                    features=features,
                )
            )
            features = info.features
            if info.done:
                features = env.reset()
        buffer.compute_gae()
        return buffer

    def collect_from_factory(self, env_factory, n_steps: Optional[int] = None) -> RolloutBuffer:
        steps = n_steps or self.config.rollout_steps
        buffer = RolloutBuffer(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        env = env_factory()
        features = env.reset()
        for _ in range(steps):
            with torch.no_grad():
                output = self.policy.act(features, eval_mode=False)
            discrete = int(output.discrete_index.item())
            u = float(output.u.item())
            info = env.step(discrete, u)
            buffer.add(
                Transition(
                    discrete_index=discrete,
                    u=u,
                    log_prob=float(output.log_prob.item()),
                    value=float(output.value.item()),
                    reward=info.reward,
                    done=info.done,
                    features=features,
                )
            )
            if info.done:
                env = env_factory()
                features = env.reset()
            else:
                features = info.features
        buffer.compute_gae()
        return buffer

    def update(self, buffer: RolloutBuffer) -> dict:
        cfg = self.config
        advantages = torch.as_tensor(buffer.advantages, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.as_tensor(buffer.returns, device=self.device)
        old_log = torch.as_tensor(
            [t.log_prob for t in buffer.transitions], device=self.device
        )
        losses = []
        for _ in range(cfg.update_epochs):
            for idx in buffer.minibatches(cfg.minibatch_size, self.rng):
                batch_adv = advantages[idx]
                batch_ret = returns[idx]
                batch_old = old_log[idx]
                log_probs = []
                values = []
                entropies = []
                for local, transition_index in enumerate(idx):
                    transition = buffer.transitions[int(transition_index)]
                    features = transition.features
                    net = self.policy.forward(features, device=self.device)
                    dist = torch.distributions.Categorical(logits=net["logits"])
                    discrete = torch.tensor(
                        [transition.discrete_index], device=self.device
                    )
                    log_disc = dist.log_prob(discrete)
                    is_station = transition.discrete_index > 0
                    station_index = max(transition.discrete_index - 1, 0)
                    alpha = net["alpha"][0, station_index]
                    beta = net["beta"][0, station_index]
                    u = torch.tensor(
                        min(1.0 - cfg.u_eps, max(cfg.u_eps, transition.u)),
                        device=self.device,
                    )
                    log_beta = torch.distributions.Beta(alpha, beta).log_prob(u)
                    log_prob = log_disc + (log_beta if is_station else 0.0)
                    log_probs.append(log_prob.reshape([]))
                    values.append(net["value"].reshape([]))
                    entropies.append(dist.entropy().reshape([]))
                log_probs = torch.stack(log_probs)
                values = torch.stack(values)
                entropies = torch.stack(entropies)
                ratio = torch.exp(log_probs - batch_old)
                unclipped = ratio * batch_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * batch_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (batch_ret - values).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropies.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.item()))
        return {"loss": float(np.mean(losses) if losses else 0.0)}

    def smoke_train(self, env: ShieldedRouteEnv, updates: int = 1) -> dict:
        stats = {}
        for _ in range(updates):
            buffer = self.collect(env, n_steps=min(8, self.config.rollout_steps))
            stats = self.update(buffer)
        return stats
