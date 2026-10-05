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
    return_scale: float = 1.0

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
            return_scale=float(raw.get("return_scale", 1.0)),
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
        self.last_raw_episode_returns: list[float] = []

    def scale_reward(self, raw_reward: float) -> float:
        scale = float(self.config.return_scale)
        if scale <= 0.0:
            raise ValueError("return_scale must be a positive global constant")
        return float(raw_reward) / scale

    def _executed_u(self, env: ShieldedRouteEnv, discrete: int, u: float) -> float:
        if discrete == 0:
            return 0.0
        return float(u)

    def _bootstrap_value(self, features) -> float:
        with torch.no_grad():
            net = self.policy.forward(features, device=self.device)
            return float(net["value"].reshape([]).item())

    def _append_step(self, buffer: RolloutBuffer, env: ShieldedRouteEnv, features, n_left: int) -> tuple:
        with torch.no_grad():
            output = self.policy.act(features, eval_mode=False)
        discrete = int(output.discrete_index.item())
        u = self._executed_u(env, discrete, float(output.u.item()))
        with torch.no_grad():
            evaluated = self.policy.evaluate_actions(
                features, discrete, u, u_eps=self.config.u_eps, device=self.device
            )
            log_prob = float(evaluated["log_prob"].item())
            value = float(output.value.item())
        info = env.step(discrete, u)
        if info.done:
            training_ret = float(env.return_value)
            # Legacy alias: historically "raw" meant env.return_value (training return),
            # which under PBRS already includes shaping — keep for compatibility.
            self.last_raw_episode_returns.append(training_ret)
            self.last_training_episode_returns.append(training_ret)
            self.last_objective_episode_returns.append(float(getattr(env, "objective_return_value", training_ret)))
            self.last_normalized_base_episode_returns.append(
                float(getattr(env, "normalized_base_return_value", getattr(env, "base_return_value", training_ret)))
            )
            self.last_shaping_episode_returns.append(float(getattr(env, "shaping_return_value", 0.0)))
            # Legacy aliases
            self.last_base_episode_returns.append(
                float(getattr(env, "normalized_base_return_value", getattr(env, "base_return_value", training_ret)))
            )
            self.last_shaped_episode_returns.append(training_ret)
            self.last_shaping_contributions.append(float(getattr(env, "shaping_return_value", 0.0)))
        buffer.add(
            Transition(
                discrete_index=info.executed_discrete,
                u=info.executed_u,
                log_prob=log_prob,
                value=value,
                reward=self.scale_reward(info.reward),
                done=info.done,
                features=features,
            )
        )
        return info, n_left - 1

    def _reset_episode_trackers(self) -> None:
        self.last_raw_episode_returns = []
        self.last_training_episode_returns = []
        self.last_objective_episode_returns = []
        self.last_normalized_base_episode_returns = []
        self.last_shaping_episode_returns = []
        self.last_base_episode_returns = []
        self.last_shaped_episode_returns = []
        self.last_shaping_contributions = []

    def collect(self, env: ShieldedRouteEnv, n_steps: Optional[int] = None) -> RolloutBuffer:
        steps = n_steps or self.config.rollout_steps
        self._reset_episode_trackers()
        buffer = RolloutBuffer(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        features = env.reset()
        remaining = steps
        while remaining > 0:
            info, remaining = self._append_step(buffer, env, features, remaining)
            features = info.features
            if info.done:
                features = env.reset()
        bootstrap = 0.0
        if buffer.transitions and not buffer.transitions[-1].done:
            bootstrap = self._bootstrap_value(features)
        buffer.compute_gae(bootstrap_value=bootstrap)
        return buffer

    def collect_from_factory(self, env_factory, n_steps: Optional[int] = None) -> RolloutBuffer:
        steps = n_steps or self.config.rollout_steps
        self._reset_episode_trackers()
        buffer = RolloutBuffer(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        env = env_factory()
        features = env.reset()
        remaining = steps
        while remaining > 0:
            info, remaining = self._append_step(buffer, env, features, remaining)
            if info.done:
                env = env_factory()
                features = env.reset()
            else:
                features = info.features
        bootstrap = 0.0
        if buffer.transitions and not buffer.transitions[-1].done:
            bootstrap = self._bootstrap_value(features)
        buffer.compute_gae(bootstrap_value=bootstrap)
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
        policy_losses = []
        value_losses = []
        entropies_out = []
        approx_kls = []
        clip_fractions = []
        grad_norms = []
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
                    evaluated = self.policy.evaluate_actions(
                        transition.features,
                        transition.discrete_index,
                        transition.u,
                        u_eps=cfg.u_eps,
                        device=self.device,
                    )
                    log_probs.append(evaluated["log_prob"].reshape([]))
                    values.append(evaluated["value"].reshape([]))
                    entropies.append(evaluated["entropy"].reshape([]))
                log_probs = torch.stack(log_probs)
                values = torch.stack(values)
                entropies = torch.stack(entropies)
                ratio = torch.exp(log_probs - batch_old)
                unclipped = ratio * batch_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * batch_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (batch_ret - values).pow(2).mean()
                entropy = entropies.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                log_ratio = log_probs - batch_old
                approx_kl = 0.5 * log_ratio.pow(2).mean()
                clip_fraction = (ratio.sub(1.0).abs() > cfg.clip_eps).float().mean()
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies_out.append(float(entropy.item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))
                grad_norms.append(float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm))
        scaled_rewards = [float(step.reward) for step in buffer.transitions]

        def _mean(xs):
            xs = list(xs)
            return float(np.mean(xs) if xs else 0.0)

        training = list(getattr(self, "last_training_episode_returns", self.last_raw_episode_returns))
        objective = list(getattr(self, "last_objective_episode_returns", []))
        normalized = list(getattr(self, "last_normalized_base_episode_returns", []))
        shaping = list(getattr(self, "last_shaping_episode_returns", []))
        return {
            "loss": float(np.mean(losses) if losses else 0.0),
            "policy_loss": float(np.mean(policy_losses) if policy_losses else 0.0),
            "value_loss": float(np.mean(value_losses) if value_losses else 0.0),
            "entropy": float(np.mean(entropies_out) if entropies_out else 0.0),
            "approx_kl": float(np.mean(approx_kls) if approx_kls else 0.0),
            "clip_fraction": float(np.mean(clip_fractions) if clip_fractions else 0.0),
            "grad_norm": float(np.mean(grad_norms) if grad_norms else 0.0),
            "grad_norm_preclip": float(np.mean(grad_norms) if grad_norms else 0.0),
            "return_scale": float(self.config.return_scale),
            "scaled_reward_mean": float(np.mean(scaled_rewards) if scaled_rewards else 0.0),
            # Preferred metrics
            "objective_episode_return_mean": _mean(objective),
            "normalized_base_episode_return_mean": _mean(normalized),
            "shaping_episode_return_mean": _mean(shaping),
            "training_episode_return_mean": _mean(training),
            # Legacy (compat): raw/shaped == training return; base == normalized objective
            "raw_episode_return_mean": _mean(training),
            "base_episode_return_mean": _mean(normalized),
            "shaped_episode_return_mean": _mean(training),
            "shaping_contribution_mean": _mean(shaping),
        }

    def smoke_train(self, env: ShieldedRouteEnv, updates: int = 1) -> dict:
        stats = {}
        for _ in range(updates):
            buffer = self.collect(env, n_steps=min(8, self.config.rollout_steps))
            stats = self.update(buffer)
        return stats
