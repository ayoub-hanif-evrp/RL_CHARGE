"""On-policy rollout storage for Hybrid PPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class Transition:
    discrete_index: int
    u: float
    log_prob: float
    value: float
    reward: float
    done: bool
    features: dict


@dataclass
class RolloutBuffer:
    gamma: float = 1.0
    gae_lambda: float = 0.95
    transitions: List[Transition] = field(default_factory=list)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def compute_gae(self) -> None:
        advantage = 0.0
        next_value = 0.0
        advantages = []
        returns = []
        for transition in reversed(self.transitions):
            delta = (
                transition.reward
                + self.gamma * next_value * (0.0 if transition.done else 1.0)
                - transition.value
            )
            advantage = (
                delta
                + self.gamma
                * self.gae_lambda
                * (0.0 if transition.done else 1.0)
                * advantage
            )
            advantages.append(advantage)
            returns.append(advantage + transition.value)
            next_value = 0.0 if transition.done else transition.value
        self.advantages = np.asarray(list(reversed(advantages)), dtype=np.float32)
        self.returns = np.asarray(list(reversed(returns)), dtype=np.float32)

    def minibatches(self, batch_size: int, rng: np.random.Generator):
        n = len(self.transitions)
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            yield order[start : start + batch_size]
