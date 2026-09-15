"""Named methods for batch evaluation. No method-specific route filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from baselines.greedy import GreedyFullCharge, GreedyMinimumSufficientCharge
from baselines.legacy_ddqn import LegacyTwoStageDDQN
from baselines.lookahead import OneStepLookahead
from rl.ablation import AblationConfig
from rl.checkpoint import load_hybrid_actor

from .actors import HybridPolicyActor


BASELINE_CTORS = {
    "GreedyMinimumSufficientCharge": GreedyMinimumSufficientCharge,
    "GreedyFullCharge": GreedyFullCharge,
    "OneStepLookahead": OneStepLookahead,
}

TRAINED_METHODS = {
    "hybrid_ppo": ("HybridPPO", "FULL"),
    "discrete_ppo": ("DiscretePPO", "A1"),
    "attention_ppo": ("AttentionPPO", "ATTENTION"),
    "legacy_ddqn": ("LegacyTwoStageDDQN", "LEGACY"),
}

ABLATION_METHODS = {name: name for name in ("FULL", "A1", "A2", "A3", "A4", "A5")}


def build_stateless(name: str) -> Callable:
    if name not in BASELINE_CTORS:
        raise KeyError(name)
    return BASELINE_CTORS[name]()


def build_from_checkpoint(method: str, checkpoint: Path) -> Callable:
    if method == "legacy_ddqn" or method == "LegacyTwoStageDDQN":
        import torch

        from baselines.legacy_ddqn import LegacyTwoStageDDQN

        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        agent = LegacyTwoStageDDQN(d_model=int(payload.get("d_model", 32)))
        if "online" in payload:
            agent.online.load_state_dict(payload["online"])
        return agent
    return load_hybrid_actor(checkpoint)


def method_display_name(key: str) -> str:
    table = {
        "hybrid_ppo": "HybridPPO",
        "discrete_ppo": "DiscretePPO",
        "attention_ppo": "AttentionPPO",
        "legacy_ddqn": "LegacyTwoStageDDQN",
        "greedy_min": "GreedyMinimumSufficientCharge",
        "greedy_full": "GreedyFullCharge",
        "lookahead": "OneStepLookahead",
    }
    return table.get(key, key)
