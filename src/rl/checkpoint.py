"""Checkpoint I/O for Hybrid PPO and related policies."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import torch

from .ablation import AblationConfig
from .normalization import Normalizer
from .policy import HybridPolicy
from .ppo import PPOConfig


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    policy: HybridPolicy,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    normalizer: Optional[Normalizer],
    ablation: AblationConfig,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.__dict__,
        "ablation": ablation.__dict__,
        "normalizer": None if normalizer is None else normalizer.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return sha256_file(path)


def load_checkpoint(
    path: Path,
    policy: HybridPolicy,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def load_hybrid_actor(path: Path):
    from experiments.actors import HybridPolicyActor

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg_raw = dict(payload["config"])
    config = PPOConfig(**cfg_raw)
    ablation = AblationConfig(**payload["ablation"])
    policy = HybridPolicy(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        ablation=ablation,
    )
    policy.load_state_dict(payload["policy"])
    policy.eval()
    normalizer = None
    if payload.get("normalizer"):
        normalizer = Normalizer.from_state_dict(payload["normalizer"])
    return HybridPolicyActor(
        policy,
        ablation=ablation,
        snap_discrete_u=ablation.discrete_u,
        normalizer=normalizer,
    )
