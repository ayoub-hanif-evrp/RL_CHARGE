"""AttentionPPO: architectural baseline, not a paper reproduction.

Public 2025/2026 EV-routing DRL (eTruckRouting heterogeneous GNN+PPO,
EV-GNN grid charging, HeVRPMD Transformer-POMO, curriculum HetGAT EVRPTW)
solves joint routing+charging or a different MDP. A 1:1 reproduction on
frozen-route EVRPTW-GR is not practical and is not attempted.

This module is Hybrid PPO plus a HetGAT-style node-type embedding
(depot / customer / station) over the existing remaining-route Transformer
and station attention. Label: attention actor-critic architectural baseline
inspired by heterogeneous-graph EVRPTW DRL; not a reproduction of Paper X.
"""

from __future__ import annotations

from .ablation import AblationConfig
from .policy import HybridPolicy


ATTENTION_LABEL = (
    "attention actor-critic architectural baseline inspired by heterogeneous-graph "
    "EVRPTW DRL; not a reproduction of joint-routing papers"
)


def attention_policy(**kwargs) -> HybridPolicy:
    ablation = kwargs.pop("ablation", AblationConfig.from_name("ATTENTION"))
    return HybridPolicy(ablation=ablation, **kwargs)
