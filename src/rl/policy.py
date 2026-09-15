"""Hybrid policy: masked discrete CONTINUE/stations plus Beta SOC when charging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.distributions import Beta

from .ablation import AblationConfig
from .encoder import HybridEncoder
from .features import FeatureBundle


@dataclass
class PolicyOutput:
    discrete_index: torch.Tensor
    u: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    logits: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


class HybridPolicy(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.0,
        ablation: Optional[AblationConfig] = None,
    ):
        super().__init__()
        self.ablation = ablation or AblationConfig()
        self.encoder = HybridEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            station_encoder=self.ablation.station_encoder,
            node_type_embedding=self.ablation.node_type_embedding,
        )
        self.discrete_continue = nn.Linear(d_model, 1)
        self.discrete_station = nn.Linear(d_model, 1)
        self.beta_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2),
        )
        self.value_head = nn.Linear(d_model, 1)

    def encode_bundle(self, features: FeatureBundle, device=None) -> dict:
        batch = features_to_batch(features, device=device)
        encoded = self.encoder(batch)
        encoded["discrete_mask"] = batch["discrete_mask"]
        encoded["station_mask"] = batch["station_mask"]
        return encoded

    def forward(self, features: FeatureBundle, device=None) -> dict:
        encoded = self.encode_bundle(features, device=device)
        h = encoded["h"]
        station_embed = encoded["station_embed"]
        continue_logit = self.discrete_continue(h)
        station_logits = self.discrete_station(station_embed).squeeze(-1)
        n_stations = station_logits.size(-1)
        logits = torch.cat([continue_logit, station_logits], dim=-1)
        mask = encoded["discrete_mask"]
        if mask.size(-1) != logits.size(-1):
            # Pad or trim if the instance has no stations (placeholder row).
            if mask.size(-1) < logits.size(-1):
                extra = torch.zeros(
                    mask.size(0),
                    logits.size(-1) - mask.size(-1),
                    dtype=torch.bool,
                    device=logits.device,
                )
                mask = torch.cat([mask, extra], dim=-1)
            else:
                mask = mask[:, : logits.size(-1)]
        logits = logits.masked_fill(~mask, -1e9)
        beta_in = torch.cat(
            [h.unsqueeze(1).expand(-1, n_stations, -1), station_embed], dim=-1
        )
        beta_params = self.beta_head(beta_in)
        alpha = torch.nn.functional.softplus(beta_params[..., 0]) + 1.0
        beta = torch.nn.functional.softplus(beta_params[..., 1]) + 1.0
        value = self.value_head(h).squeeze(-1)
        return {
            "logits": logits,
            "mask": mask,
            "alpha": alpha,
            "beta": beta,
            "value": value,
            "h": h,
            "station_embed": station_embed,
        }

    def act(
        self,
        features: FeatureBundle,
        *,
        eval_mode: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> PolicyOutput:
        net = self.forward(features)
        logits = net["logits"]
        mask = net["mask"]
        if eval_mode:
            discrete = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            discrete = dist.sample()
        log_disc = torch.nn.functional.log_softmax(logits, dim=-1)
        log_pi_disc = log_disc.gather(1, discrete.unsqueeze(-1)).squeeze(-1)
        entropy_disc = -(log_disc.exp() * log_disc).sum(dim=-1)
        is_station = discrete > 0
        station_index = (discrete - 1).clamp(min=0)
        alpha_sel = net["alpha"].gather(1, station_index.unsqueeze(-1)).squeeze(-1)
        beta_sel = net["beta"].gather(1, station_index.unsqueeze(-1)).squeeze(-1)
        beta_dist = Beta(alpha_sel, beta_sel)
        if eval_mode:
            u = alpha_sel / (alpha_sel + beta_sel)
        else:
            u = beta_dist.rsample()
        u = u.clamp(1e-4, 1.0 - 1e-4)
        log_beta = beta_dist.log_prob(u)
        log_prob = log_pi_disc + torch.where(
            is_station, log_beta, torch.zeros_like(log_beta)
        )
        entropy = entropy_disc + torch.where(
            is_station, beta_dist.entropy(), torch.zeros_like(log_beta)
        )
        return PolicyOutput(
            discrete_index=discrete,
            u=u,
            log_prob=log_prob,
            entropy=entropy,
            value=net["value"],
            logits=logits,
            alpha=alpha_sel,
            beta=beta_sel,
        )


def features_to_batch(features: FeatureBundle, device=None) -> dict:
    def _t(array):
        tensor = torch.as_tensor(array)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 2 and array.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if features.remaining.ndim == 2 and array is features.remaining:
            tensor = torch.as_tensor(array).unsqueeze(0)
        if features.stations.ndim == 2 and array is features.stations:
            tensor = torch.as_tensor(array).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor.float()

    remaining = torch.as_tensor(features.remaining).float()
    if remaining.ndim == 2:
        remaining = remaining.unsqueeze(0)
    stations = torch.as_tensor(features.stations).float()
    if stations.ndim == 2:
        stations = stations.unsqueeze(0)
    remaining_mask = torch.as_tensor(features.remaining_mask).float()
    if remaining_mask.ndim == 1:
        remaining_mask = remaining_mask.unsqueeze(0)
    station_mask = torch.as_tensor(features.station_mask).float()
    if station_mask.ndim == 1:
        station_mask = station_mask.unsqueeze(0)
    global_f = torch.as_tensor(features.global_features).float()
    if global_f.ndim == 1:
        global_f = global_f.unsqueeze(0)
    next_f = torch.as_tensor(features.next_features).float()
    if next_f.ndim == 1:
        next_f = next_f.unsqueeze(0)
    mask = torch.as_tensor(features.discrete_mask)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    payload = {
        "global": global_f,
        "next": next_f,
        "remaining": remaining,
        "remaining_mask": remaining_mask,
        "stations": stations,
        "station_mask": station_mask,
        "discrete_mask": mask,
    }
    if device is not None:
        payload = {key: value.to(device) for key, value in payload.items()}
    return payload
