"""Shared encoder: Transformer over the frozen remaining route plus station attention."""

from __future__ import annotations

import math

import torch
from torch import nn

from .features import CUSTOMER_DIM, GLOBAL_DIM, NEXT_DIM, STATION_DIM


class HybridEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.0,
        station_encoder: str = "attention",
        node_type_embedding: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.station_encoder = station_encoder
        self.node_type_embedding = bool(node_type_embedding)
        extra = 3 if self.node_type_embedding else 0
        self.global_proj = nn.Linear(GLOBAL_DIM + NEXT_DIM + extra, d_model)
        self.customer_proj = nn.Linear(CUSTOMER_DIM, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.remaining_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.station_proj = nn.Linear(STATION_DIM + extra, d_model)
        self.station_query = nn.Linear(d_model, d_model)
        self.type_embed = nn.Embedding(3, extra) if extra else None
        self.out = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, batch: dict) -> dict:
        global_next = torch.cat([batch["global"], batch["next"]], dim=-1)
        if self.type_embed is not None:
            # next-node type: depot vs customer from the last two next-feature flags
            next_flags = batch["next"][..., -2:]
            next_type = torch.argmax(next_flags, dim=-1)
            global_next = torch.cat([global_next, self.type_embed(next_type)], dim=-1)
        h_ctx = self.global_proj(global_next)
        remaining = self.customer_proj(batch["remaining"])
        remaining_mask = batch["remaining_mask"] < 0.5
        encoded = self.remaining_encoder(remaining, src_key_padding_mask=remaining_mask)
        keep = (~remaining_mask).unsqueeze(-1).to(encoded.dtype)
        denom = keep.sum(dim=1).clamp(min=1.0)
        h_rem = (encoded * keep).sum(dim=1) / denom
        stations = batch["stations"]
        if self.type_embed is not None:
            station_types = torch.full(
                (stations.size(0), stations.size(1)),
                2,
                dtype=torch.long,
                device=stations.device,
            )
            stations = torch.cat([stations, self.type_embed(station_types)], dim=-1)
        stations = self.station_proj(stations)
        station_mask = batch["station_mask"] < 0.5
        if self.station_encoder == "pool":
            keep_s = (~station_mask).unsqueeze(-1).to(stations.dtype)
            h_stat = (stations * keep_s).sum(dim=1) / keep_s.sum(dim=1).clamp(min=1.0)
        else:
            query = self.station_query(h_ctx).unsqueeze(1)
            scores = (query * stations).sum(dim=-1) / math.sqrt(self.d_model)
            scores = scores.masked_fill(station_mask, -1e9)
            weights = torch.softmax(scores, dim=-1)
            weights = torch.where(station_mask, torch.zeros_like(weights), weights)
            h_stat = (weights.unsqueeze(-1) * stations).sum(dim=1)
        h = self.out(torch.cat([h_ctx, h_rem, h_stat], dim=-1))
        return {
            "h": h,
            "station_embed": stations,
            "ctx": h_ctx,
        }
