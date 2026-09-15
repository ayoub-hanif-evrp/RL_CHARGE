"""Ablation and architectural flags. Simulator physics stay unchanged except SOC mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    name: str = "FULL"
    discrete_u: bool = False
    soc_interval: str = "continuation_to_max"
    use_remaining_route: bool = True
    use_terrain_load_features: bool = True
    station_encoder: str = "attention"
    node_type_embedding: bool = False

    @classmethod
    def from_name(cls, name: str) -> "AblationConfig":
        name = str(name)
        table = {
            "FULL": cls(name="FULL", soc_interval="continuation_to_max"),
            "A1": cls(name="A1", discrete_u=True, soc_interval="continuation_to_max"),
            "A2": cls(name="A2", soc_interval="arrival_to_max"),
            "A3": cls(name="A3", use_remaining_route=False, soc_interval="continuation_to_max"),
            "A4": cls(name="A4", use_terrain_load_features=False, soc_interval="continuation_to_max"),
            "A5": cls(name="A5", station_encoder="pool", soc_interval="continuation_to_max"),
            "ATTENTION": cls(name="ATTENTION", node_type_embedding=True, soc_interval="continuation_to_max"),
            "LEGACY": cls(name="LEGACY", soc_interval="continuation_to_max"),
        }
        if name not in table:
            raise ValueError(f"unknown ablation {name!r}")
        return table[name]
