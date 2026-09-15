"""PyVRP routing configuration loaded from TOML. No hidden scientific defaults."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from data.models import NetworkGroup
from data.paths import REPO_ROOT
from physics.errors import InvalidPhysicsParameterError, MissingPhysicsParameterError
from routing.serialize import canonical_dumps

ROUTING_CONFIG_DIR = REPO_ROOT / "configs" / "routing"
DEFAULT_ROUTING_CONFIG = "pyvrp"


@dataclass(frozen=True)
class PyVRPConfig:
    generator: str
    routing_problem: str
    physics_profile: str
    seed: int
    distance_scale: int
    demand_scale: int
    rounding_policy: str
    stop: str
    display: bool
    fleet_policy: str
    demand_mapping: str
    iterations: Mapping[str, int]

    def iterations_for(self, group: NetworkGroup) -> int:
        key = group.value
        if key not in self.iterations:
            raise MissingPhysicsParameterError(
                f"PyVRP config has no MaxIterations budget for {key}"
            )
        value = int(self.iterations[key])
        if value < 1:
            raise InvalidPhysicsParameterError(f"MaxIterations for {key} must be positive")
        return value

    def fingerprint(self) -> str:
        payload = {
            "demand_mapping": self.demand_mapping,
            "demand_scale": self.demand_scale,
            "display": self.display,
            "distance_scale": self.distance_scale,
            "fleet_policy": self.fleet_policy,
            "generator": self.generator,
            "iterations": dict(self.iterations),
            "physics_profile": self.physics_profile,
            "rounding_policy": self.rounding_policy,
            "routing_problem": self.routing_problem,
            "seed": self.seed,
            "stop": self.stop,
        }
        digest = hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()
        return digest

    @classmethod
    def from_toml(cls, path: Path | None = None) -> "PyVRPConfig":
        path = path or (ROUTING_CONFIG_DIR / f"{DEFAULT_ROUTING_CONFIG}.toml")
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        required = (
            "generator",
            "routing_problem",
            "physics_profile",
            "seed",
            "distance_scale",
            "demand_scale",
            "rounding_policy",
            "stop",
            "display",
            "fleet_policy",
            "demand_mapping",
            "iterations",
        )
        for key in required:
            if key not in raw:
                raise MissingPhysicsParameterError(
                    f"{path}: required routing parameter {key!r} is absent"
                )
        if raw["stop"] != "MaxIterations":
            raise InvalidPhysicsParameterError(
                "Part 2 requires stop = 'MaxIterations' for reproducibility"
            )
        if raw["demand_mapping"] != "pickup":
            raise InvalidPhysicsParameterError(
                "EVRPTW-GR customer routing must map demand to PyVRP pickup, not delivery"
            )
        if raw["fleet_policy"] != "unrestricted_fleet_with_fixed_cost":
            raise InvalidPhysicsParameterError(
                "fleet_policy must be 'unrestricted_fleet_with_fixed_cost'"
            )
        iterations = {str(k): int(v) for k, v in dict(raw["iterations"]).items()}
        return cls(
            generator=str(raw["generator"]),
            routing_problem=str(raw["routing_problem"]),
            physics_profile=str(raw["physics_profile"]),
            seed=int(raw["seed"]),
            distance_scale=int(raw["distance_scale"]),
            demand_scale=int(raw["demand_scale"]),
            rounding_policy=str(raw["rounding_policy"]),
            stop=str(raw["stop"]),
            display=bool(raw["display"]),
            fleet_policy=str(raw["fleet_policy"]),
            demand_mapping=str(raw["demand_mapping"]),
            iterations=iterations,
        )
