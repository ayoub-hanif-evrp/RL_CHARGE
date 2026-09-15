"""Named physics profiles and Demir / EVRPTW-GR Model 2 vehicle constants.

Nothing here invents a missing instance field. The default profile
``official_evrptwgr`` *overrides* curb mass and cargo capacity with the values
hard-coded in the official linearized Model 2 script. The ``raw_file`` profile
reads ``M`` and ``C`` from the instance and fails if they are absent.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from data.models import EVRPTWGRInstance
from data.paths import REPO_ROOT

from .errors import (
    InvalidPhysicsParameterError,
    MissingPhysicsParameterError,
    UnknownPhysicsProfileError,
)

PHYSICS_CONFIG_DIR = REPO_ROOT / "configs" / "physics"
DEFAULT_PROFILE_NAME = "official_evrptwgr"

_REQUIRED_ENERGY_KEYS = (
    "rolling_resistance_coefficient",
    "drag_coefficient",
    "air_density_kg_m3",
    "frontal_area_m2",
    "gravity_m_s2",
    "acceleration_m_s2",
    "drivetrain_efficiency",
    "motor_efficiency",
    "regenerative_efficiency",
    "physics_speed_km_h",
    "accessory_power_kw",
    "accessory_load_kw",
    "powertrain_constant_kk",
    "conversion_factor",
)


@dataclass(frozen=True)
class VehicleEnergyParameters:
    """Immutable Demir-style truck constants.

    Defaults, when loaded from ``official_evrptwgr.toml``, are copied from
    EVRPTW-GR Model 2 (sinarastani/EVRPTW-GR) / Demir et al. (2012). They are
    not SI-reinterpreted.
    """

    rolling_resistance_coefficient: float
    drag_coefficient: float
    air_density_kg_m3: float
    frontal_area_m2: float
    gravity_m_s2: float
    acceleration_m_s2: float
    drivetrain_efficiency: float
    motor_efficiency: float
    regenerative_efficiency: float
    physics_speed_km_h: float
    accessory_power_kw: float
    accessory_load_kw: float
    powertrain_constant_kk: float
    conversion_factor: float
    source: str

    @property
    def physics_speed_m_s(self) -> float:
        return self.physics_speed_km_h / 3.6


@dataclass(frozen=True)
class PhysicsProfile:
    """Resolved physical configuration for one instance under one named profile."""

    name: str
    source: str
    energy: VehicleEnergyParameters
    curb_mass_kg: float
    payload_capacity_kg: float
    battery_capacity: float
    inverse_refueling_rate: float
    average_velocity: float
    min_soc_fraction: float
    max_soc_fraction: float
    initial_soc_fraction: float
    loop_guard_station_visits: int
    file_consumption_rate_unused: float
    use_instance_curb_mass: bool
    use_instance_payload_capacity: bool

    @classmethod
    def from_instance(
        cls,
        instance: EVRPTWGRInstance,
        name: str = DEFAULT_PROFILE_NAME,
        config_dir: Optional[Path] = None,
    ) -> "PhysicsProfile":
        raw = load_profile_toml(name, config_dir=config_dir)
        return cls.from_mapping(instance, raw)

    @classmethod
    def from_mapping(
        cls, instance: EVRPTWGRInstance, raw: Mapping[str, Any]
    ) -> "PhysicsProfile":
        name = _require(raw, "name")
        source = _require(raw, "source")
        use_m = bool(_require(raw, "use_instance_curb_mass"))
        use_c = bool(_require(raw, "use_instance_payload_capacity"))

        if use_m:
            if instance.vehicle.curb_weight is None:
                raise MissingPhysicsParameterError(
                    f"{instance.metadata.relative_path}: profile {name!r} requires "
                    "instance curb weight M, which is absent"
                )
            curb = float(instance.vehicle.curb_weight)
        else:
            curb = float(_require(raw, "curb_mass_kg"))

        if use_c:
            cap = float(instance.vehicle.load_capacity)
        else:
            cap = float(_require(raw, "payload_capacity_kg"))

        if curb <= 0:
            raise InvalidPhysicsParameterError(f"curb_mass_kg must be positive, got {curb}")
        if cap <= 0:
            raise InvalidPhysicsParameterError(
                f"payload_capacity_kg must be positive, got {cap}"
            )

        velocity = float(instance.vehicle.average_velocity)
        if velocity <= 0:
            raise InvalidPhysicsParameterError(
                f"{instance.metadata.relative_path}: average velocity v must be "
                f"positive, got {velocity}"
            )

        battery = float(instance.vehicle.tank_capacity)
        if battery <= 0:
            raise InvalidPhysicsParameterError(
                f"{instance.metadata.relative_path}: battery capacity Q must be "
                f"positive, got {battery}"
            )

        g_charge = float(instance.vehicle.inverse_refueling_rate)
        if g_charge < 0:
            raise InvalidPhysicsParameterError(
                f"{instance.metadata.relative_path}: inverse refueling rate g "
                f"must be non-negative, got {g_charge}"
            )

        energy = VehicleEnergyParameters(
            source=source,
            **{key: float(_require(raw, key)) for key in _REQUIRED_ENERGY_KEYS},
        )
        _validate_efficiencies(energy)

        min_soc = float(_require(raw, "min_soc_fraction"))
        max_soc = float(_require(raw, "max_soc_fraction"))
        initial_soc = float(_require(raw, "initial_soc_fraction"))
        if not 0.0 <= min_soc <= max_soc <= 1.0:
            raise InvalidPhysicsParameterError(
                f"SOC bounds must satisfy 0 <= min <= max <= 1, got {min_soc}, {max_soc}"
            )
        if not min_soc <= initial_soc <= max_soc:
            raise InvalidPhysicsParameterError(
                f"initial_soc_fraction {initial_soc} is outside [{min_soc}, {max_soc}]"
            )

        loop_guard = int(_require(raw, "loop_guard_station_visits"))
        if loop_guard < 1:
            raise InvalidPhysicsParameterError(
                "loop_guard_station_visits must be a positive programming-loop limit"
            )

        return cls(
            name=str(name),
            source=str(source),
            energy=energy,
            curb_mass_kg=curb,
            payload_capacity_kg=cap,
            battery_capacity=battery,
            inverse_refueling_rate=g_charge,
            average_velocity=velocity,
            min_soc_fraction=min_soc,
            max_soc_fraction=max_soc,
            initial_soc_fraction=initial_soc,
            loop_guard_station_visits=loop_guard,
            file_consumption_rate_unused=float(instance.vehicle.consumption_rate),
            use_instance_curb_mass=use_m,
            use_instance_payload_capacity=use_c,
        )


def load_profile_toml(name: str, config_dir: Optional[Path] = None) -> dict[str, Any]:
    directory = PHYSICS_CONFIG_DIR if config_dir is None else Path(config_dir)
    path = directory / f"{name}.toml"
    if not path.is_file():
        raise UnknownPhysicsProfileError(f"physics profile {name!r} not found at {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise InvalidPhysicsParameterError(f"{path}: expected a TOML table")
    return raw


def _require(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw or raw[key] is None:
        raise MissingPhysicsParameterError(
            f"required physics parameter {key!r} is absent; refusing to invent a default"
        )
    return raw[key]


def _validate_efficiencies(energy: VehicleEnergyParameters) -> None:
    for label, value in (
        ("drivetrain_efficiency", energy.drivetrain_efficiency),
        ("motor_efficiency", energy.motor_efficiency),
        ("regenerative_efficiency", energy.regenerative_efficiency),
    ):
        if not 0.0 < value <= 1.0:
            raise InvalidPhysicsParameterError(f"{label} must be in (0, 1], got {value}")
    if energy.physics_speed_km_h <= 0:
        raise InvalidPhysicsParameterError("physics_speed_km_h must be positive")
    if energy.powertrain_constant_kk == 0 or energy.conversion_factor == 0:
        raise InvalidPhysicsParameterError("kk and conversion_factor must be non-zero")
