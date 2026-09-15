"""Physical model: distance, time, energy, battery, and charging.

This package has no dependency on reinforcement-learning libraries.
"""

from .battery import BatteryModel, BatteryState, BatteryTransition
from .charging import (
    BenchmarkCompatibleLinearChargingModel,
    ChargingModel,
    ChargingTimeResult,
    GenericPiecewiseLinearChargingModel,
)
from .distance import planar_euclidean, planar_euclidean_xy
from .energy import EnergyModel, EnergyResult
from .errors import (
    InvalidPhysicsParameterError,
    MissingPhysicsParameterError,
    PhysicsError,
    UnknownPhysicsProfileError,
)
from .network import DirectedArc, DirectedArcNetwork
from .parameters import (
    DEFAULT_PROFILE_NAME,
    PHYSICS_CONFIG_DIR,
    PhysicsProfile,
    VehicleEnergyParameters,
)
from .travel_time import travel_time

__all__ = [
    "DEFAULT_PROFILE_NAME",
    "PHYSICS_CONFIG_DIR",
    "BatteryModel",
    "BatteryState",
    "BatteryTransition",
    "BenchmarkCompatibleLinearChargingModel",
    "ChargingModel",
    "ChargingTimeResult",
    "DirectedArc",
    "DirectedArcNetwork",
    "EnergyModel",
    "EnergyResult",
    "GenericPiecewiseLinearChargingModel",
    "InvalidPhysicsParameterError",
    "MissingPhysicsParameterError",
    "PhysicsError",
    "PhysicsProfile",
    "UnknownPhysicsProfileError",
    "VehicleEnergyParameters",
    "planar_euclidean",
    "planar_euclidean_xy",
    "travel_time",
]
