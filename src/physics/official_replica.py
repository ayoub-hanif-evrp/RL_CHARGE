"""Literal transcription of the EVRPTW-GR Model 2 energy block.

Used as an independent numerical oracle for tests. Production code lives in
``energy.py`` and must match this replica within 1e-9 relative tolerance.

Source: sinarastani/EVRPTW-GR ``EVRPTW-GR-Model-2-Linearized.py`` energy
parameter and consumption block (``p_tract``, ``p_total``, ``consump``,
``consump_base``).
"""

from __future__ import annotations

import math
from typing import Tuple

from .parameters import VehicleEnergyParameters


def model2_tractive_power_kw(
    gross_mass_kg: float,
    angle_deg: float,
    params: VehicleEnergyParameters,
) -> float:
    """``p_tract`` in kW, including aero, grade, rolling, and acceleration."""
    v_s = params.physics_speed_m_s
    rad = math.radians(angle_deg)
    p_tract = (
        gross_mass_kg * params.acceleration_m_s2
        + gross_mass_kg * params.gravity_m_s2 * math.sin(rad)
        + 0.5
        * params.drag_coefficient
        * params.air_density_kg_m3
        * params.frontal_area_m2
        * v_s
        * v_s
        + gross_mass_kg
        * params.gravity_m_s2
        * params.rolling_resistance_coefficient
        * math.cos(rad)
    ) * v_s / 1000.0
    return p_tract


def model2_battery_power_kw(p_tract_kw: float, params: VehicleEnergyParameters) -> float:
    """``p_total`` in kW. Negative traction uses regenerative efficiency."""
    knv = params.accessory_power_kw
    p_acc = params.accessory_load_kw
    denom = params.powertrain_constant_kk * params.conversion_factor
    if p_tract_kw > 0.0:
        return (knv + (p_tract_kw / params.motor_efficiency + p_acc) / params.drivetrain_efficiency) / denom
    return params.regenerative_efficiency * (knv + (p_tract_kw + p_acc)) / denom


def model2_physical_kwh_per_km(p_total_kw: float, params: VehicleEnergyParameters) -> float:
    """``consump = p_total / vv``. Official comment: kWh per km at 60 km/h."""
    return p_total_kw / params.physics_speed_km_h


def model2_empty_flat_baseline(curb_mass_kg: float, params: VehicleEnergyParameters) -> float:
    """``consump_base``: empty vehicle, zero grade, Model 2 ``load = 0``, ``degree = 0``."""
    p_tract = model2_tractive_power_kw(curb_mass_kg, 0.0, params)
    p_total = model2_battery_power_kw(p_tract, params)
    return model2_physical_kwh_per_km(p_total, params)


def model2_normalized_rate(
    gross_mass_kg: float,
    angle_deg: float,
    curb_mass_kg: float,
    params: VehicleEnergyParameters,
) -> Tuple[float, float, float, float]:
    """Return ``(p_tract, p_total, consump, hh)`` with ``hh = consump / consump_base``."""
    p_tract = model2_tractive_power_kw(gross_mass_kg, angle_deg, params)
    p_total = model2_battery_power_kw(p_tract, params)
    consump = model2_physical_kwh_per_km(p_total, params)
    baseline = model2_empty_flat_baseline(curb_mass_kg, params)
    if baseline == 0.0:
        raise ZeroDivisionError("empty-flat baseline consumption is zero")
    return p_tract, p_total, consump, consump / baseline
