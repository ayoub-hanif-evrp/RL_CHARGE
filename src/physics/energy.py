"""Load- and gradient-dependent EV energy on a directed arc.

Energy is **not** ``r * distance``. File ``r`` is unused. The Model 2 Demir
tractive-power formula is evaluated, then divided by the empty-flat baseline
so that an empty vehicle on a flat arc consumes 1 Schneider energy unit per
distance unit — matching the thesis statement that consumption was scaled to
``h = 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from domain.quantities import (
    Distance,
    Energy,
    Gradient,
    PayloadMass,
    energy_from_normalized_rate,
)

from .network import DirectedArc, DirectedArcNetwork
from .official_replica import (
    model2_empty_flat_baseline,
    model2_normalized_rate,
)
from .parameters import PhysicsProfile


@dataclass(frozen=True)
class EnergyResult:
    """Decomposed energy evaluation for one directed arc and one payload."""

    from_id: str
    to_id: str
    payload_mass: PayloadMass
    gross_mass_kg: float
    distance: Distance
    angle_deg: float
    gradient: Gradient
    gradient_percent: float
    traction_power_kw: float
    battery_power_kw: float
    physical_kwh_per_km: float
    empty_flat_physical_kwh_per_km: float
    normalized_rate: float
    net_energy: Energy
    recovered_energy: Energy
    consumed_energy: Energy

    @property
    def is_regenerating(self) -> bool:
        return self.traction_power_kw <= 0.0


class EnergyModel:
    def __init__(self, network: DirectedArcNetwork, profile: PhysicsProfile):
        self.network = network
        self.profile = profile
        self.empty_flat_physical_kwh_per_km = model2_empty_flat_baseline(
            profile.curb_mass_kg, profile.energy
        )

    def energy_for_arc(
        self, from_id: str, to_id: str, payload_mass: PayloadMass
    ) -> EnergyResult:
        if not isinstance(payload_mass, PayloadMass):
            raise TypeError("payload_mass must be a PayloadMass")
        if payload_mass.value < 0:
            raise ValueError(f"payload mass cannot be negative: {payload_mass.value}")
        arc = self.network.arc(from_id, to_id)
        return self.energy_for_directed_arc(arc, payload_mass)

    def energy_for_directed_arc(
        self, arc: DirectedArc, payload_mass: PayloadMass
    ) -> EnergyResult:
        if arc.is_zero_length:
            return _zero_result(
                arc, payload_mass, self.empty_flat_physical_kwh_per_km, self.profile.curb_mass_kg
            )

        gross = self.profile.curb_mass_kg + payload_mass.value
        p_tract, p_total, consump, hh = model2_normalized_rate(
            gross, arc.angle_deg, self.profile.curb_mass_kg, self.profile.energy
        )
        net = energy_from_normalized_rate(hh, arc.distance)
        consumed = Energy(max(net.value, 0.0))
        recovered = Energy(max(-net.value, 0.0))
        return EnergyResult(
            from_id=arc.from_id,
            to_id=arc.to_id,
            payload_mass=payload_mass,
            gross_mass_kg=gross,
            distance=arc.distance,
            angle_deg=arc.angle_deg,
            gradient=Gradient(math.sin(math.radians(arc.angle_deg))),
            gradient_percent=arc.gradient_percent,
            traction_power_kw=p_tract,
            battery_power_kw=p_total,
            physical_kwh_per_km=consump,
            empty_flat_physical_kwh_per_km=self.empty_flat_physical_kwh_per_km,
            normalized_rate=hh,
            net_energy=net,
            recovered_energy=recovered,
            consumed_energy=consumed,
        )


def _zero_result(
    arc: DirectedArc, payload_mass: PayloadMass, baseline: float, curb_mass_kg: float
) -> EnergyResult:
    zero = Energy(0.0)
    return EnergyResult(
        from_id=arc.from_id,
        to_id=arc.to_id,
        payload_mass=payload_mass,
        gross_mass_kg=curb_mass_kg + payload_mass.value,
        distance=arc.distance,
        angle_deg=0.0,
        gradient=Gradient(0.0),
        gradient_percent=0.0,
        traction_power_kw=0.0,
        battery_power_kw=0.0,
        physical_kwh_per_km=0.0,
        empty_flat_physical_kwh_per_km=baseline,
        normalized_rate=0.0,
        net_energy=zero,
        recovered_energy=zero,
        consumed_energy=zero,
    )
