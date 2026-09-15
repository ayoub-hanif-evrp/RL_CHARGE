"""Canonical battery and SOC transition.

Sign convention
---------------
``battery_after = battery_before - net_arc_energy``

``net_arc_energy > 0`` consumes stored energy. ``net_arc_energy < 0`` is
regeneration. Surplus regeneration is clipped at ``max_soc * Q`` (the physical
ceiling, equivalent to discarding energy the MILP inequalities may leave
unstored). Dropping below ``min_soc * Q`` is infeasible.

Benchmark profile: ``min_soc_fraction = 0``, ``max_soc_fraction = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.quantities import BatteryEnergy, Energy, SocFraction

from .parameters import PhysicsProfile


@dataclass(frozen=True)
class BatteryState:
    energy: BatteryEnergy
    capacity: BatteryEnergy
    min_soc_fraction: float
    max_soc_fraction: float

    @property
    def soc(self) -> SocFraction:
        if self.capacity.value <= 0:
            raise ZeroDivisionError("battery capacity Q must be positive")
        return SocFraction(self.energy.value / self.capacity.value)

    @property
    def min_energy(self) -> BatteryEnergy:
        return BatteryEnergy(self.min_soc_fraction * self.capacity.value)

    @property
    def max_energy(self) -> BatteryEnergy:
        return BatteryEnergy(self.max_soc_fraction * self.capacity.value)


@dataclass(frozen=True)
class BatteryTransition:
    before: BatteryState
    after: BatteryState
    net_arc_energy: Energy
    floor_hit: bool
    ceiling_hit: bool
    feasible: bool


class BatteryModel:
    def __init__(self, profile: PhysicsProfile):
        self.profile = profile
        self.capacity = BatteryEnergy(profile.battery_capacity)

    def initial_state(self) -> BatteryState:
        energy = BatteryEnergy(self.profile.initial_soc_fraction * self.capacity.value)
        return BatteryState(
            energy=energy,
            capacity=self.capacity,
            min_soc_fraction=self.profile.min_soc_fraction,
            max_soc_fraction=self.profile.max_soc_fraction,
        )

    def state_from_soc(self, soc_fraction: float) -> BatteryState:
        energy = BatteryEnergy(soc_fraction * self.capacity.value)
        return BatteryState(
            energy=energy,
            capacity=self.capacity,
            min_soc_fraction=self.profile.min_soc_fraction,
            max_soc_fraction=self.profile.max_soc_fraction,
        )

    def apply_arc_energy(self, state: BatteryState, net_arc_energy: Energy) -> BatteryTransition:
        if not isinstance(net_arc_energy, Energy):
            raise TypeError("net_arc_energy must be Energy, not a distance or SOC")
        proposed = state.energy.value - net_arc_energy.value
        min_e = state.min_energy.value
        max_e = state.max_energy.value
        floor_hit = proposed < min_e - 1e-12
        ceiling_hit = proposed > max_e + 1e-12
        if floor_hit:
            after = BatteryState(
                energy=BatteryEnergy(proposed),
                capacity=state.capacity,
                min_soc_fraction=state.min_soc_fraction,
                max_soc_fraction=state.max_soc_fraction,
            )
            return BatteryTransition(
                before=state,
                after=after,
                net_arc_energy=net_arc_energy,
                floor_hit=True,
                ceiling_hit=False,
                feasible=False,
            )
        clipped = min(max(proposed, min_e), max_e)
        after = BatteryState(
            energy=BatteryEnergy(clipped),
            capacity=state.capacity,
            min_soc_fraction=state.min_soc_fraction,
            max_soc_fraction=state.max_soc_fraction,
        )
        return BatteryTransition(
            before=state,
            after=after,
            net_arc_energy=net_arc_energy,
            floor_hit=False,
            ceiling_hit=ceiling_hit,
            feasible=True,
        )

    def apply_charge(
        self, state: BatteryState, energy_added: Energy
    ) -> BatteryTransition:
        if energy_added.value < -1e-12:
            raise ValueError("charging cannot remove energy; use apply_arc_energy")
        return self.apply_arc_energy(state, Energy(-energy_added.value))
