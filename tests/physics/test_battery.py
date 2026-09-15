"""Battery floor and ceiling."""

from domain.quantities import BatteryEnergy, Energy
from physics.battery import BatteryModel
from physics.parameters import PhysicsProfile
from data.parser import parse_instance


def _battery(write_instance):
    path, root = write_instance()
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    return BatteryModel(profile)


def test_regeneration_cannot_exceed_capacity(write_instance):
    model = _battery(write_instance)
    full = model.initial_state()
    assert full.soc.value == 1.0
    transition = model.apply_arc_energy(full, Energy(-10.0))
    assert transition.feasible
    assert transition.ceiling_hit
    assert transition.after.energy == full.capacity


def test_insufficient_energy_is_infeasible(write_instance):
    model = _battery(write_instance)
    empty = model.state_from_soc(0.0)
    transition = model.apply_arc_energy(empty, Energy(1.0))
    assert not transition.feasible
    assert transition.floor_hit


def test_sign_convention_consumption_decreases_battery(write_instance):
    model = _battery(write_instance)
    start = model.initial_state()
    transition = model.apply_arc_energy(start, Energy(5.0))
    assert transition.feasible
    assert transition.after.energy == BatteryEnergy(start.energy.value - 5.0)
