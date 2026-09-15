"""Regression against stored Model 2 replica fixtures."""

import json
from pathlib import Path

import pytest

from data.parser import parse_instance
from data.paths import RAW_EVRPTW_GR_DIR
from domain.quantities import PayloadMass
from physics.energy import EnergyModel
from physics.network import DirectedArcNetwork
from physics.official_replica import model2_normalized_rate
from physics.parameters import PhysicsProfile

FIXTURE = Path(__file__).parent / "fixtures" / "energy_reference.json"


def test_energy_model_matches_stored_reference_fixtures():
    if not FIXTURE.is_file():
        pytest.skip("energy_reference.json not written yet")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["source"].startswith("EVRPTW-GR Model 2")
    for case in payload["cases"]:
        path = RAW_EVRPTW_GR_DIR / case["relative_path"]
        if not path.is_file():
            pytest.skip(f"missing {case['relative_path']}")
        instance = parse_instance(path)
        profile = PhysicsProfile.from_instance(instance)
        model = EnergyModel(DirectedArcNetwork(instance, profile.average_velocity), profile)
        result = model.energy_for_arc(
            case["from_id"], case["to_id"], PayloadMass(case["payload_kg"])
        )
        arc = model.network.arc(case["from_id"], case["to_id"])
        _, _, _, hh = model2_normalized_rate(
            profile.curb_mass_kg + case["payload_kg"],
            arc.angle_deg,
            profile.curb_mass_kg,
            profile.energy,
        )
        assert result.normalized_rate == pytest.approx(hh, rel=1e-9)
        assert result.net_energy.value == pytest.approx(case["net_energy"], rel=1e-8, abs=1e-8)
        assert result.angle_deg == pytest.approx(case["angle_deg"], abs=1e-8)
        assert result.distance.value == pytest.approx(case["distance"], rel=1e-9)
