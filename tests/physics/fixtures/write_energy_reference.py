"""Reference energy values.

Provenance: computed with ``src/physics/official_replica.py``, a line-by-line
transcription of the energy block in sinarastani/EVRPTW-GR
``EVRPTW-GR-Model-2-Linearized.py`` (no Gurobi). Production ``EnergyModel``
must match these values.

Regenerate with:

    python tests/physics/fixtures/write_energy_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.parser import parse_instance  # noqa: E402
from data.paths import RAW_EVRPTW_GR_DIR  # noqa: E402
from domain.quantities import PayloadMass  # noqa: E402
from physics.energy import EnergyModel  # noqa: E402
from physics.network import DirectedArcNetwork  # noqa: E402
from physics.official_replica import model2_normalized_rate  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402

CASES = [
    {
        "name": "flat_empty_D0_C30",
        "relative_path": "Small_Network/5_Customers/Level/c101C5_L.txt",
        "from_id": "D0",
        "to_id": "C30",
        "payload_kg": 0.0,
    },
    {
        "name": "nl_D0_S5_empty",
        "relative_path": "Small_Network/5_Customers/Nearly_Level/c101C5_NL.txt",
        "from_id": "D0",
        "to_id": "S5",
        "payload_kg": 0.0,
    },
    {
        "name": "nl_S5_D0_empty",
        "relative_path": "Small_Network/5_Customers/Nearly_Level/c101C5_NL.txt",
        "from_id": "S5",
        "to_id": "D0",
        "payload_kg": 0.0,
    },
    {
        "name": "nl_S5_D0_heavy",
        "relative_path": "Small_Network/5_Customers/Nearly_Level/c101C5_NL.txt",
        "from_id": "S5",
        "to_id": "D0",
        "payload_kg": 3650.0,
    },
    {
        "name": "vg_D0_C12_empty",
        "relative_path": "Small_Network/5_Customers/Very_Gentle/c101C5_VG.txt",
        "from_id": "D0",
        "to_id": "C12",
        "payload_kg": 0.0,
    },
]


def main() -> None:
    out = []
    for case in CASES:
        instance = parse_instance(RAW_EVRPTW_GR_DIR / case["relative_path"])
        profile = PhysicsProfile.from_instance(instance)
        model = EnergyModel(DirectedArcNetwork(instance, profile.average_velocity), profile)
        result = model.energy_for_arc(
            case["from_id"], case["to_id"], PayloadMass(case["payload_kg"])
        )
        arc = model.network.arc(case["from_id"], case["to_id"])
        p_tract, p_total, consump, hh = model2_normalized_rate(
            profile.curb_mass_kg + case["payload_kg"],
            arc.angle_deg,
            profile.curb_mass_kg,
            profile.energy,
        )
        out.append(
            {
                **case,
                "distance": result.distance.value,
                "travel_time": arc.travel_time.value,
                "angle_deg": result.angle_deg,
                "gradient_percent": result.gradient_percent,
                "p_tract_kw": p_tract,
                "p_total_kw": p_total,
                "physical_kwh_per_km": consump,
                "normalized_rate": hh,
                "net_energy": result.net_energy.value,
                "g": profile.inverse_refueling_rate,
                "linear_charge_time_full": profile.inverse_refueling_rate * profile.battery_capacity,
            }
        )
    payload = {
        "source": "EVRPTW-GR Model 2 linearized energy block, transcribed in official_replica.py",
        "note": "No Gurobi. Values are from the physical formulae only.",
        "cases": out,
    }
    target = Path(__file__).with_name("energy_reference.json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
