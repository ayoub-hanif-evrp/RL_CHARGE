"""Recompute physics-reference residuals and print energy ranges by terrain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def _load_models(path: Path):
    instance = parse_instance(path)
    profile = PhysicsProfile.from_instance(instance)
    network = DirectedArcNetwork(instance, profile.average_velocity)
    return instance, profile, EnergyModel(network, profile)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "tests" / "physics" / "fixtures" / "energy_reference.json",
    )
    args = parser.parse_args(argv)

    if args.fixtures.is_file():
        fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
        worst = 0.0
        for case in fixtures["cases"]:
            instance, profile, model = _load_models(RAW_EVRPTW_GR_DIR / case["relative_path"])
            result = model.energy_for_arc(
                case["from_id"], case["to_id"], PayloadMass(case["payload_kg"])
            )
            residual = abs(result.net_energy.value - case["net_energy"])
            rel = residual / max(abs(case["net_energy"]), 1e-12)
            worst = max(worst, rel)
            print(
                f"{case['name']}: net={result.net_energy.value:.12f} "
                f"fixture={case['net_energy']:.12f} rel={rel:.3e}"
            )
        print(f"worst_relative_residual={worst:.3e}")

    samples = {
        "Level": "Small_Network/5_Customers/Level/c101C5_L.txt",
        "Nearly_Level": "Small_Network/5_Customers/Nearly_Level/c101C5_NL.txt",
        "Very_Gentle": "Small_Network/5_Customers/Very_Gentle/c101C5_VG.txt",
    }
    for terrain, relative in samples.items():
        path = RAW_EVRPTW_GR_DIR / relative
        if not path.is_file():
            print(f"{terrain}: missing {relative}")
            continue
        instance, profile, model = _load_models(path)
        nets = []
        for customer in instance.customers:
            energy = model.energy_for_arc(
                instance.depot.string_id, customer.string_id, PayloadMass(0.0)
            )
            nets.append(energy.net_energy.value)
        print(
            f"{terrain}: depot->customer empty net energy "
            f"min={min(nets):.6f} max={max(nets):.6f}"
        )
        p_tract, _, _, hh = model2_normalized_rate(
            profile.curb_mass_kg, 0.0, profile.curb_mass_kg, profile.energy
        )
        print(f"{terrain}: empty-flat hh={hh:.12f} (expected 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
