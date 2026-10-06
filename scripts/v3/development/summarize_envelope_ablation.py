"""Summarize completed SynthCharge TRAIN/VAL envelope ablation (development only)."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "development" / "envelope_ablation"
VARIANTS = ("A_arrival_to_max", "B_energy_to_max", "C_energy_to_time")
NAMES = {
    "A_arrival_to_max": "Arrival-to-Max",
    "B_energy_to_max": "EnergyLower-to-Max",
    "C_energy_to_time": "EnergyLower-to-TimeUpper",
}
SEEDS = (42, 43, 44, 45, 46)


def main() -> None:
    summary = {}
    for variant in VARIANTS:
        feas, comps = [], []
        for seed in SEEDS:
            path = OUT / variant / f"seed_{seed}" / "validation.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            feas.append(float(data["parent_balanced_val_feasibility"]))
            comps.append(float(data["parent_balanced_completion_all"]))
        summary[variant] = {
            "display_name": NAMES[variant],
            "n": 5,
            "val_feas_mean": mean(feas),
            "val_feas_sd": stdev(feas),
            "val_feas_per_seed": {str(s): f for s, f in zip(SEEDS, feas)},
            "completion_mean": mean(comps),
            "completion_sd": stdev(comps),
        }
    payload = {
        "label": "POST-HOC DEVELOPMENT / TRAIN-VAL ONLY — NOT V3 CONFIRMATORY TEST EVIDENCE",
        "dataset": "synthcharge_final",
        "seeds": list(SEEDS),
        "variants": summary,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Envelope ablation summary",
        "",
        "**POST-HOC DEVELOPMENT / TRAIN-VAL ONLY — NOT V3 CONFIRMATORY.**",
        "",
        "| Variant | Mean VAL feas. | SD | Mean completion |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        s = summary[variant]
        lines.append(
            f"| {s['display_name']} | {s['val_feas_mean']:.3f} | {s['val_feas_sd']:.3f} | {s['completion_mean']:.3f} |"
        )
    lines += [
        "",
        "## Per-seed feasibility",
        "",
        "| Variant | 42 | 43 | 44 | 45 | 46 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        s = summary[variant]
        ps = s["val_feas_per_seed"]
        cells = " | ".join(f"{ps[str(seed)]:.3f}" for seed in SEEDS)
        lines.append(f"| {s['display_name']} | {cells} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
