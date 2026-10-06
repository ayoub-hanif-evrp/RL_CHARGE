"""Enrich envelope ablation SUMMARY with optimization stats from curves.jsonl."""

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


def _curve_means(path: Path) -> dict:
    if not path.is_file():
        return {"value_loss_mean": None, "grad_norm_preclip_mean": None}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    vl = [r.get("value_loss") for r in rows if r.get("value_loss") is not None]
    gn = [r.get("grad_norm_preclip") for r in rows if r.get("grad_norm_preclip") is not None]
    return {
        "value_loss_mean": mean(vl) if vl else None,
        "grad_norm_preclip_mean": mean(gn) if gn else None,
        "n_updates_logged": len(rows),
    }


def main() -> None:
    summary = json.loads((OUT / "SUMMARY.json").read_text(encoding="utf-8"))
    for variant in VARIANTS:
        feas, comps, route_feas, vlosses, gnorms = [], [], [], [], []
        for seed in SEEDS:
            val = json.loads((OUT / variant / f"seed_{seed}" / "validation.json").read_text(encoding="utf-8"))
            feas.append(float(val["parent_balanced_val_feasibility"]))
            comps.append(float(val["parent_balanced_completion_all"]))
            route_feas.append(float(val.get("route_weighted_feasibility", float("nan"))))
            cm = _curve_means(OUT / variant / f"seed_{seed}" / "curves.jsonl")
            if cm["value_loss_mean"] is not None:
                vlosses.append(cm["value_loss_mean"])
            if cm["grad_norm_preclip_mean"] is not None:
                gnorms.append(cm["grad_norm_preclip_mean"])
        summary["variants"][variant].update(
            {
                "display_name": NAMES[variant],
                "route_weighted_feas_mean": mean(route_feas),
                "value_loss_mean": mean(vlosses) if vlosses else None,
                "value_loss_sd": stdev(vlosses) if len(vlosses) > 1 else None,
                "grad_norm_preclip_mean": mean(gnorms) if gnorms else None,
                "grad_norm_preclip_sd": stdev(gnorms) if len(gnorms) > 1 else None,
                "note_missing_metrics": (
                    "NO_FEASIBLE_ACTION / station visits / charging time were not logged in the "
                    "initial validation.json schema; parent-balanced feas/completion and curve "
                    "optimization stats are reported. Re-rollout optional and not required for V3 integrity."
                ),
            }
        )
    summary["label"] = "POST-HOC DEVELOPMENT / TRAIN-VAL ONLY — NOT V3 CONFIRMATORY TEST EVIDENCE"
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Envelope ablation summary",
        "",
        "**POST-HOC / DEVELOPMENT-ONLY MECHANISM STUDY — NO V3 TEST USED.**",
        "",
        "| Variant | Mean VAL feas. | SD | Route feas. | Completion | Value loss | Grad norm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        s = summary["variants"][variant]
        lines.append(
            "| {name} | {vf:.3f} | {vs:.3f} | {rf:.3f} | {c:.3f} | {vl:.4f} | {g:.3f} |".format(
                name=s["display_name"],
                vf=s["val_feas_mean"],
                vs=s["val_feas_sd"],
                rf=s["route_weighted_feas_mean"],
                c=s["completion_mean"],
                vl=s["value_loss_mean"] if s["value_loss_mean"] is not None else float("nan"),
                g=s["grad_norm_preclip_mean"] if s["grad_norm_preclip_mean"] is not None else float("nan"),
            )
        )
    lines += [
        "",
        "Honest observation: on this SynthCharge VAL set, Arrival-to-Max mean feasibility is "
        "slightly highest; EnergyLower-to-TimeUpper is slightly lower. This is development evidence "
        "only and must not rewrite V3 confirmatory claims.",
        "",
        "## Per-seed feasibility",
        "",
        "| Variant | 42 | 43 | 44 | 45 | 46 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        s = summary["variants"][variant]
        ps = s["val_feas_per_seed"]
        cells = " | ".join(f"{ps[str(seed)]:.3f}" for seed in SEEDS)
        lines.append(f"| {s['display_name']} | {cells} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("enriched", OUT / "SUMMARY.md")


if __name__ == "__main__":
    main()
