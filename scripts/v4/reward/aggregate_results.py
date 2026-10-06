"""Aggregate V4 reward ablation validation.json into SUMMARY.md/json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ABLATION = ROOT / "results" / "v4" / "reward_development" / "ablation"
ORDER = ("V3_TIME", "V4_BASE", "V4_PBRS", "V4_BASE_NO_L_FAIL")


def main() -> None:
    rows = []
    for path in sorted(ABLATION.rglob("validation.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    by: dict[str, list] = defaultdict(list)
    for row in rows:
        by[row["variant"]].append(row)

    summary = {}
    for variant in ORDER:
        rs = sorted(by.get(variant, []), key=lambda r: int(r["seed"]))
        feas = [float(r["parent_balanced_val_feasibility"]) for r in rs]
        comp = [float(r["parent_balanced_completion_all"]) for r in rs]
        best = [r.get("best_update") for r in rs if r.get("best_update") is not None]
        summary[variant] = {
            "n": len(rs),
            "feas_mean": float(np.mean(feas)) if feas else None,
            "feas_sd": float(np.std(feas, ddof=1)) if len(feas) > 1 else 0.0,
            "feas_per_seed": {str(r["seed"]): float(r["parent_balanced_val_feasibility"]) for r in rs},
            "comp_mean": float(np.mean(comp)) if comp else None,
            "comp_sd": float(np.std(comp, ddof=1)) if len(comp) > 1 else 0.0,
            "comp_per_seed": {str(r["seed"]): float(r["parent_balanced_completion_all"]) for r in rs},
            "best_update_mean": float(np.mean(best)) if best else None,
            "statuses": [r.get("status") for r in rs],
        }

    payload = {
        "c_train": 10.0,
        "n_cells": len(rows),
        "variants": summary,
        "confirmatory_test_consumed": False,
        "note": "TRAIN/VAL development only",
    }
    out_dir = ROOT / "results" / "v4" / "reward_development"
    (out_dir / "SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# V4 reward ablation SUMMARY (TRAIN/VAL only)",
        "",
        "C_train = 10.0 (TRAIN median horizon). No confirmatory TEST.",
        "",
        "| Variant | VAL feas. mean±SD | Completion mean±SD | Mean best update |",
        "|---|---:|---:|---:|",
    ]
    for variant in ORDER:
        s = summary[variant]
        lines.append(
            "| {v} | {fm:.1f}% ± {fs:.1f} | {cm:.3f} ± {cs:.3f} | {bu:.0f} |".format(
                v=variant,
                fm=100.0 * s["feas_mean"],
                fs=100.0 * s["feas_sd"],
                cm=s["comp_mean"],
                cs=s["comp_sd"],
                bu=s["best_update_mean"] if s["best_update_mean"] is not None else -1,
            )
        )
    lines += ["", "## Per-seed feasibility", ""]
    for variant in ORDER:
        s = summary[variant]
        cells = ", ".join(f"{k}:{100.0 * val:.1f}%" for k, val in s["feas_per_seed"].items())
        lines.append(f"- **{variant}**: {cells}")
    lines += ["", "## Per-seed completion", ""]
    for variant in ORDER:
        s = summary[variant]
        cells = ", ".join(f"{k}:{val:.3f}" for k, val in s["comp_per_seed"].items())
        lines.append(f"- **{variant}**: {cells}")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
