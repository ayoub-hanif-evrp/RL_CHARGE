"""Publication-ready operational metrics from exported VAL route rows.

Reports both:
- method-specific feasible-route summaries (label survivor bias clearly)
- matched common-feasible subset (same route × seed, both methods feasible)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ROWS = ROOT / "results" / "v4" / "reward_development" / "val_route_rows"
OUT = ROOT / "results" / "v4" / "reward_development" / "tables"
SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("V3_TIME", "V4_BASE", "V4_PBRS", "V4_BASE_NO_L_FAIL")
PAIRS = (
    ("V3_TIME", "V4_BASE_NO_L_FAIL"),
    ("V4_BASE_NO_L_FAIL", "V4_PBRS"),
)
OPS = (
    ("route_completion_time", "completion_time"),
    ("total_charging_time", "charging_time"),
    ("total_energy_charged", "energy_charged"),
    ("n_station_visits", "n_station_visits"),
    ("terminal_soc", "terminal_soc"),
    ("total_distance", "total_distance"),
)


def _load(variant: str) -> list[dict]:
    path = ROWS / f"{variant}_val_rows.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index(rows: list[dict]) -> dict[tuple, dict]:
    return {(int(r["seed"]), r["route_id"]): r for r in rows}


def _mean_sd(vals: list[float]) -> tuple[float | None, float | None, int]:
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), sd, len(arr)


def method_feasible_summary() -> list[dict]:
    rows_out = []
    for variant in VARIANTS:
        rows = _load(variant)
        feas = [r for r in rows if r.get("feasible") and r.get("completed")]
        for field, label in OPS:
            mu, sd, n = _mean_sd([float(r[field]) for r in feas])
            rows_out.append(
                {
                    "subset": "method_feasible_routes_only",
                    "survivor_bias_warning": True,
                    "variant": variant,
                    "metric": label,
                    "n": n,
                    "mean": mu,
                    "sd": sd,
                }
            )
    return rows_out


def common_feasible_pairs() -> list[dict]:
    out = []
    for a, b in PAIRS:
        ia, ib = _index(_load(a)), _index(_load(b))
        keys = sorted(set(ia) & set(ib))
        common = [
            (ia[k], ib[k])
            for k in keys
            if ia[k].get("feasible") and ib[k].get("feasible") and ia[k].get("completed") and ib[k].get("completed")
        ]
        for field, label in OPS:
            a_vals = [float(ra[field]) for ra, _ in common]
            b_vals = [float(rb[field]) for _, rb in common]
            deltas = [bv - av for av, bv in zip(a_vals, b_vals)]
            mu_a, sd_a, n = _mean_sd(a_vals)
            mu_b, sd_b, _ = _mean_sd(b_vals)
            mu_d, sd_d, _ = _mean_sd(deltas)
            out.append(
                {
                    "subset": "matched_common_feasible",
                    "survivor_bias_warning": False,
                    "method_a": a,
                    "method_b": b,
                    "metric": label,
                    "n": n,
                    "mean_a": mu_a,
                    "sd_a": sd_a,
                    "mean_b": mu_b,
                    "sd_b": sd_b,
                    "mean_b_minus_a": mu_d,
                    "sd_b_minus_a": sd_d,
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md_method(path: Path, rows: list[dict]) -> None:
    lines = [
        "# VAL operational metrics — method-feasible routes only",
        "",
        "> **Survivor-bias warning:** means are over each method's own feasible successes only.",
        "> Do not interpret these as cross-method efficiency gains.",
        "",
        "| Variant | Metric | n | Mean | SD |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        mean_s = f"{r['mean']:.6g}" if r["mean"] is not None else "NA"
        sd_s = f"{r['sd']:.6g}" if r["sd"] is not None else "NA"
        lines.append(f"| {r['variant']} | {r['metric']} | {r['n']} | {mean_s} | {sd_s} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_md_common(path: Path, rows: list[dict]) -> None:
    lines = [
        "# VAL operational metrics — matched common-feasible subset",
        "",
        "Same `route_id` × `seed`; both methods feasible and completed.",
        "",
        "| A | B | Metric | n | Mean A | Mean B | Mean(B−A) | SD(B−A) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method_a']} | {r['method_b']} | {r['metric']} | {r['n']} | "
            f"{r['mean_a']:.6g} | {r['mean_b']:.6g} | {r['mean_b_minus_a']:.6g} | {r['sd_b_minus_a']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    method_rows = method_feasible_summary()
    common_rows = common_feasible_pairs()
    _write_csv(OUT / "ops_method_feasible.csv", method_rows)
    _write_csv(OUT / "ops_common_feasible.csv", common_rows)
    _write_md_method(OUT / "ops_method_feasible.md", method_rows)
    _write_md_common(OUT / "ops_common_feasible.md", common_rows)
    meta = {
        "source": "results/v4/reward_development/val_route_rows/",
        "n_method_rows": len(method_rows),
        "n_common_rows": len(common_rows),
        "confirmatory_test_consumed": False,
        "note": "Development VAL only. Common-feasible subset avoids survivor bias for cross-method ops claims.",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
