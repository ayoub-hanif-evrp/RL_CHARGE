"""Paired V4 vs V3 TEST analysis + stratified summaries. No reselection."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "results" / "v4_test"
RAW = V4 / "raw" / "synthcharge_v4_test.jsonl"
OUT = V4 / "statistics"
SEEDS = (42, 43, 44, 45, 46)
OPS = (
    "route_completion_time",
    "total_charging_time",
    "total_energy_charged",
    "n_station_visits",
    "terminal_soc",
    "total_distance",
    "runtime_s",
)


def _load() -> list[dict]:
    return [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index(rows: list[dict], method: str) -> dict[tuple, dict]:
    out = {}
    for r in rows:
        if r["method"] != method:
            continue
        seed = r.get("seed")
        if seed is None:
            continue
        out[(int(seed), r["route_id"])] = r
    return out


def _mean_sd(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "sd": None}
    if len(vals) == 1:
        return {"n": 1, "mean": float(vals[0]), "sd": 0.0}
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "sd": float(statistics.stdev(vals))}


def paired_v4_v3(rows: list[dict]) -> dict:
    ia, ib = _index(rows, "V4_FA_HPPO"), _index(rows, "V3_FA_HPPO")
    keys = sorted(set(ia) & set(ib))
    v4_only = v3_only = both_ok = both_bad = 0
    d_comp = []
    common = []
    per_seed = {s: {"v4": 0, "v3": 0, "n": 0} for s in SEEDS}
    for key in keys:
        a, b = ia[key], ib[key]
        seed = key[0]
        per_seed[seed]["n"] += 1
        fa, fb = bool(a["feasible"]), bool(b["feasible"])
        per_seed[seed]["v4"] += int(fa)
        per_seed[seed]["v3"] += int(fb)
        if fa and not fb:
            v4_only += 1
        elif fb and not fa:
            v3_only += 1
        elif fa and fb:
            both_ok += 1
            common.append((a, b))
        else:
            both_bad += 1
        d_comp.append(float(a["completion_time_all_routes"]) - float(b["completion_time_all_routes"]))

    ops = {}
    for field in OPS:
        deltas = [float(a[field]) - float(b[field]) for a, b in common]
        ops[field] = {
            "n": len(deltas),
            "mean_v4_minus_v3": float(np.mean(deltas)) if deltas else None,
            "sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else (0.0 if deltas else None),
        }

    def feas_rate(method: str) -> dict:
        vals = []
        idx = _index(rows, method)
        for seed in SEEDS:
            subset = [idx[(seed, rid)] for (s, rid) in idx if s == seed]
            vals.append(sum(1 for r in subset if r["feasible"]) / len(subset))
        return _mean_sd(vals)

    return {
        "n_paired": len(keys),
        "feasibility_counts": {
            "v4_only_feasible": v4_only,
            "v3_only_feasible": v3_only,
            "both_feasible": both_ok,
            "both_infeasible": both_bad,
        },
        "feasibility_mean_across_seeds": {"V4_FA_HPPO": feas_rate("V4_FA_HPPO"), "V3_FA_HPPO": feas_rate("V3_FA_HPPO")},
        "per_seed_feasibility": {
            str(s): {
                "V4_FA_HPPO": per_seed[s]["v4"] / per_seed[s]["n"],
                "V3_FA_HPPO": per_seed[s]["v3"] / per_seed[s]["n"],
            }
            for s in SEEDS
        },
        "paired_completion_all_v4_minus_v3": _mean_sd(d_comp),
        "common_feasible_operational_v4_minus_v3": ops,
    }


def method_overall(rows: list[dict]) -> dict:
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    for method, recs in sorted(by.items()):
        # For seeded methods, mean across seeds of route-mean feasibility
        if any(r.get("seed") is not None for r in recs):
            seed_feas = []
            seed_comp = []
            for seed in SEEDS:
                subset = [r for r in recs if r.get("seed") == seed]
                if not subset:
                    continue
                seed_feas.append(sum(1 for r in subset if r["feasible"]) / len(subset))
                seed_comp.append(sum(float(r["completion_time_all_routes"]) for r in subset) / len(subset))
            out[method] = {
                "feasibility": _mean_sd(seed_feas),
                "completion_all": _mean_sd(seed_comp),
            }
        else:
            out[method] = {
                "feasibility": {
                    "n": len(recs),
                    "mean": sum(1 for r in recs if r["feasible"]) / len(recs),
                    "sd": None,
                },
                "completion_all": _mean_sd([float(r["completion_time_all_routes"]) for r in recs]),
            }
    return out


def stratified(rows: list[dict], key: str) -> dict:
    """V4 vs V3 feasibility by stratum, averaged over seeds."""
    ia, ib = _index(rows, "V4_FA_HPPO"), _index(rows, "V3_FA_HPPO")
    strata = sorted({ia[k].get(key) for k in ia})
    out = {}
    for stratum in strata:
        v4_rates, v3_rates = [], []
        for seed in SEEDS:
            a_sub = [ia[k] for k in ia if k[0] == seed and ia[k].get(key) == stratum]
            b_sub = [ib[k] for k in ib if k[0] == seed and ib[k].get(key) == stratum]
            if not a_sub:
                continue
            v4_rates.append(sum(1 for r in a_sub if r["feasible"]) / len(a_sub))
            v3_rates.append(sum(1 for r in b_sub if r["feasible"]) / len(b_sub))
        out[str(stratum)] = {"V4_FA_HPPO": _mean_sd(v4_rates), "V3_FA_HPPO": _mean_sd(v3_rates)}
    return out


def main() -> None:
    if not RAW.is_file():
        raise SystemExit("raw TEST rows missing")
    rows = _load()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "paired_v4_vs_v3": paired_v4_v3(rows),
        "method_overall": method_overall(rows),
        "by_layout": stratified(rows, "layout"),
        "by_length_bin": stratified(rows, "length_bin"),
        "by_charge_class": stratified(rows, "charge_class"),
        "by_n_customers": stratified(rows, "n_customers_instance"),
        "confirmatory_test_consumed": True,
        "note": "Report regimes where V4 is worse; do not claim superiority without support.",
    }
    (OUT / "main_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Markdown brief
    p = payload["paired_v4_vs_v3"]
    c = p["feasibility_counts"]
    lines = [
        "# V4 vs V3 fresh TEST summary",
        "",
        f"- V4-only feasible: {c['v4_only_feasible']}",
        f"- V3-only feasible: {c['v3_only_feasible']}",
        f"- both feasible: {c['both_feasible']}",
        f"- both infeasible: {c['both_infeasible']}",
        "",
        f"- mean seed feasibility V4: {p['feasibility_mean_across_seeds']['V4_FA_HPPO']}",
        f"- mean seed feasibility V3: {p['feasibility_mean_across_seeds']['V3_FA_HPPO']}",
        f"- paired completion_all mean(V4-V3): {p['paired_completion_all_v4_minus_v3']}",
        "",
        "Common-feasible operational mean(V4-V3):",
    ]
    for k, v in p["common_feasible_operational_v4_minus_v3"].items():
        lines.append(f"- {k}: {v}")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["paired_v4_vs_v3"]["feasibility_counts"], indent=2))


if __name__ == "__main__":
    main()
