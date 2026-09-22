"""Auto-generate manuscript tables from a single scenario. Never hand-edit numbers."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import RESULTS_DIR  # noqa: E402
from experiments.batch import read_jsonl_dicts  # noqa: E402
from experiments.isolation import require_scenario  # noqa: E402
from experiments.stats import (
    LEARNED_PAPER_METHODS,
    attach_feas_rate,
    cluster_bootstrap_ci,
    dedupe_soc_greedy,
    exclude_non_paper_methods,
    hierarchical_bootstrap_ci,
    map_ablation_method,
    mean_sd_across_training_seeds,
    parent_balanced_mean,
    route_weighted_mean,
    summarize_method,
)


def _load(raw_dir: Path) -> List[dict]:
    rows = []
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*.jsonl")):
            rows.extend(read_jsonl_dicts(path))
    return rows


def _md(path: Path, title: str, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(k) is None else str(row.get(k)) for k in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _groups(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("method"):
            out[str(row["method"])].append(row)
    return dict(out)


def _is_learned(method: str) -> bool:
    return (
        method in LEARNED_PAPER_METHODS
        or str(method).startswith("HybridPPO")
        or method in {"FULL", "A1", "A2", "A3", "A4", "A5"}
    )


def _method_row(method: str, group: List[dict]) -> dict:
    tagged = attach_feas_rate(group)
    n_routes = len({row.get("route_id") for row in group})
    seeds = sorted({int(row.get("seed", 0)) for row in group})
    learned = _is_learned(method)
    if learned:
        time_ci = hierarchical_bootstrap_ci(group, "completion_time_all_routes")
        feas_ci = hierarchical_bootstrap_ci(tagged, "feas_rate")
        seed_stats = mean_sd_across_training_seeds(group, "completion_time_all_routes")
        n_seeds = seed_stats.get("n_seeds")
        sd = seed_stats.get("sd_across_seeds")
        ci_lo = time_ci.get("lo")
        ci_hi = time_ci.get("hi")
        uncertainty = "hierarchical_seed_then_parent"
    else:
        time_ci = cluster_bootstrap_ci(group, "completion_time_all_routes")
        feas_ci = cluster_bootstrap_ci(tagged, "feas_rate")
        n_seeds = 1
        sd = None
        ci_lo = time_ci.get("lo")
        ci_hi = time_ci.get("hi")
        uncertainty = "parent_cluster"
    return {
        "method": method,
        "n_routes": n_routes,
        "n_seeds": n_seeds,
        "n_rows": len(group),
        "route_weighted_feasibility": route_weighted_mean(tagged, "feas_rate"),
        "parent_balanced_feasibility": parent_balanced_mean(tagged, "feas_rate"),
        "feasibility_ci95_lo": feas_ci.get("lo"),
        "feasibility_ci95_hi": feas_ci.get("hi"),
        "route_weighted_completion_all": route_weighted_mean(group, "completion_time_all_routes"),
        "parent_balanced_completion_all": parent_balanced_mean(group, "completion_time_all_routes"),
        "completion_all_ci95_lo": ci_lo,
        "completion_all_ci95_hi": ci_hi,
        "seed_sd_completion_all": sd,
        "uncertainty": uncertainty,
        "seeds": ",".join(str(s) for s in seeds),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RESULTS_DIR / "raw")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "tables")
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args(argv)
    scenario = require_scenario(args.scenario)
    rows = exclude_non_paper_methods(_load(args.raw))
    rows = [row for row in rows if row.get("scenario") == scenario]
    if scenario in {"main_test", "ablation", "soc_reserve", "exact_small"}:
        rows = [row for row in rows if row.get("split") == "test" or scenario == "exact_small"]
    if scenario == "soc_reserve":
        rows = dedupe_soc_greedy(rows)
    out = args.out / scenario
    out.mkdir(parents=True, exist_ok=True)
    groups = _groups(rows)

    fields_a = [
        "method",
        "n_routes",
        "n_seeds",
        "n_rows",
        "route_weighted_feasibility",
        "parent_balanced_feasibility",
        "feasibility_ci95_lo",
        "feasibility_ci95_hi",
        "route_weighted_completion_all",
        "parent_balanced_completion_all",
        "completion_all_ci95_lo",
        "completion_all_ci95_hi",
        "seed_sd_completion_all",
        "uncertainty",
        "seeds",
    ]
    table_a = [_method_row(method, group) for method, group in sorted(groups.items())]
    title_a = f"Table A — {scenario} performance (failures retained; H for infeasible completion)"
    _md(out / "table_A_completion_time.md", title_a, table_a, fields_a)

    table_b = []
    for method, group in sorted(groups.items()):
        feas = [row for row in group if row.get("feasible") and row.get("route_completion_time") is not None]
        s = summarize_method(feas, "route_completion_time") if feas else {}
        ops = {}
        for key in (
            "total_travel_time",
            "total_charging_time",
            "total_waiting_at_customers",
            "n_station_visits",
            "total_energy_consumed",
            "total_energy_regenerated",
            "total_energy_charged",
            "total_distance",
        ):
            ops[key] = summarize_method(feas, key).get("mean") if feas and any(key in r for r in feas) else None
        table_b.append(
            {
                "method": method,
                "n_feasible": len(feas),
                "n_rows": len(group),
                "note": "conditional_on_feasibility",
                "completion_time": s.get("mean"),
                **ops,
            }
        )
    _md(
        out / "table_B_feasible_only.md",
        f"Table B — {scenario} feasible-only operational metrics (conditional on feasibility)",
        table_b,
        [
            "method",
            "n_feasible",
            "n_rows",
            "note",
            "completion_time",
            "total_travel_time",
            "total_charging_time",
            "total_waiting_at_customers",
            "n_station_visits",
            "total_energy_consumed",
            "total_energy_regenerated",
            "total_energy_charged",
            "total_distance",
        ],
    )

    table_c = []
    for method, group in sorted(groups.items()):
        by_t: Dict[str, List[dict]] = defaultdict(list)
        for row in group:
            by_t[str(row.get("terrain") or row.get("terrain_variant") or "")].append(row)
        for terrain, items in sorted(by_t.items()):
            if not terrain:
                continue
            tagged = attach_feas_rate(items)
            table_c.append(
                {
                    "method": method,
                    "terrain": terrain,
                    "n_rows": len(items),
                    "feasibility": route_weighted_mean(tagged, "feas_rate"),
                    "completion_all": route_weighted_mean(items, "completion_time_all_routes"),
                    "parent_balanced_completion_all": parent_balanced_mean(items, "completion_time_all_routes"),
                }
            )
    _md(
        out / "table_C_terrain.md",
        f"Table C — {scenario} terrain (frozen terrain-sibling customer sequences)",
        table_c,
        ["method", "terrain", "n_rows", "feasibility", "completion_all", "parent_balanced_completion_all"],
    )

    table_d = []
    for method, group in sorted(groups.items()):
        for key in ("network_group", "customer_distribution", "schedule_type"):
            buckets: Dict[str, List[dict]] = defaultdict(list)
            for row in group:
                buckets[str(row.get(key, ""))].append(row)
            for label, items in sorted(buckets.items()):
                if not label or label == "None":
                    continue
                tagged = attach_feas_rate(items)
                table_d.append(
                    {
                        "method": method,
                        "slice": key,
                        "value": label,
                        "n_rows": len(items),
                        "feasibility": route_weighted_mean(tagged, "feas_rate"),
                        "completion_all": route_weighted_mean(items, "completion_time_all_routes"),
                    }
                )
    _md(
        out / "table_D_size_family.md",
        f"Table D — {scenario} size / C-R-RC / schedule",
        table_d,
        ["method", "slice", "value", "n_rows", "feasibility", "completion_all"],
    )

    if scenario == "ablation":
        mapped = [{**row, "method": map_ablation_method(row.get("method"))} for row in rows]
        table_e = [_method_row(method, group) for method, group in sorted(_groups(mapped).items())]
        _md(out / "table_E_ablations.md", "Table E — Hybrid PPO ablations FULL/A1–A5", table_e, fields_a)
    else:
        _md(
            out / "table_E_ablations.md",
            "Table E — ablations (empty unless --scenario ablation)",
            [{"method": "not_this_scenario", "n_routes": 0, "n_seeds": 0, "n_rows": 0}],
            ["method", "n_routes", "n_seeds", "n_rows"],
        )

    if scenario in {"frvcpy_native", "nonlinear_sensitivity"}:
        native_note = (
            "official upstream e-VRO/frvcpy reference routes/objectives; exact only for native FRVCP; not EVRPTW-GR"
            if scenario == "frvcpy_native"
            else "native Montoya/FRVCP nonlinear charging sensitivity; not original EVRPTW-GR"
        )
        table_f = []
        for method, group in sorted(groups.items()):
            if method == "evrptwgr_surrogate_flag":
                continue
            feas = [row for row in group if row.get("feasible")]
            gaps = [
                float(row["optimality_gap_percent"])
                for row in group
                if row.get("optimality_gap_percent") is not None
            ]
            times = []
            n_nonfinite = 0
            for row in feas:
                if row.get("route_completion_time") is None:
                    continue
                value = float(row["route_completion_time"])
                if value != value or value in (float("inf"), float("-inf")):
                    n_nonfinite += 1
                    continue
                times.append(value)
            table_f.append(
                {
                    "block": "native_FRVCP" if scenario == "frvcpy_native" else "native_Montoya_FRVCP_nonlinear_sensitivity",
                    "method": method,
                    "n_routes": len({row.get("route_id") for row in group if row.get("method") != "evrptwgr_surrogate_flag"}),
                    "n_feasible": len(feas),
                    "n_finite_duration": len(times),
                    "n_nonfinite_duration": n_nonfinite,
                    "mean_duration_feasible": float(sum(times) / len(times)) if times else None,
                    "mean_gap_percent_vs_frvcpy_solver": float(sum(gaps) / len(gaps)) if gaps else None,
                    "n_gap_rows": len(gaps),
                    "note": native_note,
                }
            )
        if not table_f:
            table_f = [
                {
                    "block": "empty",
                    "method": None,
                    "n_routes": 0,
                    "n_feasible": 0,
                    "mean_duration_feasible": None,
                    "mean_gap_percent_vs_frvcpy_solver": None,
                    "n_gap_rows": 0,
                    "note": None,
                }
            ]
        title = (
            "Table F — native FRVCP (gaps only vs native frvcpy Solver; official upstream e-VRO/frvcpy reference routes/objectives; not EVRPTW-GR)"
            if scenario == "frvcpy_native"
            else "Table F — native Montoya/FRVCP nonlinear charging sensitivity (not original EVRPTW-GR)"
        )
        _md(
            out / "table_F_frvcpy.md",
            title,
            table_f,
            [
                "block",
                "method",
                "n_routes",
                "n_feasible",
                "n_finite_duration",
                "n_nonfinite_duration",
                "mean_duration_feasible",
                "mean_gap_percent_vs_frvcpy_solver",
                "n_gap_rows",
                "note",
            ],
        )
        route_rows = []
        for row in rows:
            if row.get("method") == "evrptwgr_surrogate_flag":
                continue
            route_rows.append(
                {
                    "block": table_f[0]["block"] if table_f else scenario,
                    "method": row.get("method"),
                    "route_id": row.get("route_id"),
                    "duration": row.get("route_completion_time"),
                    "gap_percent": row.get("optimality_gap_percent"),
                    "feasible": row.get("feasible"),
                    "reference_label": "official upstream e-VRO/frvcpy reference routes/objectives",
                    "note": native_note,
                }
            )
        if route_rows:
            _md(
                out / "table_F_frvcpy_routes.md",
                title + " — per-route records",
                route_rows,
                ["block", "method", "route_id", "duration", "gap_percent", "feasible", "reference_label", "note"],
            )
    else:
        _md(
            out / "table_F_frvcpy.md",
            "Table F — native FRVCP (empty unless --scenario frvcpy_native or nonlinear_sensitivity)",
            [{"block": "not_this_scenario", "method": None, "route_id": None, "duration": None, "gap_percent": None, "feasible": None, "note": None}],
            ["block", "method", "route_id", "duration", "gap_percent", "feasible", "note"],
        )

    table_g = []
    for method, group in sorted(groups.items()):
        timed = [row for row in group if row.get("runtime_s") is not None]
        s = summarize_method(timed, "runtime_s") if timed else {}
        table_g.append({"method": method, "n": s.get("n") or 0, "mean_runtime_s": s.get("mean"), "median_runtime_s": s.get("median")})
    _md(out / "table_G_runtime.md", f"Table G — {scenario} runtime", table_g, ["method", "n", "mean_runtime_s", "median_runtime_s"])

    if scenario == "soc_reserve":
        table_h = []
        for method, group in sorted(groups.items()):
            by_level: Dict[str, List[dict]] = defaultdict(list)
            for row in group:
                by_level[str(row.get("min_soc_fraction"))].append(row)
            for level, items in sorted(by_level.items(), key=lambda kv: float(kv[0]) if kv[0] not in {"None", ""} else 0.0):
                tagged = attach_feas_rate(items)
                table_h.append(
                    {
                        "method": method,
                        "min_soc_fraction": level,
                        "n_rows": len(items),
                        "feasibility": route_weighted_mean(tagged, "feas_rate"),
                        "parent_balanced_feasibility": parent_balanced_mean(tagged, "feas_rate"),
                        "completion_all": route_weighted_mean(items, "completion_time_all_routes"),
                        "parent_balanced_completion_all": parent_balanced_mean(items, "completion_time_all_routes"),
                    }
                )
        _md(
            out / "table_H_soc_reserve.md",
            "Table H — SOC reserve sensitivity (not used for model selection; not in main_test n)",
            table_h,
            [
                "method",
                "min_soc_fraction",
                "n_rows",
                "feasibility",
                "parent_balanced_feasibility",
                "completion_all",
                "parent_balanced_completion_all",
            ],
        )
    else:
        _md(
            out / "table_H_soc_reserve.md",
            "Table H — SOC reserve (empty unless --scenario soc_reserve)",
            [{"method": "not_this_scenario", "min_soc_fraction": None, "n_rows": 0}],
            ["method", "min_soc_fraction", "n_rows"],
        )

    table_i = []
    for method, group in sorted(groups.items()):
        reasons = Counter(
            str(row.get("reason") or ("feasible" if row.get("feasible") else "unknown")) for row in group
        )
        n_fail = sum(1 for row in group if not row.get("feasible"))
        table_i.append(
            {
                "method": method,
                "n_rows": len(group),
                "n_feasible": sum(1 for row in group if row.get("feasible")),
                "n_failed": n_fail,
                "fail_reasons": dict(reasons),
            }
        )
    reason_rows = []
    for row in table_i:
        if not row["fail_reasons"]:
            reason_rows.append({**{k: v for k, v in row.items() if k != "fail_reasons"}, "reason": None, "count": 0})
            continue
        for reason, count in sorted(row["fail_reasons"].items()):
            reason_rows.append(
                {
                    "method": row["method"],
                    "n_rows": row["n_rows"],
                    "n_feasible": row["n_feasible"],
                    "n_failed": row["n_failed"],
                    "reason": reason,
                    "count": count,
                }
            )
    _md(
        out / "table_I_failure_reasons.md",
        f"Table I — {scenario} failure reasons (failures retained in all-routes metrics)",
        reason_rows,
        ["method", "n_rows", "n_feasible", "n_failed", "reason", "count"],
    )
    print(f"wrote tables under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
