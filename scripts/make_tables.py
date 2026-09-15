"""Auto-generate tables A–G from result files. Never hand-edit numbers."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import RESULTS_DIR  # noqa: E402
from experiments.batch import read_jsonl_dicts  # noqa: E402
from experiments.stats import cluster_bootstrap_ci, summarize_method  # noqa: E402


def _load(raw_dir: Path) -> List[dict]:
    rows = []
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*.jsonl")):
            rows.extend(read_jsonl_dicts(path))
    summary = RESULTS_DIR / "summaries" / "all_rows.jsonl"
    if not rows and summary.is_file():
        rows = read_jsonl_dicts(summary)
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


def _method_groups(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("method"):
            out[str(row["method"])].append(row)
    return dict(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RESULTS_DIR / "raw")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "tables")
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args(argv)
    from experiments.isolation import filter_scenario, require_scenario

    scenario = require_scenario(args.scenario)
    rows = _load(args.raw)
    rows = [r for r in rows if r.get("scenario") == scenario]
    if scenario == "main_test":
        rows = [r for r in rows if r.get("split") == "test"]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    test_like = [
        r
        for r in rows
        if r.get("method")
        not in {
            None,
            "frvcpy_Solver",
            "FRVCPGreedyMin",
            "FRVCPGreedyFull",
            "LabelSettingRCSPP",
            "evrptwgr_surrogate_flag",
        }
    ]
    primary = list(test_like)
    groups = _method_groups(primary)

    table_a = []
    for method, group in sorted(groups.items()):
        feas = [r for r in group if r.get("feasible") and r.get("route_completion_time") is not None]
        all_rows = [r for r in group if r.get("completion_time_all_routes") is not None]
        s_f = summarize_method(feas, "route_completion_time") if feas else {}
        s_a = summarize_method(all_rows, "completion_time_all_routes") if all_rows else {}
        table_a.append(
            {
                "method": method,
                "n": s_f.get("n") or 0,
                "mean_feasible": s_f.get("mean"),
                "median_feasible": s_f.get("median"),
                "sd_feasible": s_f.get("sd"),
                "ci95_lo_feasible": s_f.get("ci95_lo"),
                "ci95_hi_feasible": s_f.get("ci95_hi"),
                "mean_all_H_for_failures": s_a.get("mean"),
                "ci95_lo_all": s_a.get("ci95_lo"),
                "ci95_hi_all": s_a.get("ci95_hi"),
            }
        )
    fields_a = [
        "method",
        "n",
        "mean_feasible",
        "median_feasible",
        "sd_feasible",
        "ci95_lo_feasible",
        "ci95_hi_feasible",
        "mean_all_H_for_failures",
        "ci95_lo_all",
        "ci95_hi_all",
    ]
    _md(out / "table_A_completion_time.md", "Table A — route completion time (cluster CI)", table_a, fields_a)

    table_b = []
    for method, group in sorted(groups.items()):
        tagged = [{**r, "feas_rate": 1.0 if r.get("feasible") else 0.0} for r in group]
        ci = cluster_bootstrap_ci(tagged, "feas_rate") if tagged else {}
        table_b.append(
            {
                "method": method,
                "n": len(group),
                "feasibility": ci.get("mean"),
                "ci95_lo": ci.get("lo"),
                "ci95_hi": ci.get("hi"),
            }
        )
    _md(out / "table_B_feasibility.md", "Table B — feasibility rates (cluster CI)", table_b, ["method", "n", "feasibility", "ci95_lo", "ci95_hi"])

    table_c = []
    for method, group in sorted(groups.items()):
        by_t: Dict[str, List[dict]] = defaultdict(list)
        for row in group:
            by_t[str(row.get("terrain") or row.get("terrain_variant") or "")].append(row)
        for terrain, items in sorted(by_t.items()):
            if not terrain:
                continue
            feas = [r for r in items if r.get("feasible") and r.get("route_completion_time") is not None]
            s = summarize_method(feas, "route_completion_time") if feas else {}
            table_c.append({"method": method, "terrain": terrain, "n": s.get("n") or 0, "mean": s.get("mean"), "ci95_lo": s.get("ci95_lo"), "ci95_hi": s.get("ci95_hi")})
    _md(out / "table_C_terrain.md", "Table C — terrain (paired siblings share customer_ids)", table_c, ["method", "terrain", "n", "mean", "ci95_lo", "ci95_hi"])

    table_d = []
    for method, group in sorted(groups.items()):
        for key in ("network_group", "customer_distribution", "schedule_type"):
            buckets: Dict[str, List[dict]] = defaultdict(list)
            for row in group:
                buckets[str(row.get(key, ""))].append(row)
            for label, items in sorted(buckets.items()):
                if not label or label == "None":
                    continue
                feas = [r for r in items if r.get("feasible") and r.get("route_completion_time") is not None]
                s = summarize_method(feas, "route_completion_time") if feas else {}
                table_d.append({"method": method, "slice": key, "value": label, "n": s.get("n") or 0, "mean": s.get("mean")})
    _md(out / "table_D_size_family.md", "Table D — size / C-R-RC / schedule", table_d, ["method", "slice", "value", "n", "mean"])

    ablations = [r for r in rows if scenario == "ablation" and str(r.get("method", "")).startswith("HybridPPO_")]
    if scenario == "ablation":
        for row in rows:
            if row.get("method") == "HybridPPO":
                ablations.append({**row, "method": "HybridPPO_FULL"})
    table_e = []
    for method, group in sorted(_method_groups(ablations).items()):
        feas = [r for r in group if r.get("feasible") and r.get("route_completion_time") is not None]
        s = summarize_method(feas, "route_completion_time") if feas else {}
        table_e.append({"method": method, "n": s.get("n") or 0, "mean": s.get("mean"), "ci95_lo": s.get("ci95_lo"), "ci95_hi": s.get("ci95_hi")})
    if not table_e:
        table_e = [{"method": "no_ablation_rows", "n": 0, "mean": None, "ci95_lo": None, "ci95_hi": None}]
    _md(out / "table_E_ablations.md", "Table E — ablations A1–A5", table_e, ["method", "n", "mean", "ci95_lo", "ci95_hi"])

    frv = [r for r in rows if r.get("split") == "native_frvcp" or r.get("method") in {"frvcpy_Solver", "FRVCPGreedyMin", "FRVCPGreedyFull"}]
    nonlinear = [r for r in rows if r.get("equivalent_to_evrptwgr") == "not_original_evrptwgr"]
    table_f = []
    for row in frv:
        table_f.append(
            {
                "block": "native_FRVCP_exact",
                "method": row.get("method"),
                "route_id": row.get("route_id"),
                "duration": row.get("route_completion_time"),
                "gap_percent": row.get("optimality_gap_percent"),
                "feasible": row.get("feasible"),
                "note": row.get("equivalent_to_evrptwgr"),
            }
        )
    for row in nonlinear:
        table_f.append(
            {
                "block": "EVRPTW-GR_Montoya_sensitivity_not_exact",
                "method": row.get("method"),
                "route_id": row.get("route_id"),
                "duration": row.get("route_completion_time"),
                "gap_percent": None,
                "feasible": row.get("feasible"),
                "note": "not original EVRPTW-GR",
            }
        )
    if not table_f:
        table_f = [{"block": "empty", "method": None, "route_id": None, "duration": None, "gap_percent": None, "feasible": None, "note": None}]
    _md(
        out / "table_F_frvcpy_nonlinear.md",
        "Table F — native FRVCP (exact) vs Montoya sensitivity (not exact, not official)",
        table_f,
        ["block", "method", "route_id", "duration", "gap_percent", "feasible", "note"],
    )

    table_g = []
    for method, group in sorted(groups.items()):
        timed = [r for r in group if r.get("runtime_s") is not None]
        s = summarize_method(timed, "runtime_s") if timed else {}
        table_g.append({"method": method, "n": s.get("n") or 0, "mean_runtime_s": s.get("mean"), "median_runtime_s": s.get("median")})
    _md(out / "table_G_runtime.md", "Table G — runtime", table_g, ["method", "n", "mean_runtime_s", "median_runtime_s"])
    print(f"wrote tables under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
