"""Cluster-aware analysis of result JSONL files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import RESULTS_DIR  # noqa: E402
from experiments.batch import read_jsonl_dicts  # noqa: E402
from experiments.stats import cluster_bootstrap_ci, holm, paired_parent_diff, permutation_pvalue, summarize_method  # noqa: E402
from routing.serialize import canonical_dumps  # noqa: E402


def load_all_raw(raw_dir: Path) -> List[dict]:
    rows = []
    if not raw_dir.is_dir():
        return rows
    for path in sorted(raw_dir.glob("*.jsonl")):
        for row in read_jsonl_dicts(path):
            row["_source"] = path.name
            rows.append(row)
    return rows


def _group(rows: List[dict], key: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, ""))].append(row)
    return dict(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RESULTS_DIR / "raw")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "summaries")
    args = parser.parse_args(argv)
    rows = load_all_raw(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "all_rows.jsonl").write_text(
        "\n".join(canonical_dumps({k: v for k, v in row.items() if k != "_source"}) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    by_method = _group([r for r in rows if r.get("method")], "method")
    summaries = []
    for method, group in sorted(by_method.items()):
        feas = [r for r in group if "feasible" in r]
        if not feas:
            continue
        feas_rows = [
            {
                **r,
                "feas_rate": 1.0 if r.get("feasible") else 0.0,
                "completion_time_all_routes": float(r.get("completion_time_all_routes") or r.get("horizon") or 0.0),
            }
            for r in feas
        ]
        feasible_only = [r for r in feas_rows if r.get("feasible") and r.get("route_completion_time") is not None]
        all_obj = summarize_method(feas_rows, "completion_time_all_routes") if feas_rows else {}
        feas_ci = cluster_bootstrap_ci(feas_rows, "feas_rate") if feas_rows else {}
        time_ci = summarize_method(feasible_only, "route_completion_time") if feasible_only else {}
        runtime = summarize_method(feas_rows, "runtime_s") if any("runtime_s" in r for r in feas_rows) else {}
        summaries.append(
            {
                "method": method,
                "n": len(feas_rows),
                "feasibility_mean": feas_ci.get("mean"),
                "feasibility_ci95_lo": feas_ci.get("lo"),
                "feasibility_ci95_hi": feas_ci.get("hi"),
                "time_feasible_mean": time_ci.get("mean"),
                "time_feasible_ci95_lo": time_ci.get("ci95_lo"),
                "time_feasible_ci95_hi": time_ci.get("ci95_hi"),
                "time_all_mean": all_obj.get("mean"),
                "time_all_ci95_lo": all_obj.get("ci95_lo"),
                "time_all_ci95_hi": all_obj.get("ci95_hi"),
                "runtime_mean": runtime.get("mean"),
            }
        )
    if summaries:
        keys = list(summaries[0].keys())
        with (args.out / "method_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)

    hybrid = by_method.get("HybridPPO", [])
    tests = []
    if hybrid:
        for other_name, other in by_method.items():
            if other_name == "HybridPPO":
                continue
            if not other or "route_completion_time" not in other[0]:
                continue
            diffs = paired_parent_diff(hybrid, other, "completion_time_all_routes")
            p = permutation_pvalue(list(diffs.values()))
            tests.append((other_name, p))
        adjusted = holm(tests)
        with (args.out / "paired_holm.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["comparison", "p_raw", "p_holm"])
            writer.writeheader()
            for name, p, adj in adjusted:
                writer.writerow({"comparison": f"HybridPPO vs {name}", "p_raw": p, "p_holm": adj})
    print(f"analyzed {len(rows)} rows, {len(summaries)} methods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
