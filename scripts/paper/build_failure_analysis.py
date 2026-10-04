"""Deeper FA-HPPO failure analysis from frozen V3 raw rows only.

Does not replay TEST policies or reconstruct unrecorded trajectories.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v3_hppo"))
from common import SEEDS, V3, load_json  # noqa: E402

RAW = V3 / "raw" / "synthcharge_test.jsonl"
OUT = ROOT / "results_paper" / "failure_analysis"
FIG = ROOT / "results_paper" / "figures" / "appendix" / "figA05_failure_consistency.png"


def main() -> None:
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    hybrid = [r for r in rows if r["method"] == "HybridPPO"]
    by_route = defaultdict(list)
    for r in hybrid:
        by_route[r["route_id"]].append(r)

    fail_counts = []
    route_records = []
    for rid, rr in sorted(by_route.items()):
        n_fail = sum(1 for r in rr if not r["feasible"])
        fail_counts.append(n_fail)
        if n_fail == 0:
            continue
        fails = [r for r in rr if not r["feasible"]]
        oks = [r for r in rr if r["feasible"]]
        sample = fails[0]
        route_records.append(
            {
                "route_id": rid,
                "layout": sample.get("layout"),
                "length_bin": sample.get("length_bin"),
                "charge_class": sample.get("charge_class"),
                "n_customers": sample.get("n_customers"),
                "certificate_station_visits": sample.get("certificate_station_visits"),
                "n_seeds_fail": n_fail,
                "fail_seeds": sorted(int(r["seed"]) for r in fails),
                "ok_seeds": sorted(int(r["seed"]) for r in oks),
                "reason": sample.get("reason"),
                "mean_n_station_visits_fail": float(np.mean([r.get("n_station_visits") or 0 for r in fails])),
                "mean_terminal_soc_fail": float(np.mean([r["terminal_soc"] for r in fails if r.get("terminal_soc") is not None])),
            }
        )

    conc = Counter(fail_counts)
    all5 = [r for r in route_records if r["n_seeds_fail"] == 5]
    unique1 = [r for r in route_records if r["n_seeds_fail"] == 1]
    by_seed_fails = Counter()
    for r in hybrid:
        if not r["feasible"]:
            by_seed_fails[int(r["seed"])] += 1

    findings = {
        "label": "Frozen V3 raw failure analysis — no TEST replay",
        "n_routes": len(by_route),
        "n_route_seed_failures": sum(1 for r in hybrid if not r["feasible"]),
        "routes_with_any_failure": len(route_records),
        "routes_failing_all_5_seeds": len(all5),
        "routes_failing_exactly_1_seed": len(unique1),
        "fail_count_histogram": {str(k): conc.get(k, 0) for k in range(6)},
        "failures_per_seed": {str(s): by_seed_fails.get(s, 0) for s in SEEDS},
        "seed_46_vs_45": {
            "seed_45_failures": by_seed_fails.get(45, 0),
            "seed_46_failures": by_seed_fails.get(46, 0),
            "delta_46_minus_45": by_seed_fails.get(46, 0) - by_seed_fails.get(45, 0),
        },
        "reason_counts": dict(Counter(r.get("reason") for r in hybrid if not r["feasible"])),
        "all_seeds_fail_routes": [r["route_id"] for r in all5],
        "unavailable": [
            "Pre-failure action/SOC trajectories were not stored in V3 raw rows.",
            "Causal step-level diagnosis would require forbidden TEST replay.",
        ],
        "answers": {
            "few_hard_routes": (
                f"{len(all5)} routes fail for all five seeds; "
                f"{len(route_records)} routes have ≥1 seed failure out of {len(by_route)}."
            ),
            "same_routes_all_seeds": f"{len(all5)} routes fail for every seed (see all_seeds_fail_routes).",
            "layout_length": "See tableA03 / per-route records; failures are charging_required only in frozen rows inspected.",
            "seed_46_vs_45": (
                f"Seed 46 has {by_seed_fails.get(46, 0)} route failures vs "
                f"{by_seed_fails.get(45, 0)} for seed 45."
            ),
        },
        "routes": route_records,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FINDINGS.json").write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# FA-HPPO failure analysis (frozen V3 raw)",
        "",
        "**No TEST replay. No unrecorded trajectory reconstruction.**",
        "",
        f"- Routes: {findings['n_routes']}",
        f"- Route×seed failures: {findings['n_route_seed_failures']}",
        f"- Routes with any failure: {findings['routes_with_any_failure']}",
        f"- Routes failing all 5 seeds: {findings['routes_failing_all_5_seeds']}",
        f"- Routes failing exactly 1 seed: {findings['routes_failing_exactly_1_seed']}",
        f"- Failures per seed: {findings['failures_per_seed']}",
        f"- Seed 46 vs 45: {findings['answers']['seed_46_vs_45']}",
        f"- Reasons: {findings['reason_counts']}",
        "",
        "## Answers",
        "",
    ]
    for k, v in findings["answers"].items():
        lines.append(f"- **{k}:** {v}")
    lines += [
        "",
        "## Unavailable without TEST replay",
        "",
    ]
    lines += [f"- {u}" for u in findings["unavailable"]]
    lines += [
        "",
        "Publication table: `results_paper/tables/tableA03_failure_routes.*`",
        "Figure: `results_paper/figures/appendix/figA05_failure_consistency.png`",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Ensure appendix figure exists (idempotent with builder)
    if not FIG.is_file():
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        xs = np.arange(0, 6)
        ys = [conc.get(int(x), 0) for x in xs]
        ax.bar(xs, ys, color="#0072B2")
        ax.set_xlabel("Number of FA-HPPO seeds failing route")
        ax.set_ylabel("Number of routes")
        ax.set_title("Frozen-raw failure consistency")
        fig.tight_layout()
        FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(json.dumps({"wrote": str(OUT), "n_fail_routes": len(route_records), "all5": len(all5)}, indent=2))


if __name__ == "__main__":
    main()
