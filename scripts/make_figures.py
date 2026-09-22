"""Scientific figures from a single scenario. Not decorative. Never mix scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import CHECKPOINTS_DIR, RESULTS_DIR  # noqa: E402
from experiments.batch import read_jsonl_dicts  # noqa: E402
from experiments.isolation import require_scenario  # noqa: E402
from experiments.stats import (
    attach_feas_rate,
    dedupe_soc_greedy,
    exclude_non_paper_methods,
    map_ablation_method,
    parent_balanced_mean,
    route_weighted_mean,
)


def _load_rows(raw_dir: Path) -> list[dict]:
    rows = []
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*.jsonl")):
            rows.extend(read_jsonl_dicts(path))
    return rows


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "figures")
    parser.add_argument("--raw", type=Path, default=RESULTS_DIR / "raw")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--curves-root", type=Path, default=None, help="HybridPPO checkpoint root for learning curves")
    args = parser.parse_args(argv)
    scenario = require_scenario(args.scenario)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "README.md").write_text(
            "matplotlib is not installed; tables were still generated. `pip install matplotlib` to emit PNGs.\n",
            encoding="utf-8",
        )
        print("matplotlib missing; skipped figures")
        return 0

    rows = exclude_non_paper_methods(_load_rows(args.raw))
    rows = [row for row in rows if row.get("scenario") == scenario]
    if scenario == "soc_reserve":
        rows = dedupe_soc_greedy(rows)
    out = args.out / scenario
    out.mkdir(parents=True, exist_ok=True)

    if scenario == "main_test":
        fig, ax = plt.subplots(figsize=(6, 4))
        found = False
        curves_root = args.curves_root or (CHECKPOINTS_DIR / "HybridPPO")
        if curves_root.is_dir():
            for curves in sorted(curves_root.glob("seed_*/curves.jsonl")):
                updates, vals = [], []
                for line in curves.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "val_parent_balanced_feasibility" in row:
                        updates.append(row.get("update"))
                        vals.append(row["val_parent_balanced_feasibility"])
                    elif "val_feasibility" in row:
                        updates.append(row.get("update"))
                        vals.append(row["val_feasibility"])
                if updates:
                    ax.plot(updates, vals, label=curves.parent.name)
                    found = True
        if found:
            ax.set_xlabel("update")
            ax.set_ylabel("validation parent-balanced feasibility")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no HybridPPO curves.jsonl", ha="center")
        _save(fig, out / "hybrid_ppo_val_curves.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        by_m = defaultdict(list)
        for row in rows:
            if row.get("method"):
                by_m[row["method"]].append(row)
        names = sorted(by_m)
        if names:
            ax.bar(range(len(names)), [route_weighted_mean(attach_feas_rate(by_m[n]), "feas_rate") or 0.0 for n in names])
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("route-weighted feasibility")
        _save(fig, out / "feasibility_bars.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        if names:
            ax.bar(
                range(len(names)),
                [parent_balanced_mean(by_m[n], "completion_time_all_routes") or 0.0 for n in names],
            )
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("parent-balanced all-routes completion (H for failures)")
        _save(fig, out / "completion_all_bars.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        terrains = sorted({str(r.get("terrain") or r.get("terrain_variant") or "") for r in rows} - {""})
        methods = ("HybridPPO", "GreedyMinimumSufficientCharge")
        for method in methods:
            ys = []
            for terrain in terrains:
                items = [r for r in rows if r.get("method") == method and str(r.get("terrain") or r.get("terrain_variant")) == terrain]
                ys.append(parent_balanced_mean(items, "completion_time_all_routes") or 0.0)
            if terrains:
                ax.plot(range(len(terrains)), ys, marker="o", label=method)
        if terrains:
            ax.set_xticks(list(range(len(terrains))))
            ax.set_xticklabels(terrains)
            ax.set_ylabel("parent-balanced completion (H for failures)")
            ax.legend()
        _save(fig, out / "terrain_comparison.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        from collections import Counter

        reasons = Counter()
        for row in rows:
            if row.get("method") != "HybridPPO":
                continue
            if row.get("feasible"):
                reasons["feasible"] += 1
            else:
                reasons[str(row.get("reason") or "unknown")] += 1
        if reasons:
            labels = sorted(reasons)
            ax.bar(range(len(labels)), [reasons[k] for k in labels])
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("HybridPPO TEST rows")
        _save(fig, out / "hybrid_failure_reasons.png")

    elif scenario == "ablation":
        fig, ax = plt.subplots(figsize=(6, 4))
        mapped = [{**row, "method": map_ablation_method(row.get("method"))} for row in rows]
        by_m = defaultdict(list)
        for row in mapped:
            by_m[row["method"]].append(row)
        order = [name for name in ("FULL", "A1", "A2", "A3", "A4", "A5") if name in by_m]
        if order:
            ax.bar(range(len(order)), [parent_balanced_mean(by_m[n], "completion_time_all_routes") or 0.0 for n in order])
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order)
            ax.set_ylabel("parent-balanced completion (H for failures)")
        else:
            ax.text(0.5, 0.5, "no ablation rows", ha="center")
        _save(fig, out / "ablation_comparison.png")

    elif scenario == "soc_reserve":
        fig, ax = plt.subplots(figsize=(6, 4))
        by_m = defaultdict(list)
        for row in rows:
            by_m[str(row.get("method"))].append(row)
        for method, items in sorted(by_m.items()):
            levels = sorted({float(r.get("min_soc_fraction") or 0.0) for r in items})
            ys = []
            for level in levels:
                subset = [r for r in items if float(r.get("min_soc_fraction") or 0.0) == level]
                ys.append(route_weighted_mean(attach_feas_rate(subset), "feas_rate") or 0.0)
            ax.plot(levels, ys, marker="o", label=method)
        ax.set_xlabel("min SOC fraction")
        ax.set_ylabel("feasibility")
        ax.legend(fontsize=7)
        _save(fig, out / "soc_reserve.png")

    elif scenario == "frvcpy_native":
        fig, ax = plt.subplots(figsize=(6, 4))
        gaps = [r["optimality_gap_percent"] for r in rows if r.get("optimality_gap_percent") is not None]
        if gaps:
            ax.hist(gaps, bins=min(10, max(len(gaps), 1)))
            ax.set_xlabel("native FRVCP greedy gap vs frvcpy Solver (%)")
        else:
            ax.text(0.5, 0.5, "no native FRVCP gaps", ha="center")
        _save(fig, out / "frvcp_gap_hist.png")

    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"no default figure set for scenario={scenario}", ha="center")
        _save(fig, out / "placeholder.png")

    print(f"wrote figures under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
