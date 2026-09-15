"""Scientific figures from result files. Not decorative."""

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


def _load_rows() -> list[dict]:
    rows = []
    raw = RESULTS_DIR / "raw"
    if raw.is_dir():
        for path in sorted(raw.glob("*.jsonl")):
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
    args = parser.parse_args(argv)
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

    rows = _load_rows()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Train/val curves
    fig, ax = plt.subplots(figsize=(6, 4))
    found = False
    if CHECKPOINTS_DIR.is_dir():
        for curves in CHECKPOINTS_DIR.glob("**/curves.jsonl"):
            updates, vals = [], []
            for line in curves.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "val_completion_all" in row:
                    updates.append(row["update"])
                    vals.append(row["val_completion_all"])
            if updates:
                ax.plot(updates, vals, label=str(curves.parent.relative_to(CHECKPOINTS_DIR)))
                found = True
    if found:
        ax.set_xlabel("update")
        ax.set_ylabel("val completion time (all-routes, H for failures)")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no curves.jsonl yet", ha="center")
    _save(fig, out / "train_val_curves.png")

    # Feasibility bars
    fig, ax = plt.subplots(figsize=(7, 4))
    by_m = defaultdict(list)
    for row in rows:
        if "feasible" in row and row.get("method"):
            by_m[row["method"]].append(1.0 if row["feasible"] else 0.0)
    if by_m:
        names = sorted(by_m)
        ax.bar(range(len(names)), [sum(by_m[n]) / len(by_m[n]) for n in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("feasibility")
    _save(fig, out / "feasibility_bars.png")

    # Terrain paired differences Hybrid vs greedy
    fig, ax = plt.subplots(figsize=(6, 4))
    hybrid = [r for r in rows if r.get("method") == "HybridPPO" and r.get("terrain")]
    greedy = [r for r in rows if r.get("method") == "GreedyMinimumSufficientCharge" and r.get("terrain")]
    terrains = sorted({r["terrain"] for r in hybrid} & {r["terrain"] for r in greedy})
    if terrains:
        h_mean = []
        g_mean = []
        for t in terrains:
            hv = [r["route_completion_time"] for r in hybrid if r["terrain"] == t and r.get("feasible") and r.get("route_completion_time") is not None]
            gv = [r["route_completion_time"] for r in greedy if r["terrain"] == t and r.get("feasible") and r.get("route_completion_time") is not None]
            h_mean.append(sum(hv) / len(hv) if hv else 0.0)
            g_mean.append(sum(gv) / len(gv) if gv else 0.0)
        x = range(len(terrains))
        ax.plot(list(x), h_mean, marker="o", label="HybridPPO")
        ax.plot(list(x), g_mean, marker="s", label="GreedyMin")
        ax.set_xticks(list(x))
        ax.set_xticklabels(terrains)
        ax.set_ylabel("mean feasible completion time")
        ax.legend()
    _save(fig, out / "terrain_paired.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ab = defaultdict(list)
    for row in rows:
        m = str(row.get("method", ""))
        if m.startswith("HybridPPO_") and row.get("feasible") and row.get("route_completion_time") is not None:
            ab[m].append(row["route_completion_time"])
    if ab:
        names = sorted(ab)
        ax.bar(range(len(names)), [sum(ab[n]) / len(ab[n]) for n in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mean completion time")
    else:
        ax.text(0.5, 0.5, "no ablation rows", ha="center")
    _save(fig, out / "ablation_deltas.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    gaps = [r["optimality_gap_percent"] for r in rows if r.get("optimality_gap_percent") is not None]
    if gaps:
        ax.hist(gaps, bins=min(10, max(len(gaps), 1)))
        ax.set_xlabel("FRVCP optimality gap (%)")
    else:
        ax.text(0.5, 0.5, "no FRVCP gaps", ha="center")
    _save(fig, out / "frvcp_gap_hist.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    soc = [r.get("terminal_soc") for r in rows if r.get("terminal_soc") is not None]
    charge_t = [r.get("total_charging_time") for r in rows if r.get("total_charging_time") is not None]
    if soc and charge_t:
        ax.scatter(charge_t[: len(soc)], soc[: len(charge_t)], s=12)
        ax.set_xlabel("total charging time")
        ax.set_ylabel("terminal SOC")
    else:
        ax.text(0.5, 0.5, "no SOC/charge rows", ha="center")
    _save(fig, out / "soc_charge_scatter.png")
    print(f"wrote figures under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
