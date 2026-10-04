"""Build publication-facing results_paper/ (PNG figures + tables) from frozen data.

READ ONLY for TEST evidence. Case-study JSON is a VAL illustration (not TEST).
Never trains final models, evaluates TEST, or modifies frozen raw rows.

Official single publication-figure generator. PNG only.
All figures live flat in results_paper/figures/ (no appendix/).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3_hppo"))

from experiments.stats import hierarchical_bootstrap_ci, mean_sd_across_training_seeds  # noqa: E402
from common import SEEDS, V3, lf_sha256, load_json  # noqa: E402

RAW = V3 / "raw" / "synthcharge_test.jsonl"
STATS = V3 / "statistics"
OUT = ROOT / "results_paper"
FIG = OUT / "figures"
TAB = OUT / "tables"
CASE = OUT / "case_study" / "illustrative_val_episode.json"
ENVELOPE_SUMMARY = ROOT / "results" / "development" / "envelope_ablation" / "SUMMARY.json"
N_BOOT = 5000
RNG = 20261003
DPI = 600

ORDER_BASE = ["HybridPPO", "OneStepLookahead", "GreedyFullCharge", "GreedyMinimumSufficientCharge"]
AMOUNT_ORDER = ["HybridPPO", "FA-HPPO-Max", "FA-HPPO-Min"]
LABEL = {
    "HybridPPO": "FA-HPPO",
    "FA-HPPO-Max": "FA-HPPO-Max",
    "FA-HPPO-Min": "FA-HPPO-Min",
    "GreedyMinimumSufficientCharge": "Greedy Min",
    "GreedyFullCharge": "Greedy Full",
    "OneStepLookahead": "Lookahead",
}
COLOR = {
    "HybridPPO": "#0072B2",
    "FA-HPPO-Max": "#56B4E9",
    "FA-HPPO-Min": "#CC79A7",
    "OneStepLookahead": "#D55E00",
    "GreedyFullCharge": "#009E73",
    "GreedyMinimumSufficientCharge": "#E69F00",
}
LS = {
    "HybridPPO": "-",
    "OneStepLookahead": "--",
    "GreedyFullCharge": "-.",
    "GreedyMinimumSufficientCharge": ":",
}
ABL_ORDER = [("B0", "B0"), ("B1", "B1"), ("B3", "B3"), ("B2", "B2")]
MC_P_FLOOR = 5e-5

FIGS = [
    "fig01_usecase_route",
    "fig02_method_schematic",
    "fig03_soc_envelope",
    "fig04_illustrative_soc",
    "fig05_main_feasibility",
    "fig06_main_completion",
    "fig07_charging_required",
    "fig08_performance_profiles",
    "fig09_by_layout",
    "fig10_by_length",
    "fig11_difficulty_heatmap",
    "fig12_gain_vs_lookahead",
    "fig13_ablation_bars",
    "fig14_amount_policies",
    "fig15_charging_effort",
    "fig16_runtime",
    "fig17_failure_by_regime",
    "fig18_envelope_variants",
    "fig19_charge_decision",
    "fig20_frvcp_reference",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows() -> list[dict]:
    return [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]


def method_rows(rows, method, seed=None):
    out = [r for r in rows if r["method"] == method]
    if seed is not None:
        out = [r for r in out if r.get("seed") == seed]
    return out


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "NA"
    return f"{v:.{d}f}"


def fmt_p(p) -> str:
    if p is None:
        return "NA"
    p = float(p)
    if p <= MC_P_FLOOR + 1e-15:
        return "<5e-5"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.4f}"


def summarize_learned(rows, method):
    learned = [{**r, "feasible_f": float(bool(r["feasible"]))} for r in method_rows(rows, method)]
    feas = mean_sd_across_training_seeds(learned, "feasible_f")
    comp = mean_sd_across_training_seeds(learned, "completion_time_all_routes")
    feas_ci = hierarchical_bootstrap_ci(learned, "feasible_f", n_boot=N_BOOT, seed=RNG)
    comp_ci = hierarchical_bootstrap_ci(learned, "completion_time_all_routes", n_boot=N_BOOT, seed=RNG)
    return {
        "feasibility_mean": feas["mean_across_seeds"],
        "feasibility_ci95": [feas_ci["lo"], feas_ci["hi"]],
        "completion_all_mean": comp["mean_across_seeds"],
        "completion_all_ci95": [comp_ci["lo"], comp_ci["hi"]],
        "per_seed_feasibility": {int(k): v["mean"] for k, v in feas["per_seed"].items()},
        "per_seed_completion": {int(k): v["mean"] for k, v in comp["per_seed"].items()},
        "runtime_s": mean(r.get("runtime_s") for r in method_rows(rows, method)),
    }


def summarize_baseline(rows, method):
    subset = method_rows(rows, method)
    return {
        "feasibility_mean": mean(float(r["feasible"]) for r in subset),
        "feasibility_ci95": None,
        "completion_all_mean": mean(r["completion_time_all_routes"] for r in subset),
        "completion_all_ci95": None,
        "per_seed_feasibility": None,
        "per_seed_completion": None,
        "runtime_s": mean(r.get("runtime_s") for r in subset),
    }


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


def save_png(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cleanup_figures() -> None:
    keep = {f"{s}.png" for s in FIGS}
    FIG.mkdir(parents=True, exist_ok=True)
    for path in list(FIG.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg"} and path.name not in keep:
            path.unlink()
    appendix = FIG / "appendix"
    if appendix.is_dir():
        for path in appendix.rglob("*"):
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
        try:
            shutil.rmtree(appendix, ignore_errors=True)
        except OSError:
            pass
        # Last resort on Windows/OneDrive locks: leave empty dir only if undeletable
        if appendix.is_dir() and not any(appendix.rglob("*")):
            try:
                appendix.rmdir()
            except OSError:
                pass


def _vbar(ax, methods, values, cis, *, ylabel, ylim, label_fmt, as_pct=False):
    """Vertical bars with CI whiskers; value labels sit clearly above the upper whisker."""
    xs = np.arange(len(methods))
    scale = 100.0 if as_pct else 1.0
    span = ylim[1] - ylim[0]
    for i, m in enumerate(methods):
        mu = float(values[m]) * scale
        ax.bar(i, mu, width=0.62, color=COLOR[m], edgecolor="none", zorder=2)
        top = mu
        if cis.get(m) is not None:
            lo = float(cis[m][0]) * scale
            hi = float(cis[m][1]) * scale
            ax.errorbar(
                i,
                mu,
                yerr=[[max(0.0, mu - lo)], [max(0.0, hi - mu)]],
                fmt="none",
                ecolor="#222222",
                elinewidth=0.9,
                capsize=2.8,
                zorder=3,
            )
            top = hi
        ax.text(
            i,
            top + span * 0.055,
            label_fmt.format(mu),
            ha="center",
            va="bottom",
            fontsize=8,
            zorder=5,
            clip_on=False,
        )
    ax.set_xticks(xs, [LABEL[m] for m in methods])
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(-0.55, len(methods) - 0.45)


# ---------------------------------------------------------------------------
# Fig. 1 — clear use-case route map
# ---------------------------------------------------------------------------
def fig01_usecase_route():
    if not CASE.is_file():
        raise SystemExit("missing case study JSON; run scripts/paper/record_illustrative_val_episode.py")
    ep = load_json(CASE)
    nodes, customers, depot = ep["nodes"], ep["customer_ids"], ep["depot_id"]
    stations, path = ep["station_ids"], ep["path_nodes"]
    visited = [n for n in path if n in stations]

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    # Fixed customer sequence (no stations)
    seq = [depot] + customers + [depot]
    ax.plot(
        [nodes[i]["x"] for i in seq],
        [nodes[i]["y"] for i in seq],
        ls="--",
        color="#8A8A8A",
        lw=1.15,
        zorder=1,
    )
    # Executed path (under markers)
    px = [nodes[i]["x"] for i in path]
    py = [nodes[i]["y"] for i in path]
    ax.plot(px, py, color=COLOR["HybridPPO"], lw=2.2, zorder=2, solid_capstyle="round")

    # Stations
    for sid in stations:
        n = nodes[sid]
        used = sid in visited
        ax.scatter(
            n["x"],
            n["y"],
            marker="^",
            s=160 if used else 100,
            color="#D55E00" if used else "#C8C8C8",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        ax.annotate(
            sid,
            xy=(n["x"], n["y"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="#D55E00" if used else "#666666",
            fontweight="bold" if used else "normal",
            zorder=6,
        )

    for i, cid in enumerate(customers, start=1):
        n = nodes[cid]
        # white halo so path/dashed lines never cut through numbers
        ax.scatter(n["x"], n["y"], s=150, facecolors="white", edgecolors="none", zorder=7)
        ax.scatter(n["x"], n["y"], s=100, facecolors="white", edgecolors="#111111", linewidths=1.2, zorder=8)
        ax.text(n["x"], n["y"], str(i), ha="center", va="center", fontsize=8, fontweight="bold", zorder=9)

    nd = nodes[depot]
    ax.scatter(nd["x"], nd["y"], marker="s", s=140, facecolors="white", edgecolors="none", zorder=9)
    ax.scatter(nd["x"], nd["y"], marker="s", s=115, color="#111111", zorder=10)
    ax.annotate("Depot", xy=(nd["x"], nd["y"]), xytext=(8, 8), textcoords="offset points", fontsize=8, color="#111111", zorder=11)

    xs = [nodes[i]["x"] for i in nodes] 
    ys = [nodes[i]["y"] for i in nodes]
    pad = 0.07
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#111111", markersize=8, label="Depot"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#111111", markersize=8, label="Customer (visit order)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#D55E00", markersize=9, label="Charging station (used)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#C8C8C8", markersize=8, label="Charging station (unused)"),
        Line2D([0], [0], ls="--", color="#8A8A8A", lw=1.2, label="Fixed customer order"),
        Line2D([0], [0], color=COLOR["HybridPPO"], lw=2.0, label="Executed path (with charges)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=7.5, borderaxespad=0.2)
    fig.tight_layout()
    save_png(fig, "fig01_usecase_route")


# ---------------------------------------------------------------------------
# Fig. 2 — method schematic
# ---------------------------------------------------------------------------
def fig02_method_schematic():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F3F7FB", ec="#0072B2"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.2, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#333333"))

    box(0.3, 1.5, 1.8, 1.2, "Route state\n(SOC, time,\nnext customer)", fc="#F5F5F5", ec="#555555")
    box(2.6, 2.35, 2.3, 1.2, "Discrete actor\nCONTINUE / station", fc="#E8F3FA", ec="#0072B2")
    box(2.6, 0.55, 2.3, 1.2, "If CHARGE:\ncontinuous $u\\in[0,1]$\n(Beta policy)", fc="#FFF3E8", ec="#D55E00")
    box(5.5, 1.5, 2.2, 1.2, "SOC envelope\n$\\mathrm{lower}\\!\\to\\!\\mathrm{upper}$\ntarget from $u$", fc="#EEF8F3", ec="#009E73")
    box(8.2, 1.5, 1.5, 1.2, "Environment\nstep", fc="#F5F5F5", ec="#555555")

    arrow(2.1, 2.1, 2.6, 2.8)
    arrow(2.1, 2.0, 2.6, 1.15)
    arrow(4.9, 2.9, 5.5, 2.2)
    arrow(4.9, 1.15, 5.5, 1.9)
    arrow(7.7, 2.1, 8.2, 2.1)
    ax.text(5.0, 0.2, "FA-HPPO: hybrid discrete station choice + continuous charge amount inside a feasibility envelope", ha="center", fontsize=8, color="#333333")
    fig.tight_layout()
    save_png(fig, "fig02_method_schematic")


# ---------------------------------------------------------------------------
# Fig. 3 — SOC envelope schematic
# ---------------------------------------------------------------------------
def fig03_soc_envelope():
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    arrival, lo, hi, u = 0.18, 0.42, 0.86, 0.78
    tgt = lo + u * (hi - lo)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.add_patch(Rectangle((lo, 0.32), hi - lo, 0.28, facecolor="#56B4E9", alpha=0.45, edgecolor="#0072B2", lw=1.6))
    ax.axvline(arrival, color="#666666", ls="--", lw=1.3, ymin=0.28, ymax=0.72)
    ax.axvline(lo, color="#0072B2", lw=1.7, ymin=0.28, ymax=0.72)
    ax.axvline(hi, color="#D55E00", lw=1.7, ymin=0.28, ymax=0.72)
    ax.scatter([tgt], [0.46], s=80, color="#111111", zorder=5, edgecolors="white", linewidths=0.6)
    ax.text(arrival, 0.84, r"arrival", ha="center", fontsize=8, color="#555555")
    ax.text(lo, 0.84, r"lower", ha="center", fontsize=8, color="#0072B2")
    ax.text(hi, 0.84, r"upper", ha="center", fontsize=8, color="#D55E00")
    ax.text(tgt, 0.12, r"target $= lower + u\,(upper-lower)$", ha="center", fontsize=8)
    ax.set_xlabel("State of charge (SOC)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.tight_layout()
    save_png(fig, "fig03_soc_envelope")


# ---------------------------------------------------------------------------
# Fig. 4 — illustrative SOC trajectory
# ---------------------------------------------------------------------------
def fig04_illustrative_soc():
    ep = load_json(CASE)
    decisions, path = ep["decisions"], ep["path_nodes"]
    fig, ax = plt.subplots(figsize=(6.8, 3.25))
    points = [(0.0, 100.0)]
    cursor = 0
    labels = ["Depot"]
    for d in decisions:
        if d.get("action") == "CONTINUE":
            cursor += 1
            points.append((float(cursor), 100 * float(d["soc_after"])))
            labels.append(d.get("location_after", ""))
        elif d.get("action") == "CHARGE":
            cursor += 1
            points.append((float(cursor), 100 * float(d.get("soc_arrival", d["soc_before_decision"]))))
            points.append((float(cursor), 100 * float(d.get("soc_departure", d["soc_after"]))))
            labels.append(d.get("station_id", "S"))
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=COLOR["HybridPPO"], lw=2.0)
    for d in decisions:
        if d.get("action") != "CHARGE":
            continue
        try:
            xi = path.index(d["station_id"])
        except (ValueError, KeyError):
            continue
        lo, hi = 100 * float(d["soc_lower"]), 100 * float(d["soc_upper"])
        ax.fill_between([xi - 0.22, xi + 0.22], lo, hi, color="#56B4E9", alpha=0.35, zorder=1)
        ax.scatter([xi], [100 * float(d.get("soc_departure", d["soc_after"]))], marker="^", s=90, color="#D55E00", zorder=4, edgecolors="white", linewidths=0.5)
        ax.text(xi, min(104, hi + 4), d["station_id"], ha="center", fontsize=8, color="#D55E00", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_xlabel("Visit index along executed path")
    ax.set_ylabel("SOC (%)")
    ax.set_xticks(range(0, int(max(xs)) + 1, 2))
    fig.tight_layout()
    save_png(fig, "fig04_illustrative_soc")


# ---------------------------------------------------------------------------
# Fig. 5–7 — primary TEST bars
# ---------------------------------------------------------------------------
def fig05_main_feasibility(rows):
    summaries = {m: summarize_learned(rows, m) if m == "HybridPPO" else summarize_baseline(rows, m) for m in ORDER_BASE}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _vbar(
        ax,
        ORDER_BASE,
        {m: summaries[m]["feasibility_mean"] for m in ORDER_BASE},
        {m: summaries[m]["feasibility_ci95"] for m in ORDER_BASE},
        ylabel="Route feasibility (%)",
        ylim=(0, 118),
        label_fmt="{:.1f}",
        as_pct=True,
    )
    fig.tight_layout()
    save_png(fig, "fig05_main_feasibility")
    return summaries


def fig06_main_completion(rows, summaries):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _vbar(
        ax,
        ORDER_BASE,
        {m: summaries[m]["completion_all_mean"] for m in ORDER_BASE},
        {m: summaries[m]["completion_all_ci95"] for m in ORDER_BASE},
        ylabel="Failure-retaining completion time",
        ylim=(0, 9.2),
        label_fmt="{:.2f}",
        as_pct=False,
    )
    fig.tight_layout()
    save_png(fig, "fig06_main_completion")


def fig07_charging_required(rows):
    charge = [r for r in rows if r.get("charge_class") == "charging_required"]
    ch_s = {m: summarize_learned(charge, m) if m == "HybridPPO" else summarize_baseline(charge, m) for m in ORDER_BASE}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _vbar(
        ax,
        ORDER_BASE,
        {m: ch_s[m]["feasibility_mean"] for m in ORDER_BASE},
        {m: ch_s[m]["feasibility_ci95"] for m in ORDER_BASE},
        ylabel="Feasibility on charging-required routes (%)",
        ylim=(0, 118),
        label_fmt="{:.1f}",
        as_pct=True,
    )
    fig.tight_layout()
    save_png(fig, "fig07_charging_required")
    return ch_s


# ---------------------------------------------------------------------------
# Fig. 8 — performance profiles
# ---------------------------------------------------------------------------
def _profile_for_subset(subset, grid):
    by_route = defaultdict(list)
    for r in subset:
        by_route[r["route_id"]].append(r)
    route_ok_time = []
    for rr in by_route.values():
        vals = []
        for t in grid:
            vals.append(mean(float(bool(r["feasible"]) and float(r["completion_time_all_routes"]) <= t) for r in rr))
        route_ok_time.append(vals)
    return np.mean(np.asarray(route_ok_time), axis=0)


def fig08_performance_profiles(rows):
    grid = np.linspace(2.0, 10.0, 161)
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for m in ORDER_BASE:
        ys = 100.0 * _profile_for_subset(method_rows(rows, m), grid)
        ax.plot(grid, ys, color=COLOR[m], ls=LS[m], lw=2.0, label=LABEL[m])
    ax.set_xlabel("Completion-time threshold")
    ax.set_ylabel("Feasible routes finished by threshold (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_png(fig, "fig08_performance_profiles")


# ---------------------------------------------------------------------------
# Fig. 9–10 — stratified feasibility (grouped bars)
# ---------------------------------------------------------------------------
def _grouped_feasibility(rows, key, levels):
    """Mean feasibility (%) by stratum; HybridPPO averaged over all its rows."""
    out = {m: [] for m in ORDER_BASE}
    for level in levels:
        for m in ORDER_BASE:
            sub = [
                r
                for r in method_rows(rows, m)
                if r.get("charge_class") == "charging_required" and r.get(key) == level
            ]
            out[m].append(100.0 * mean(float(r["feasible"]) for r in sub) if sub else np.nan)
    return out


def _grouped_bar_plot(stem, rows, key, levels, xlabel):
    data = _grouped_feasibility(rows, key, levels)
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    x = np.arange(len(levels))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(ORDER_BASE)) * width
    for off, m in zip(offsets, ORDER_BASE):
        vals = data[m]
        bars = ax.bar(x + off, vals, width=width, color=COLOR[m], label=LABEL[m], edgecolor="none")
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, levels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feasibility (%)")
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    save_png(fig, stem)


def fig09_by_layout(rows):
    _grouped_bar_plot("fig09_by_layout", rows, "layout", ["R", "C", "RC"], "Customer layout")


def fig10_by_length(rows):
    _grouped_bar_plot("fig10_by_length", rows, "length_bin", ["short", "medium", "long"], "Route length")


# ---------------------------------------------------------------------------
# Fig. 11–12 — difficulty heatmaps
# ---------------------------------------------------------------------------
def _cell_feasibility(rows, method, layout, length):
    cell = [
        r
        for r in method_rows(rows, method)
        if r.get("charge_class") == "charging_required" and r.get("layout") == layout and r.get("length_bin") == length
    ]
    return mean(float(r["feasible"]) for r in cell) if cell else np.nan


def fig11_difficulty_heatmap(rows):
    layouts, bins = ["R", "C", "RC"], ["short", "medium", "long"]
    feas = np.zeros((3, 3))
    for i, layout in enumerate(layouts):
        for j, length in enumerate(bins):
            fh = _cell_feasibility(rows, "HybridPPO", layout, length)
            feas[i, j] = 100.0 * fh if fh is not None else np.nan
    fig, ax = plt.subplots(figsize=(4.9, 3.9))
    im = ax.imshow(feas, vmin=70, vmax=100, cmap="Blues")
    ax.set_xticks(range(3), bins)
    ax.set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            v = feas[i, j]
            ax.text(j, i, "NA" if np.isnan(v) else f"{v:.0f}%", ha="center", va="center", color="white" if (not np.isnan(v) and v > 88) else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("FA-HPPO feasibility (%)")
    ax.set_xlabel("Route length")
    ax.set_ylabel("Layout")
    fig.tight_layout()
    save_png(fig, "fig11_difficulty_heatmap")


def fig12_gain_vs_lookahead(rows):
    layouts, bins = ["R", "C", "RC"], ["short", "medium", "long"]
    delta = np.zeros((3, 3))
    for i, layout in enumerate(layouts):
        for j, length in enumerate(bins):
            fh = _cell_feasibility(rows, "HybridPPO", layout, length)
            fl = _cell_feasibility(rows, "OneStepLookahead", layout, length)
            delta[i, j] = np.nan if fh is None or fl is None else 100.0 * (fh - fl)
    fig, ax = plt.subplots(figsize=(4.9, 3.9))
    dmax = max(1.0, float(np.nanmax(delta)))
    im = ax.imshow(delta, vmin=0, vmax=dmax, cmap="YlOrBr")
    ax.set_xticks(range(3), bins)
    ax.set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            v = delta[i, j]
            txt = "NA" if np.isnan(v) else ("0" if abs(v) < 0.05 else f"{v:+.0f}")
            ax.text(j, i, txt, ha="center", va="center", color="white" if (not np.isnan(v) and v > 0.55 * dmax) else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("Feasibility gain vs Lookahead (pp)")
    ax.set_xlabel("Route length")
    ax.set_ylabel("Layout")
    fig.tight_layout()
    save_png(fig, "fig12_gain_vs_lookahead")


# ---------------------------------------------------------------------------
# Fig. 13 — gold ablation as simple bars (no seed lines)
# ---------------------------------------------------------------------------
def load_ablation():
    rows = []
    for code, _ in ABL_ORDER:
        for seed in SEEDS:
            rows.append(load_json(V3 / "ablation" / code / f"seed_{seed}" / "validation.json"))
    return rows


def fig13_ablation_bars(ablation):
    labels = {
        "B0": "B0\n(no cap,\nno scale)",
        "B1": "B1\n(cap only)",
        "B3": "B3\n(scale only)",
        "B2": "B2\n(cap + scale)",
    }
    colors = {"B0": "#999999", "B1": "#E69F00", "B3": "#56B4E9", "B2": "#0072B2"}
    means = []
    for code, _ in ABL_ORDER:
        vals = [100.0 * r["parent_balanced_val_feasibility"] for r in ablation if r["variant"] == code]
        means.append(mean(vals))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    xs = np.arange(4)
    ax.bar(xs, means, color=[colors[c] for c, _ in ABL_ORDER], width=0.66, edgecolor="none")
    for i, v in enumerate(means):
        ax.text(i, v + 1.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [labels[c] for c, _ in ABL_ORDER])
    ax.set_ylabel("Parent-balanced VAL feasibility (%)")
    ax.set_ylim(0, 108)
    fig.tight_layout()
    save_png(fig, "fig13_ablation_bars")


# ---------------------------------------------------------------------------
# Fig. 14 — amount policies as clear grouped comparison
# ---------------------------------------------------------------------------
def fig14_amount_policies(rows):
    summaries = {m: summarize_learned(rows, m) for m in AMOUNT_ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.45))
    _vbar(
        axes[0],
        AMOUNT_ORDER,
        {m: summaries[m]["feasibility_mean"] for m in AMOUNT_ORDER},
        {m: summaries[m]["feasibility_ci95"] for m in AMOUNT_ORDER},
        ylabel="Route feasibility (%)",
        ylim=(0, 118),
        label_fmt="{:.1f}",
        as_pct=True,
    )
    axes[0].set_title("(a) Feasibility", loc="left")
    _vbar(
        axes[1],
        AMOUNT_ORDER,
        {m: summaries[m]["completion_all_mean"] for m in AMOUNT_ORDER},
        {m: summaries[m]["completion_all_ci95"] for m in AMOUNT_ORDER},
        ylabel="Failure-retaining completion",
        ylim=(0, 10.5),
        label_fmt="{:.2f}",
        as_pct=False,
    )
    axes[1].set_title("(b) Completion", loc="left")
    fig.tight_layout()
    save_png(fig, "fig14_amount_policies")


# ---------------------------------------------------------------------------
# Fig. 15 — charging effort among feasible routes
# ---------------------------------------------------------------------------
def fig15_charging_effort(rows):
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    vals = []
    for m in ORDER_BASE:
        sub = [r for r in method_rows(rows, m) if r["feasible"]]
        vals.append(mean(float(r["n_station_visits"]) for r in sub) if sub else np.nan)
    xs = np.arange(len(ORDER_BASE))
    ax.bar(xs, vals, color=[COLOR[m] for m in ORDER_BASE], width=0.62, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [LABEL[m] for m in ORDER_BASE])
    ax.set_ylabel("Mean station visits (feasible routes)")
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    save_png(fig, "fig15_charging_effort")


# ---------------------------------------------------------------------------
# Fig. 16 — runtime
# ---------------------------------------------------------------------------
def fig16_runtime(rows, summaries):
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    vals = [1000.0 * summaries[m]["runtime_s"] for m in ORDER_BASE]
    xs = np.arange(len(ORDER_BASE))
    ax.bar(xs, vals, color=[COLOR[m] for m in ORDER_BASE], width=0.62, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [LABEL[m] for m in ORDER_BASE])
    ax.set_ylabel("Mean evaluation runtime (ms / route)")
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    save_png(fig, "fig16_runtime")


# ---------------------------------------------------------------------------
# Fig. 17 — failure rate by regime (no seed counts)
# ---------------------------------------------------------------------------
def fig17_failure_by_regime(rows):
    layouts, bins = ["R", "C", "RC"], ["short", "medium", "long"]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    x = np.arange(len(bins))
    width = 0.22
    offsets = {"R": -width, "C": 0.0, "RC": width}
    colors = {"R": "#0072B2", "C": "#009E73", "RC": "#D55E00"}
    for layout in layouts:
        ys = []
        for length in bins:
            cell = [
                r
                for r in method_rows(rows, "HybridPPO")
                if r.get("charge_class") == "charging_required"
                and r.get("layout") == layout
                and r.get("length_bin") == length
            ]
            ys.append(100.0 * mean(float(not r["feasible"]) for r in cell) if cell else np.nan)
        bars = ax.bar(x + offsets[layout], ys, width=width, color=colors[layout], label=layout, edgecolor="none")
        for b, v in zip(bars, ys):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, bins)
    ax.set_xlabel("Route length")
    ax.set_ylabel("FA-HPPO failure rate (%)")
    ax.set_ylim(0, 28)
    ax.legend(frameon=False, title="Layout")
    fig.tight_layout()
    save_png(fig, "fig17_failure_by_regime")


# ---------------------------------------------------------------------------
# Fig. 18 — envelope variants (means only)
# ---------------------------------------------------------------------------
def fig18_envelope_variants():
    summary = load_json(ENVELOPE_SUMMARY)
    order = [
        ("A_arrival_to_max", "Arrival→Max"),
        ("B_energy_to_max", "EnergyLower→Max"),
        ("C_energy_to_time", "EnergyLower→TimeUpper"),
    ]
    means = [100.0 * float(summary["variants"][key]["val_feas_mean"]) for key, _ in order]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    xs = np.arange(3)
    ax.bar(xs, means, color=["#999999", "#56B4E9", "#0072B2"], width=0.62, edgecolor="none")
    for i, v in enumerate(means):
        ax.text(i, v + 0.15, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [lab for _, lab in order])
    ax.set_ylabel("Parent-balanced VAL feasibility (%)")
    lo = min(means) - 1.5
    ax.set_ylim(max(90, lo), 101.5)
    fig.tight_layout()
    save_png(fig, "fig18_envelope_variants")


# ---------------------------------------------------------------------------
# Fig. 19 — one charge decision (discrete + continuous)
# ---------------------------------------------------------------------------
def fig19_charge_decision():
    ep = load_json(CASE)
    charge = next(d for d in ep["decisions"] if d.get("action") == "CHARGE")
    names = charge["policy"]["action_names"]
    probs = [float(p) for p in charge["policy"]["probs"]]
    mask = [bool(m) for m in charge["policy"]["mask"]]
    lo, hi, tgt = float(charge["soc_lower"]), float(charge["soc_upper"]), float(charge["soc_target"])
    u = float(charge["u"])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.35))
    # Discrete action probabilities
    ax = axes[0]
    xs = np.arange(len(names))
    colors = ["#BDBDBD" if not ok else ("#0072B2" if n != "CONTINUE" else "#56B4E9") for n, ok in zip(names, mask)]
    ax.bar(xs, probs, color=colors, width=0.66, edgecolor="none")
    for i, (p, ok) in enumerate(zip(probs, mask)):
        ax.text(i, p + 0.03, "masked" if not ok else f"{p:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs, names)
    ax.set_ylabel("Action probability")
    ax.set_ylim(0, 1.18)
    ax.set_title("(a) Discrete station choice", loc="left")

    # Continuous amount inside envelope
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.add_patch(Rectangle((lo, 0.35), hi - lo, 0.25, facecolor="#56B4E9", alpha=0.45, edgecolor="#0072B2", lw=1.5))
    ax.axvline(lo, color="#0072B2", lw=1.5, ymin=0.3, ymax=0.7)
    ax.axvline(hi, color="#D55E00", lw=1.5, ymin=0.3, ymax=0.7)
    ax.scatter([tgt], [0.475], s=85, color="#111111", zorder=5, edgecolors="white", linewidths=0.5)
    ax.text(lo, 0.78, "lower", ha="center", color="#0072B2", fontsize=8)
    ax.text(hi, 0.78, "upper", ha="center", color="#D55E00", fontsize=8)
    ax.text(tgt, 0.15, f"target (u={u:.2f})", ha="center", fontsize=8)
    ax.set_xlabel("SOC")
    ax.set_title(f"(b) Charge amount at {charge['station_id']}", loc="left")
    fig.tight_layout()
    save_png(fig, "fig19_charge_decision")


# ---------------------------------------------------------------------------
# Fig. 20 — native FRVCP reference
# ---------------------------------------------------------------------------
def fig20_frvcp():
    summary_csv = ROOT / "results/final/statistics/frvcpy_native/method_summary.csv"
    gap_csv = ROOT / "results/final/tables/frvcpy_native/table_F_frvcpy.csv"
    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    gaps = {}
    for r in csv.DictReader(gap_csv.open(encoding="utf-8")):
        raw = (r.get("mean_gap_percent_vs_frvcpy_solver") or "").strip()
        if raw:
            gaps[r["method"]] = float(raw)
    order = ["frvcpy_Solver", "FRVCPGreedyMin", "FRVCPGreedyFull"]
    labels = {"frvcpy_Solver": "Exact solver", "FRVCPGreedyMin": "Greedy Min", "FRVCPGreedyFull": "Greedy Full"}
    colors = {"frvcpy_Solver": "#0072B2", "FRVCPGreedyMin": "#E69F00", "FRVCPGreedyFull": "#009E73"}
    feas = {r["method"]: 100.0 * float(r["feasibility_mean"]) for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    x = np.arange(len(order))
    axes[0].bar(x, [feas[m] for m in order], color=[colors[m] for m in order], width=0.62, edgecolor="none")
    for i, m in enumerate(order):
        axes[0].text(i, feas[m] + 1.5, f"{feas[m]:.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(x, [labels[m] for m in order])
    axes[0].set_ylabel("Feasibility (%)")
    axes[0].set_ylim(0, 118)
    axes[0].set_title("(a) Feasibility", loc="left")
    gm = ["FRVCPGreedyMin", "FRVCPGreedyFull"]
    x2 = np.arange(len(gm))
    axes[1].bar(x2, [gaps[m] for m in gm], color=[colors[m] for m in gm], width=0.55, edgecolor="none")
    for i, m in enumerate(gm):
        axes[1].text(i, gaps[m] + 0.4, f"{gaps[m]:.1f}", ha="center", fontsize=8)
    axes[1].set_xticks(x2, [labels[m] for m in gm])
    axes[1].set_ylabel("Mean gap vs exact solver (%)")
    axes[1].set_title("(b) Optimality gap", loc="left")
    fig.tight_layout()
    save_png(fig, "fig20_frvcp_reference")


# ---------------------------------------------------------------------------
# Failure table helper + tables / captions / verify
# ---------------------------------------------------------------------------
def _failure_problem_routes(rows):
    hybrid = method_rows(rows, "HybridPPO")
    by_route = defaultdict(list)
    for r in hybrid:
        by_route[r["route_id"]].append(r)
    problem, n_all_ok = [], 0
    for rid, rr in by_route.items():
        n_fail = sum(1 for r in rr if not r["feasible"])
        if n_fail == 0:
            n_all_ok += 1
            continue
        sample = next(r for r in rr if not r["feasible"])
        problem.append({"route_id": rid, "n_fail": n_fail, "layout": sample.get("layout"), "length": sample.get("length_bin")})
    problem.sort(key=lambda d: (-d["n_fail"], d["route_id"]))
    return problem, n_all_ok, by_route


def write_table(stem, title, header, rows, note):
    TAB.mkdir(parents=True, exist_ok=True)
    md = [f"# {title}", "", note, "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        md.append("| " + " | ".join(str(c) for c in row) + " |")
    (TAB / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (TAB / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(header)
        w.writerows(rows)
    cols = "l" + "c" * (len(header) - 1)
    tex = ["% Auto-generated from frozen results.", r"\begin{table}[t]", r"\centering", rf"\caption{{{title}}}", rf"\label{{tab:{stem}}}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        tex.append(" & ".join(str(c) for c in row) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TAB / f"{stem}.tex").write_text("\n".join(tex), encoding="utf-8")


def build_tables(all_s, ch_s, ablation, rows):
    write_table(
        "table01_benchmark_protocol",
        "Table 1 — Benchmark / protocol",
        ["Corpus", "Role", "Physics", "Routes", "Charge-req", "Seeds", "Fresh confirmatory?", "Ckpt selection?", "Final claims?"],
        [
            ["SynthCharge TRAIN", "development", "synthcharge_linear", "180", "~80%", "—", "no", "normalizer/return scale", "no"],
            ["SynthCharge VAL", "development", "synthcharge_linear", "90", "~80%", "—", "no", "yes (parent-bal.)", "no"],
            ["V3 SynthCharge TEST", "fresh independently generated confirmatory", "synthcharge_linear", "180", "144/180", "5 (eval)", "yes", "no", "yes"],
            ["Gold TRAIN/VAL", "methodology ablation", "official_evrptwgr", "119/47", "—", "42–46", "no", "ablation only", "methodology only"],
            ["Native FRVCP archive", "separate reference", "native FRVCP", "133", "—", "—", "no", "no", "reference only"],
        ],
        "V3 is a fresh independently generated SynthCharge TEST, not external-domain generalization.",
    )
    write_table(
        "table02_main_results",
        "Table 2 — Main V3 TEST results",
        ["Method", "All feas.", "95% CI", "All completion", "Charge-req feas.", "Charge-req completion", "Runtime (s)"],
        [
            [
                LABEL[m],
                fmt(all_s[m]["feasibility_mean"]),
                "NA" if not all_s[m]["feasibility_ci95"] else f"[{fmt(all_s[m]['feasibility_ci95'][0])}, {fmt(all_s[m]['feasibility_ci95'][1])}]",
                fmt(all_s[m]["completion_all_mean"], 3),
                fmt(ch_s[m]["feasibility_mean"]),
                fmt(ch_s[m]["completion_all_mean"], 3),
                fmt(all_s[m]["runtime_s"], 3),
            ]
            for m in ORDER_BASE
        ],
        "FA-HPPO = mean over 5 training seeds; baselines deterministic. Infeasible completion = horizon H. CI = hierarchical bootstrap. See Figs. 5–7.",
    )
    rows_abl = []
    for code, lab in ABL_ORDER:
        subset = [r for r in ablation if r["variant"] == code]
        rows_abl.append(
            [
                lab,
                "on" if subset[0]["time_aware"] else "off",
                "on" if subset[0]["return_scale_enabled"] else "off",
                fmt(mean(r["parent_balanced_val_feasibility"] for r in subset)),
                fmt(float(np.std([r["parent_balanced_val_feasibility"] for r in subset], ddof=1))),
                fmt(mean(r["charging_required_feasibility"] for r in subset)),
                fmt(mean(r["parent_balanced_completion_all"] for r in subset), 1),
                fmt(mean(r["optimization"]["value_loss_mean"] for r in subset), 4),
                fmt(mean(r["optimization"]["grad_norm_preclip_mean"] for r in subset), 3),
            ]
        )
    write_table(
        "table03_development_ablation",
        "Table 3 — Development methodology ablation",
        ["Variant", "Time-aware cap", "Return scale", "VAL feas.", "SD", "Charge-req feas.", "Completion", "Value loss", "Grad norm"],
        rows_abl,
        "DEVELOPMENT / GOLD VALIDATION ONLY. B0/B1/B3/B2 are plotted in Fig. 13.",
    )
    amt = {m: summarize_learned(rows, m) for m in ("HybridPPO", "FA-HPPO-Max", "FA-HPPO-Min")}
    tie = load_json(STATS / "amount_sensitivity.json")["route_averaged_feasibility_vs_max"]
    write_table(
        "table04_amount_sensitivity",
        "Table 4 — TEST amount-policy sensitivity",
        ["Method", "u policy", "Feasibility", "95% CI", "Completion", "Route vs Max", "Interpretation"],
        [
            ["FA-HPPO", "learned Beta", fmt(amt["HybridPPO"]["feasibility_mean"]), f"[{fmt(amt['HybridPPO']['feasibility_ci95'][0])}, {fmt(amt['HybridPPO']['feasibility_ci95'][1])}]", fmt(amt["HybridPPO"]["completion_all_mean"], 3), f"better={tie['free_better']}, tied={tie['tied']}, worse={tie['max_better']}", "primary"],
            ["FA-HPPO-Max", "forced u=1", fmt(amt["FA-HPPO-Max"]["feasibility_mean"]), f"[{fmt(amt['FA-HPPO-Max']['feasibility_ci95'][0])}, {fmt(amt['FA-HPPO-Max']['feasibility_ci95'][1])}]", fmt(amt["FA-HPPO-Max"]["completion_all_mean"], 3), "—", "≈ free"],
            ["FA-HPPO-Min", "forced u=0", fmt(amt["FA-HPPO-Min"]["feasibility_mean"]), f"[{fmt(amt['FA-HPPO-Min']['feasibility_ci95'][0])}, {fmt(amt['FA-HPPO-Min']['feasibility_ci95'][1])}]", fmt(amt["FA-HPPO-Min"]["completion_all_mean"], 3), "—", "degenerate stress"],
        ],
        "Same frozen checkpoints on V3 TEST. Visual summary in Fig. 14.",
    )
    write_table(
        "tableA01_per_seed",
        "Table A1 — FA-HPPO per seed",
        ["Seed", "Feasibility", "Completion", "Charge-req feasibility"],
        [[str(s), fmt(all_s["HybridPPO"]["per_seed_feasibility"][s]), fmt(all_s["HybridPPO"]["per_seed_completion"][s], 3), fmt(ch_s["HybridPPO"]["per_seed_feasibility"][s])] for s in SEEDS],
        "Individual training seeds on V3 TEST (table only; not plotted).",
    )
    paired = load_json(STATS / "paired_primary.json")
    write_table(
        "tableA02_primary_statistics",
        "Table A2 — Primary paired statistics",
        ["Comparison", "Metric", "Effect", "95% CI", "p_raw", "p_Holm", "Procedure", "Units"],
        [
            [t["comparison"], t["family"], fmt(t["effect"], 4), f"[{fmt(t['ci95'][0], 4)}, {fmt(t['ci95'][1], 4)}]", fmt_p(t["p_raw"]), fmt_p(t["p_holm"]), t.get("procedure", "predeclared"), str(t["n_independent_units"])]
            for t in paired
        ],
        "Positive feasibility and negative completion favor FA-HPPO. Holm-corrected permutation tests; MC floor <5e-5. Paired effects are table-only (not duplicated as a figure).",
    )
    problem, _n_ok, by_route = _failure_problem_routes(rows)
    alias = {f"R{i:02d}": d["route_id"] for i, d in enumerate(problem, start=1)}
    (OUT / "failure_analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "failure_analysis" / "ROUTE_ALIASES.json").write_text(
        json.dumps({"n_all_ok": _n_ok, "n_routes": len(by_route), "aliases": alias}, indent=2) + "\n",
        encoding="utf-8",
    )
    table_rows = []
    for i, d in enumerate(problem, start=1):
        rr = by_route[d["route_id"]]
        seeds_fail = sorted(int(r["seed"]) for r in rr if not r["feasible"])
        seeds_ok = sorted(int(r["seed"]) for r in rr if r["feasible"])
        sample = next(r for r in rr if not r["feasible"])
        visits = [float(r.get("n_station_visits") or 0) for r in rr if not r["feasible"]]
        terms = [r.get("terminal_soc") for r in rr if not r["feasible"] and r.get("terminal_soc") is not None]
        table_rows.append(
            [
                f"R{i:02d}",
                d["route_id"],
                sample.get("layout"),
                sample.get("length_bin"),
                sample.get("charge_class"),
                str(d["n_fail"]),
                ",".join(map(str, seeds_fail)),
                ",".join(map(str, seeds_ok)) if seeds_ok else "—",
                sample.get("reason") or "NA",
                fmt(mean(visits), 2),
                fmt(mean(terms), 3) if terms else "NA",
            ]
        )
    write_table(
        "tableA03_failure_routes",
        "Table A3 — FA-HPPO failing routes (frozen V3 raw)",
        ["Alias", "Route", "Layout", "Length", "Charge class", "n_seeds_fail", "Fail seeds", "OK seeds", "Reason", "Mean visits (fail)", "Mean terminal SOC (fail)"],
        table_rows,
        "Per-route failure detail (table). Aggregate regime view in Fig. 17.",
    )


def write_captions() -> None:
    text = """# Figure captions (`results_paper/figures/`)

PNG only (600 dpi). Flat folder (no appendix/). Captions carry interpretation.

**Figure 1 — Problem setting.** Illustrative SynthCharge VAL episode (not TEST). Depot (square), numbered customers in fixed order, unused/used charging stations (triangles with IDs), dashed fixed customer sequence, and solid executed FA-HPPO path with charging detours.

**Figure 2 — Method overview.** FA-HPPO hybrid control: discrete CONTINUE/station choice, continuous charge fraction $u$ when charging, mapped through a feasibility-aware SOC envelope, then environment step.

**Figure 3 — SOC envelope.** Arrival SOC, energy-continuation lower bound, time-aware upper bound, and target SOC $= lower + u(upper-lower)$.

**Figure 4 — Illustrative SOC trajectory.** Same VAL episode as Fig. 1. SOC along the executed visit sequence; shaded intervals mark feasible departure bands at charging stops.

**Figure 5 — Main TEST feasibility.** Locked V3 SynthCharge TEST (180 routes). Vertical bars; FA-HPPO whiskers are hierarchical 95% bootstrap intervals. Higher is better.

**Figure 6 — Main TEST completion.** Failure-retaining completion on the same TEST (infeasible episodes keep horizon $H$). Lower is better.

**Figure 7 — Charging-required subset.** Feasibility restricted to routes that require charging (harder stratum).

**Figure 8 — Performance profiles.** Share of routes that are feasible and finished by a completion-time threshold. Curves asymptote at each method’s feasibility rate.

**Figure 9 — Feasibility by layout.** Charging-required TEST routes stratified by R / C / RC.

**Figure 10 — Feasibility by length.** Charging-required TEST routes stratified by short / medium / long.

**Figure 11 — Difficulty heatmap.** FA-HPPO feasibility (%) on charging-required cells (layout × length).

**Figure 12 — Gain vs Lookahead.** Feasibility improvement (pp) of FA-HPPO over OneStepLookahead by layout × length.

**Figure 13 — Development ablation.** Gold EVRPTW-GR VAL means for B0–B2 (time-aware cap × return scaling). Development evidence only.

**Figure 14 — Amount policies.** Same frozen checkpoints with learned $u$, forced $u=1$ (Max), and forced $u=0$ (Min).

**Figure 15 — Charging effort.** Mean station visits among feasible TEST routes.

**Figure 16 — Runtime.** Mean evaluation runtime per route (ms).

**Figure 17 — Failure regimes.** FA-HPPO failure rate (%) on charging-required cells by layout and length (aggregate; not per-seed).

**Figure 18 — Envelope variants.** Post-hoc SynthCharge VAL means for Arrival→Max / EnergyLower→Max / EnergyLower→TimeUpper. Development nuance only.

**Figure 19 — Charge decision snapshot.** One VAL charging decision: (a) discrete action probabilities (masked illegal stations); (b) continuous target inside the local SOC envelope.

**Figure 20 — Native FRVCP reference.** Separate FRVCP archive (n=133). Not an exact SynthCharge/EVRPTW-GR comparator.
"""
    (OUT / "CAPTIONS.md").write_text(text, encoding="utf-8")


def write_readme() -> None:
    lines = [
        "# Publication outputs (`results_paper/`)",
        "",
        "Canonical manuscript outputs. **PNG only** (600 dpi). All figures are flat in `figures/`.",
        "",
        "## Figures (20)",
        "",
        "| # | File | Question |",
        "|---|------|----------|",
        "| 1 | `fig01_usecase_route.png` | What is the fixed-route charging use case? |",
        "| 2 | `fig02_method_schematic.png` | How does FA-HPPO decide? |",
        "| 3 | `fig03_soc_envelope.png` | What is the SOC envelope? |",
        "| 4 | `fig04_illustrative_soc.png` | How does SOC evolve on the use-case? |",
        "| 5 | `fig05_main_feasibility.png` | Main TEST feasibility? |",
        "| 6 | `fig06_main_completion.png` | Main TEST completion? |",
        "| 7 | `fig07_charging_required.png` | Hard charging-required subset? |",
        "| 8 | `fig08_performance_profiles.png` | Feasibility + speed together? |",
        "| 9 | `fig09_by_layout.png` | Effect of layout? |",
        "| 10 | `fig10_by_length.png` | Effect of route length? |",
        "| 11 | `fig11_difficulty_heatmap.png` | Layout × length feasibility? |",
        "| 12 | `fig12_gain_vs_lookahead.png` | Where are gains largest? |",
        "| 13 | `fig13_ablation_bars.png` | Which methodology factors matter? |",
        "| 14 | `fig14_amount_policies.png` | Learned u vs Max vs Min? |",
        "| 15 | `fig15_charging_effort.png` | How much charging is used? |",
        "| 16 | `fig16_runtime.png` | Evaluation cost? |",
        "| 17 | `fig17_failure_by_regime.png` | Where do failures concentrate? |",
        "| 18 | `fig18_envelope_variants.png` | Envelope design nuance (dev)? |",
        "| 19 | `fig19_charge_decision.png` | What does one charge decision look like? |",
        "| 20 | `fig20_frvcp_reference.png` | Native FRVCP reference? |",
        "",
        "Captions: `CAPTIONS.md`. Map: `../paper/RESULTS_MAP.md`.",
        "",
        "```bash",
        "python scripts/paper/build_results_paper.py",
        "python scripts/paper/build_results_paper.py --verify",
        "```",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_manifest(artifacts):
    for item in artifacts:
        path = ROOT / item["path"]
        item["sha256"] = _sha(path) if path.is_file() else None
        item["generation_script"] = "scripts/paper/build_results_paper.py"
    (OUT / "MANIFEST.json").write_text(json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify() -> None:
    consumed = load_json(V3 / "EVALUATION_CONSUMED.json")
    if lf_sha256(RAW) != consumed["raw_sha256_lf"]:
        raise SystemExit("raw TEST hash mismatch")
    rows = load_rows()
    if len(rows) != 3240:
        raise SystemExit(f"expected 3240 rows, got {len(rows)}")
    lock = load_json(V3 / "TEST_LOCK.json")
    bad = [rel for rel, digest in lock["files"].items() if lf_sha256(ROOT / rel) != digest]
    if bad:
        raise SystemExit(f"TEST_LOCK mismatch: {bad[:5]}")
    import subprocess

    tracked = subprocess.check_output(["git", "ls-files", "data/routes_v2/synthcharge_v3_test/instances"], text=True).splitlines()
    if len(tracked) != 180:
        raise SystemExit(f"expected 180 tracked instances, got {len(tracked)}")
    if not CASE.is_file():
        raise SystemExit("missing case study JSON")
    expected = {f"{s}.png" for s in FIGS}
    pngs = {p.name for p in FIG.glob("*.png")}
    nested = [p for p in FIG.rglob("*.png") if p.parent != FIG]
    pdfs = list(FIG.rglob("*.pdf"))
    if nested:
        raise SystemExit(
            "figures must be flat in figures/; found nested: "
            + str([str(p.relative_to(FIG)) for p in nested[:5]])
        )
    if expected - pngs:
        raise SystemExit(f"missing PNGs: {sorted(expected - pngs)}")
    if pngs - expected:
        raise SystemExit(f"unexpected PNGs: {sorted(pngs - expected)}")
    if pdfs:
        raise SystemExit(f"PDF figures not allowed: {[p.name for p in pdfs[:5]]}")
    print(json.dumps({"verify": "ok", "n_rows": 3240, "n_png": len(pngs), "n_pdf": 0}, indent=2))


def main() -> None:
    if "--verify" in sys.argv:
        verify()
        return
    style()
    rows = load_rows()
    fig01_usecase_route()
    fig02_method_schematic()
    fig03_soc_envelope()
    fig04_illustrative_soc()
    all_s = fig05_main_feasibility(rows)
    fig06_main_completion(rows, all_s)
    ch_s = fig07_charging_required(rows)
    fig08_performance_profiles(rows)
    fig09_by_layout(rows)
    fig10_by_length(rows)
    fig11_difficulty_heatmap(rows)
    fig12_gain_vs_lookahead(rows)
    ablation = load_ablation()
    fig13_ablation_bars(ablation)
    fig14_amount_policies(rows)
    fig15_charging_effort(rows)
    fig16_runtime(rows, all_s)
    fig17_failure_by_regime(rows)
    fig18_envelope_variants()
    fig19_charge_decision()
    fig20_frvcp()
    cleanup_figures()
    build_tables(all_s, ch_s, ablation, rows)
    write_captions()
    write_readme()
    artifacts = [
        {"path": "results_paper/README.md", "role": "index"},
        {"path": "results_paper/CAPTIONS.md", "role": "captions"},
        {"path": "results_paper/case_study/illustrative_val_episode.json", "role": "methodology-illustration", "note": "VAL not TEST"},
    ]
    for stem in FIGS:
        artifacts.append({"path": f"results_paper/figures/{stem}.png", "role": "figure"})
    for stem in (
        "table01_benchmark_protocol",
        "table02_main_results",
        "table03_development_ablation",
        "table04_amount_sensitivity",
        "tableA01_per_seed",
        "tableA02_primary_statistics",
        "tableA03_failure_routes",
    ):
        for ext in ("csv", "md", "tex"):
            artifacts.append({"path": f"results_paper/tables/{stem}.{ext}", "role": "table"})
    build_manifest(artifacts)
    print("results_paper written")


if __name__ == "__main__":
    main()
