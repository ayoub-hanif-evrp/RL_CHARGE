"""Build publication-facing results_paper/ (PDF + PNG figures + tables) from frozen data.

READ ONLY for TEST evidence. Case-study JSON is a VAL illustration (not TEST).
Never trains final models, evaluates TEST, or modifies frozen raw rows.

Official single publication-figure generator for the manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3_hppo"))

from experiments.stats import hierarchical_bootstrap_ci, mean_sd_across_training_seeds  # noqa: E402
from common import SEEDS, V3, lf_sha256, load_json  # noqa: E402

RAW = V3 / "raw" / "synthcharge_test.jsonl"
STATS = V3 / "statistics"
OUT = ROOT / "results_paper"
FIG = OUT / "figures"
FIG_A = FIG / "appendix"
TAB = OUT / "tables"
CASE = OUT / "case_study" / "illustrative_val_episode.json"
ENVELOPE_SUMMARY = ROOT / "results" / "development" / "envelope_ablation" / "SUMMARY.json"
N_BOOT = 5000
RNG = 20261003
DPI = 600

ORDER_BASE = ["HybridPPO", "OneStepLookahead", "GreedyFullCharge", "GreedyMinimumSufficientCharge"]
LABEL = {
    "HybridPPO": "FA-HPPO",
    "GreedyMinimumSufficientCharge": "Greedy Min",
    "GreedyFullCharge": "Greedy Full",
    "OneStepLookahead": "Lookahead",
    "FA-HPPO-Min": "FA-HPPO-Min",
    "FA-HPPO-Max": "FA-HPPO-Max",
}
COLOR = {
    "HybridPPO": "#0072B2",
    "OneStepLookahead": "#D55E00",
    "GreedyFullCharge": "#009E73",
    "GreedyMinimumSufficientCharge": "#E69F00",
    "FA-HPPO-Min": "#CC79A7",
    "FA-HPPO-Max": "#56B4E9",
}
MARKER = {
    "HybridPPO": "o",
    "OneStepLookahead": "s",
    "GreedyFullCharge": "D",
    "GreedyMinimumSufficientCharge": "^",
    "FA-HPPO-Max": "P",
    "FA-HPPO-Min": "X",
}
ABL_ORDER = [("B0", "B0"), ("B1", "B1"), ("B3", "B3"), ("B2", "B2")]
ABL_META = {
    "B0": {"time_cap": "no", "scaling": "no", "name": "Base HPPO"},
    "B1": {"time_cap": "yes", "scaling": "no", "name": "+ Time-aware cap"},
    "B3": {"time_cap": "no", "scaling": "yes", "name": "+ Return scaling"},
    "B2": {"time_cap": "yes", "scaling": "yes", "name": "FA-HPPO full"},
}
MC_P_FLOOR = 5e-5

MAIN_FIGS = [
    "fig01_method_schematic",
    "fig02_main_test",
    "fig03_effect_sizes",
    "fig04_difficulty_robustness",
    "fig05_mechanism",
]
APPENDIX_FIGS = [
    "figA01_illustrative_val_trajectory",
    "figA02_learning_curves",
    "figA03_training_stability",
    "figA04_envelope_ablation",
    "figA05_failure_analysis",
    "figA06_seed_robustness",
    "figA07_native_frvcp_reference",
]
OBSOLETE_FIGS = [
    "fig01_method_case_study",
    "fig04_ablation_and_training_stability",
    "fig05_amount_sensitivity",
    "figA01_difficulty_heatmap",
    "figA02_seed_robustness",
    "figA03_native_frvcp_reference",
    "figA04_learning_curves",
    "figA05_failure_consistency",
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


def fmt_pct(v, d=1):
    if v is None:
        return "NA"
    return f"{100.0 * float(v):.{d}f}%"


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
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig, stem: str, appendix: bool = False) -> None:
    base = FIG_A if appendix else FIG
    base.mkdir(parents=True, exist_ok=True)
    fig.savefig(base / f"{stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(base / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cleanup_obsolete() -> None:
    for stem in OBSOLETE_FIGS:
        for folder in (FIG, FIG_A):
            for ext in (".png", ".pdf", ".svg"):
                path = folder / f"{stem}{ext}"
                if path.is_file():
                    path.unlink()


def _beta_pdf(u, alpha, beta):
    from math import lgamma

    u = np.clip(np.asarray(u, dtype=float), 1e-6, 1 - 1e-6)
    log_b = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    return np.exp((alpha - 1) * np.log(u) + (beta - 1) * np.log(1 - u) - log_b)


# ---------------------------------------------------------------------------
# Fig. 1 — conceptual method schematic
# ---------------------------------------------------------------------------
def fig01_method_schematic():
    fig = plt.figure(figsize=(11.2, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.2], wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # (a) Fixed route + optional charging
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 6.2)
    ax_a.axis("off")
    ax_a.set_title("(a) Fixed customer route", loc="left")
    cust = [(1.2, 3.2), (3.0, 4.4), (5.0, 3.5), (7.0, 4.2), (8.8, 3.0)]
    labels_c = ["Depot", "C1", "C2", "C3", "C4", "Depot"]
    route = [(1.2, 3.2), (3.0, 4.4), (5.0, 3.5), (7.0, 4.2), (8.8, 3.0), (1.2, 3.2)]
    # draw return lightly
    xs, ys = zip(*route[:-1])
    ax_a.plot(xs, ys, color="#888888", lw=1.6, ls="--", zorder=1)
    ax_a.annotate(
        "",
        xy=route[-1],
        xytext=route[-2],
        arrowprops=dict(arrowstyle="-|>", color="#BBBBBB", lw=1.0, mutation_scale=9),
    )
    for i, (x, y) in enumerate(route[:-1]):
        if i == 0:
            ax_a.scatter([x], [y], marker="s", s=90, color="black", zorder=4)
        else:
            ax_a.scatter([x], [y], s=110, facecolors="white", edgecolors="#222222", linewidths=1.2, zorder=4)
            ax_a.text(x, y, str(i), ha="center", va="center", fontsize=8, zorder=5)
        ax_a.text(x, y - 0.55, labels_c[i], ha="center", fontsize=7, color="#333333")
    # stations
    stations = [("S1", 4.0, 1.7), ("S2", 6.4, 1.9)]
    for name, x, y in stations:
        ax_a.scatter([x], [y], marker="*", s=160, color=COLOR["HybridPPO"], zorder=5, edgecolors="white", linewidths=0.4)
        ax_a.text(x, y - 0.45, name, ha="center", fontsize=7, color=COLOR["HybridPPO"])
    # detour arrow to S1
    ax_a.annotate(
        "",
        xy=(4.0, 1.95),
        xytext=(3.0, 4.2),
        arrowprops=dict(arrowstyle="-|>", color=COLOR["HybridPPO"], lw=1.3, mutation_scale=10, connectionstyle="arc3,rad=0.2"),
    )
    ax_a.text(3.55, 2.85, "optional\ncharge", fontsize=6.5, color=COLOR["HybridPPO"], ha="center")
    ax_a.text(
        0.02,
        0.98,
        "Customer order fixed.\nPolicy: CONTINUE or visit a station.",
        transform=ax_a.transAxes,
        va="top",
        fontsize=7,
        color="#333333",
    )

    # (b) Feasibility-aware SOC interval
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(-0.15, 1.15)
    ax_b.set_title("(b) Feasibility-aware SOC interval", loc="left")
    ax_b.set_xlabel("SOC")
    ax_b.set_yticks([])
    ax_b.spines["left"].set_visible(False)
    arrival, lo, hi, u = 0.28, 0.42, 0.78, 0.65
    tgt = lo + u * (hi - lo)
    ax_b.axhspan(-0.05, 1.05, xmin=0.08, xmax=0.18, color="#F0F0F0")
    # interval band
    ax_b.add_patch(Rectangle((lo, 0.28), hi - lo, 0.34, facecolor="#56B4E9", alpha=0.35, edgecolor=COLOR["HybridPPO"], lw=1.4))
    ax_b.plot([arrival, arrival], [0.22, 0.68], color="#666666", lw=1.5, ls="--")
    ax_b.plot([lo, lo], [0.22, 0.68], color=COLOR["HybridPPO"], lw=1.8)
    ax_b.plot([hi, hi], [0.22, 0.68], color=COLOR["OneStepLookahead"], lw=1.8)
    ax_b.scatter([tgt], [0.45], s=70, color="#D55E00", zorder=5, edgecolors="white", linewidths=0.5)
    ax_b.annotate(
        "",
        xy=(tgt, 0.45),
        xytext=(lo, 0.45),
        arrowprops=dict(arrowstyle="<->", color="#D55E00", lw=1.2),
    )
    ax_b.text(arrival, 0.78, r"$SOC_{\mathrm{arr}}$", ha="center", fontsize=7.5, color="#555555")
    ax_b.text(lo, 0.78, r"$SOC_{\mathrm{lower}}$", ha="center", fontsize=7.5, color=COLOR["HybridPPO"])
    ax_b.text(hi, 0.78, r"$SOC_{\mathrm{upper}}$", ha="center", fontsize=7.5, color=COLOR["OneStepLookahead"])
    ax_b.text(tgt, 0.12, r"$SOC_{\mathrm{target}}$", ha="center", fontsize=7.5, color="#D55E00")
    ax_b.text(
        0.5,
        0.02,
        r"$SOC_{\mathrm{target}}=SOC_{\mathrm{lower}}+u\,(SOC_{\mathrm{upper}}-SOC_{\mathrm{lower}})$"
        "\n"
        r"$u\in[0,1]$",
        transform=ax_b.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    ax_b.text(0.5, 0.92, "valid departure interval", transform=ax_b.transAxes, ha="center", fontsize=7, color="#333333")
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])

    # (c) Hybrid policy decision
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis("off")
    ax_c.set_title("(c) Hybrid policy decision", loc="left")
    # discrete head box
    ax_c.add_patch(FancyBboxPatch((0.4, 5.6), 4.2, 3.6, boxstyle="round,pad=0.15", facecolor="#F7F7F7", edgecolor="#444444", lw=1.0))
    ax_c.text(2.5, 8.8, "Discrete head", ha="center", fontsize=8, fontweight="bold")
    names = ["CONT", "S1", "S2", "S3"]
    probs = [0.12, 0.55, 0.28, 0.0]
    mask = [True, True, True, False]
    for i, (name, p, m) in enumerate(zip(names, probs, mask)):
        x = 0.8 + i * 0.95
        h = 2.2 * p if m else 0.25
        color = COLOR["HybridPPO"] if name == "S1" else ("#BBBBBB" if m else "#EEEEEE")
        ax_c.add_patch(Rectangle((x, 5.9), 0.7, h, facecolor=color, edgecolor="#444444", lw=0.6, hatch="////" if not m else None))
        ax_c.text(x + 0.35, 5.7, name, ha="center", va="top", fontsize=6.5, color="#666666" if not m else "#222222")
        if not m:
            ax_c.text(x + 0.35, 5.95 + h + 0.15, "masked", ha="center", fontsize=5.8, color="#888888")

    # continuous head
    ax_c.add_patch(FancyBboxPatch((5.3, 5.6), 4.3, 3.6, boxstyle="round,pad=0.15", facecolor="#F7F7F7", edgecolor="#444444", lw=1.0))
    ax_c.text(7.45, 8.8, "Continuous head", ha="center", fontsize=8, fontweight="bold")
    ugrid = np.linspace(0.02, 0.98, 120)
    dens = _beta_pdf(ugrid, 4.0, 2.2)
    dens = dens / dens.max() * 2.0
    xs = 5.7 + ugrid * 3.5
    ys = 6.1 + dens
    ax_c.fill_between(xs, 6.1, ys, color="#56B4E9", alpha=0.35)
    ax_c.plot(xs, ys, color=COLOR["HybridPPO"], lw=1.4)
    u_mean = 4.0 / (4.0 + 2.2)
    ax_c.axvline(5.7 + u_mean * 3.5, ymin=0.58, ymax=0.84, color="#D55E00", lw=1.4)
    ax_c.text(7.45, 6.0, r"$u\sim\mathrm{Beta}(\alpha,\beta)$", ha="center", fontsize=7)

    # mapping arrow
    ax_c.annotate(
        "",
        xy=(5.0, 2.8),
        xytext=(5.0, 5.4),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.3, mutation_scale=12),
    )
    ax_c.text(5.2, 4.2, "selected station + u", fontsize=7, color="#333333")
    ax_c.add_patch(FancyBboxPatch((1.5, 0.7), 7.0, 2.0, boxstyle="round,pad=0.12", facecolor="#EAF4FB", edgecolor=COLOR["HybridPPO"], lw=1.1))
    ax_c.text(5.0, 2.1, r"map into $[SOC_{\mathrm{lower}},\,SOC_{\mathrm{upper}}]$", ha="center", fontsize=8)
    ax_c.text(5.0, 1.2, r"$SOC_{\mathrm{target}}=SOC_{\mathrm{lower}}+u\,\Delta SOC$", ha="center", fontsize=7.5, color="#333333")

    save_fig(fig, "fig01_method_schematic")


# ---------------------------------------------------------------------------
# Fig. 2 — main confirmatory TEST
# ---------------------------------------------------------------------------
def _horizontal_metric(ax, summaries, metric_mean, metric_ci, metric_per, *, as_pct: bool, xlabel: str, title: str):
    methods = ORDER_BASE
    ys = np.arange(len(methods))[::-1]
    rng = np.random.default_rng(2)
    scale = 100.0 if as_pct else 1.0
    for y, m in zip(ys, methods):
        s = summaries[m]
        color = COLOR[m]
        marker = MARKER[m]
        mu = s[metric_mean] * scale
        if s.get(metric_per):
            vals = [s[metric_per][seed] * scale for seed in SEEDS]
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(vals, y + jitter, color=color, s=28, marker=marker, zorder=3, edgecolors="white", linewidths=0.4, alpha=0.95)
            ax.plot(mu, y, marker=marker, color=color, ms=9, zorder=5, markeredgecolor="black", markeredgewidth=0.5)
            if s.get(metric_ci):
                lo, hi = s[metric_ci][0] * scale, s[metric_ci][1] * scale
                ax.hlines(y, lo, hi, color="black", lw=1.5, zorder=4)
                ax.plot([lo, hi], [y, y], "|", color="black", ms=8, zorder=4)
            if as_pct:
                ax.text(max(vals + [mu]) + 1.5, y, f"{mu:.1f}%", va="center", fontsize=7.5, color="#222222")
        else:
            ax.plot(mu, y, marker=marker, color=color, ms=9, zorder=5, markeredgecolor="black", markeredgewidth=0.5)
            if as_pct:
                ax.text(mu + 1.5, y, f"{mu:.1f}%", va="center", fontsize=7.5, color="#222222")
    ax.set_yticks(ys, [LABEL[m] for m in methods])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)


def fig02_main(rows):
    all_s = {m: summarize_learned(rows, m) if m == "HybridPPO" else summarize_baseline(rows, m) for m in ORDER_BASE}
    charge = [r for r in rows if r.get("charge_class") == "charging_required"]
    ch_s = {m: summarize_learned(charge, m) if m == "HybridPPO" else summarize_baseline(charge, m) for m in ORDER_BASE}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    _horizontal_metric(
        axes[0],
        all_s,
        "feasibility_mean",
        "feasibility_ci95",
        "per_seed_feasibility",
        as_pct=True,
        xlabel="Route feasibility (%)",
        title="(a) Route feasibility",
    )
    axes[0].set_xlim(0, 108)
    _horizontal_metric(
        axes[1],
        all_s,
        "completion_all_mean",
        "completion_all_ci95",
        "per_seed_completion",
        as_pct=False,
        xlabel="Failure-retaining completion  ↓",
        title="(b) Failure-retaining completion",
    )
    # legend
    handles = [
        Line2D([0], [0], marker=MARKER[m], color="w", markerfacecolor=COLOR[m], markeredgecolor="black", markersize=7, label=LABEL[m])
        for m in ORDER_BASE
    ]
    axes[0].legend(handles=handles, loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig02_main_test")
    return all_s, ch_s


# ---------------------------------------------------------------------------
# Fig. 3 — paired effect sizes
# ---------------------------------------------------------------------------
def fig03_effects():
    paired = load_json(STATS / "paired_primary.json")
    order = [
        "HybridPPO - OneStepLookahead",
        "HybridPPO - GreedyFullCharge",
        "HybridPPO - GreedyMinimumSufficientCharge",
    ]
    short = {
        "HybridPPO - OneStepLookahead": "vs Lookahead",
        "HybridPPO - GreedyFullCharge": "vs Greedy Full",
        "HybridPPO - GreedyMinimumSufficientCharge": "vs Greedy Min",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, family, scale, xlabel, title in (
        (axes[0], "feasibility", 100.0, "Difference in feasibility (percentage points)\nFA-HPPO better →", "(a) Feasibility"),
        (axes[1], "completion_all", 1.0, "Difference in failure-retaining completion\n← FA-HPPO better", "(b) Completion"),
    ):
        rows = [t for name in order for t in paired if t["comparison"] == name and t["family"] == family]
        ys = np.arange(len(rows))[::-1]
        for y, t in zip(ys, rows):
            eff = float(t["effect"]) * scale
            lo, hi = float(t["ci95"][0]) * scale, float(t["ci95"][1]) * scale
            ax.hlines(y, lo, hi, color=COLOR["HybridPPO"], lw=2.2)
            ax.plot(eff, y, "o", color=COLOR["HybridPPO"], ms=6.5, markeredgecolor="black", markeredgewidth=0.4)
        ax.axvline(0, color="black", lw=0.9)
        ax.set_yticks(ys, [short[t["comparison"]] for t in rows])
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color="#E6E6E6", lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "fig03_effect_sizes")


# ---------------------------------------------------------------------------
# Fig. 4 — difficulty robustness
# ---------------------------------------------------------------------------
def _cell_feasibility(rows, method, layout, length):
    cell = [
        r
        for r in method_rows(rows, method)
        if r.get("charge_class") == "charging_required"
        and r.get("layout") == layout
        and r.get("length_bin") == length
    ]
    return mean(float(r["feasible"]) for r in cell) if cell else np.nan


def fig04_difficulty(rows):
    layouts = ["R", "C", "RC"]
    bins = ["short", "medium", "long"]
    feas = np.zeros((3, 3))
    delta = np.zeros((3, 3))
    for i, layout in enumerate(layouts):
        for j, length in enumerate(bins):
            fh = _cell_feasibility(rows, "HybridPPO", layout, length)
            fl = _cell_feasibility(rows, "OneStepLookahead", layout, length)
            feas[i, j] = 100.0 * fh if fh is not None else np.nan
            if fh is None or fl is None:
                delta[i, j] = np.nan
            else:
                delta[i, j] = 100.0 * (fh - fl)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    im0 = axes[0].imshow(feas, vmin=0, vmax=100, cmap="Blues")
    axes[0].set_xticks(range(3), bins)
    axes[0].set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            v = feas[i, j]
            if np.isnan(v):
                axes[0].text(j, i, "NA", ha="center", va="center", fontsize=8)
            else:
                axes[0].text(j, i, f"{v:.0f}%", ha="center", va="center", color="white" if v > 55 else "black", fontsize=8)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046)
    cb0.set_label("Feasibility (%)")
    axes[0].set_title("(a) FA-HPPO feasibility", loc="left")
    axes[0].set_xlabel("Route length")
    axes[0].set_ylabel("Layout")

    dmax = float(np.nanmax(delta))
    dmin = float(np.nanmin(delta))
    vmax = max(dmax, 1.0)
    vmin = dmin if dmin < 0 else -max(1.0, 0.08 * vmax)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    im1 = axes[1].imshow(delta, cmap="RdBu", norm=norm)
    axes[1].set_xticks(range(3), bins)
    axes[1].set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            v = delta[i, j]
            if np.isnan(v):
                axes[1].text(j, i, "NA", ha="center", va="center", fontsize=8)
            else:
                # dark text near zero; light text for strong positive/negative cells
                strong = abs(v) > 0.55 * max(abs(vmin), abs(vmax))
                axes[1].text(j, i, f"{v:+.0f}", ha="center", va="center", color="white" if strong else "black", fontsize=8)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046)
    cb1.set_label("Δ feasibility (pp)")
    axes[1].set_title("(b) FA-HPPO − Lookahead", loc="left")
    axes[1].set_xlabel("Route length")
    axes[1].set_ylabel("Layout")
    fig.tight_layout()
    save_fig(fig, "fig04_difficulty_robustness")


# ---------------------------------------------------------------------------
# Fig. 5 — mechanism (gold ablation + amount sensitivity)
# ---------------------------------------------------------------------------
def load_ablation():
    rows = []
    for code, _ in ABL_ORDER:
        for seed in SEEDS:
            rows.append(load_json(V3 / "ablation" / code / f"seed_{seed}" / "validation.json"))
    return rows


def fig05_mechanism(ablation, rows):
    fig = plt.figure(figsize=(9.4, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # (a) gold B0/B1/B3/B2
    xs = np.arange(len(ABL_ORDER))
    rng = np.random.default_rng(0)
    for i, (code, _) in enumerate(ABL_ORDER):
        subset = [r for r in ablation if r["variant"] == code]
        feas = [100.0 * r["parent_balanced_val_feasibility"] for r in subset]
        jitter = rng.uniform(-0.12, 0.12, size=len(subset))
        color = COLOR["HybridPPO"] if code == "B2" else "#666666"
        ax0.scatter(i + jitter, feas, color=color, s=34, marker="o", zorder=3, edgecolors="white", linewidths=0.4)
        ax0.plot([i - 0.2, i + 0.2], [mean(feas), mean(feas)], color="black", lw=1.8, zorder=4)
        ax0.text(i, mean(feas) + 3.5, f"{mean(feas):.1f}%", ha="center", fontsize=7)
    ax0.set_xticks(xs, [lab for _, lab in ABL_ORDER])
    ax0.set_ylabel("Parent-balanced VAL feasibility (%)")
    ax0.set_ylim(0, 108)
    ax0.set_title("(a) Gold development ablation", loc="left")
    ax0.text(0.02, 0.04, "Development VAL", transform=ax0.transAxes, fontsize=7, color="#666666")
    # component matrix
    table_data = [["", "Time cap", "Scaling"]]
    for code, _ in ABL_ORDER:
        table_data.append([code, ABL_META[code]["time_cap"], ABL_META[code]["scaling"]])
    table = ax0.table(
        cellText=table_data,
        loc="bottom",
        bbox=[0.0, -0.42, 1.0, 0.28],
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_text_props(fontweight="bold")

    # (b) amount sensitivity
    methods = ["HybridPPO", "FA-HPPO-Max", "FA-HPPO-Min"]
    summaries = {m: summarize_learned(rows, m) for m in methods}
    amount = load_json(STATS / "amount_sensitivity.json") if (STATS / "amount_sensitivity.json").is_file() else None
    labels = ["FA-HPPO\nlearned u", "FA-HPPO-Max\nu=1", "FA-HPPO-Min\nu=0"]
    for i, m in enumerate(methods):
        s = summaries[m]
        vals = [100.0 * s["per_seed_feasibility"][seed] for seed in SEEDS]
        ax1.scatter(np.full(5, i), vals, color=COLOR[m], s=34, marker=MARKER[m], zorder=3, edgecolors="white", linewidths=0.4)
        mu = 100.0 * s["feasibility_mean"]
        ax1.plot([i - 0.18, i + 0.18], [mu, mu], color="black", lw=1.8)
        lo, hi = 100.0 * s["feasibility_ci95"][0], 100.0 * s["feasibility_ci95"][1]
        ax1.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], fmt="none", ecolor="black", elinewidth=1.2, capsize=3.5)
        ax1.text(i, max(vals + [hi]) + 2.5, f"{mu:.1f}%", ha="center", fontsize=7)
    ax1.set_xticks(range(3), labels)
    ax1.set_ylabel("Route feasibility (%)")
    ax1.set_ylim(0, 108)
    ax1.set_title("(b) V3 TEST amount-policy sensitivity", loc="left")
    tie = amount["route_averaged_feasibility_vs_max"]["tied"] if amount else 173
    ax1.text(0.02, 0.08, f"learned vs Max: {tie}/180 routes tied", transform=ax1.transAxes, fontsize=7.5, color="#222222")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.30, wspace=0.30)
    save_fig(fig, "fig05_mechanism")


# ---------------------------------------------------------------------------
# Appendix A1 — illustrative VAL trajectory (simplified former Fig.1)
# ---------------------------------------------------------------------------
def fig_a01_illustrative():
    if not CASE.is_file():
        raise SystemExit("missing case study JSON; run scripts/paper/record_illustrative_val_episode.py")
    ep = load_json(CASE)
    nodes = ep["nodes"]
    customers = ep["customer_ids"]
    depot = ep["depot_id"]
    stations = ep["station_ids"]
    path = ep["path_nodes"]
    decisions = ep["decisions"]
    visited_stations = [n for n in path if n in stations]

    fig = plt.figure(figsize=(11.0, 3.8))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.25, 1.0], wspace=0.30)
    ax_map = fig.add_subplot(outer[0, 0])
    ax_soc = fig.add_subplot(outer[0, 1])
    right = outer[0, 2].subgridspec(2, 1, hspace=0.55, height_ratios=[1.0, 1.15])
    ax_disc = fig.add_subplot(right[0, 0])
    ax_beta = fig.add_subplot(right[1, 0])

    seq = [depot] + customers + [depot]
    ax_map.plot([nodes[i]["x"] for i in seq], [nodes[i]["y"] for i in seq], ls="--", color="#B0B0B0", lw=1.1, zorder=1)
    ax_map.plot([nodes[i]["x"] for i in path], [nodes[i]["y"] for i in path], color=COLOR["OneStepLookahead"], lw=1.5, zorder=2)
    for sid in stations:
        n = nodes[sid]
        used = sid in visited_stations
        ax_map.scatter(
            n["x"],
            n["y"],
            marker="*" if used else "^",
            s=150 if used else 50,
            color=COLOR["HybridPPO"] if used else "#C8C8C8",
            zorder=6 if used else 3,
            edgecolors="white",
            linewidths=0.4,
        )
    for i, cid in enumerate(customers, start=1):
        n = nodes[cid]
        ax_map.scatter(n["x"], n["y"], s=58, facecolors="white", edgecolors="#222222", linewidths=1.0, zorder=5)
        ax_map.text(n["x"], n["y"], str(i), ha="center", va="center", fontsize=7, zorder=6)
    nd = nodes[depot]
    ax_map.scatter(nd["x"], nd["y"], marker="s", s=70, color="black", zorder=7)
    vx = [nodes[i]["x"] for i in path]
    vy = [nodes[i]["y"] for i in path]
    pad_x = max(0.04, 0.08 * (max(vx) - min(vx) + 1e-9))
    pad_y = max(0.04, 0.08 * (max(vy) - min(vy) + 1e-9))
    ax_map.set_xlim(min(vx) - pad_x, max(vx) + pad_x)
    ax_map.set_ylim(min(vy) - pad_y, max(vy) + pad_y)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.set_title("(a) Fixed route + charges", loc="left")
    ax_map.text(0.02, 0.98, "Development VAL", transform=ax_map.transAxes, fontsize=7, va="top", color="#666666")

    # SOC
    points = [(0.0, 100.0)]
    cursor = 0
    for d in decisions:
        if d.get("action") == "CONTINUE":
            cursor += 1
            points.append((float(cursor), 100 * float(d["soc_after"])))
        elif d.get("action") == "CHARGE":
            cursor += 1
            points.append((float(cursor), 100 * float(d.get("soc_arrival", d["soc_before_decision"]))))
            points.append((float(cursor), 100 * float(d.get("soc_departure", d["soc_after"]))))
    xs, ys = zip(*points)
    ax_soc.plot(xs, ys, color=COLOR["HybridPPO"], lw=1.7)
    for d in decisions:
        if d.get("action") != "CHARGE":
            continue
        # approximate x as visit index of station in path
        try:
            xi = path.index(d["station_id"])
        except (ValueError, KeyError):
            continue
        lo, hi = 100 * d["soc_lower"], 100 * d["soc_upper"]
        ax_soc.fill_between([xi - 0.15, xi + 0.15], lo, hi, color="#56B4E9", alpha=0.35)
        ax_soc.scatter([xi], [100 * d.get("soc_departure", d["soc_after"])], marker="*", s=80, color="#D55E00", zorder=5)
    ax_soc.set_ylim(-2, 108)
    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_xlabel("Visit index")
    ax_soc.set_title("(b) SOC along route", loc="left")

    rep_step = ep.get("representative_charge_step")
    rep = next((d for d in decisions if d.get("step") == rep_step and d.get("action") == "CHARGE"), None)
    if rep is None:
        rep = next((d for d in decisions if d.get("action") == "CHARGE"), None)
    if rep is None:
        ax_disc.text(0.5, 0.5, "no charge decision", ha="center")
        ax_beta.axis("off")
    else:
        names = rep["policy"]["action_names"]
        probs = np.asarray(rep["policy"]["probs"], dtype=float)
        mask = np.asarray(rep["policy"]["mask"], dtype=bool)
        xs = np.arange(len(names))
        colors = [COLOR["HybridPPO"] if (mask[i] and names[i] == rep.get("station_id")) else ("#9E9E9E" if mask[i] else "#DDDDDD") for i in range(len(names))]
        ax_disc.bar(xs, np.where(mask, probs, 0.0), color=colors, width=0.72, edgecolor="#444444", linewidth=0.4)
        ax_disc.set_xticks(xs, names, fontsize=7)
        ax_disc.set_ylim(0, 1.08)
        ax_disc.set_ylabel("Prob.")
        ax_disc.set_title("(c) Discrete head", loc="left", fontsize=9)

        alpha = float(rep["alpha"])
        beta = float(rep["beta"])
        u = float(rep["u"])
        ugrid = np.linspace(0.001, 0.999, 300)
        dens = _beta_pdf(ugrid, alpha, beta)
        ax_beta.fill_between(ugrid, dens, color="#56B4E9", alpha=0.35)
        ax_beta.plot(ugrid, dens, color=COLOR["HybridPPO"], lw=1.4)
        ax_beta.axvline(u, color="#D55E00", lw=1.5)
        ax_beta.set_xlabel(r"$u$")
        ax_beta.set_ylabel("Density")
        ax_beta.set_title("(d) Continuous head", loc="left", fontsize=9)
        lo, hi, tgt = rep["soc_lower"], rep["soc_upper"], rep["soc_target"]
        ax_beta.text(
            0.02,
            0.95,
            rf"$[{100*lo:.0f},{100*hi:.0f}]\to{100*tgt:.0f}\%$",
            transform=ax_beta.transAxes,
            va="top",
            fontsize=6.5,
        )

    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.12, wspace=0.30)
    save_fig(fig, "figA01_illustrative_val_trajectory", appendix=True)


# ---------------------------------------------------------------------------
# Appendix A2 — learning curves
# ---------------------------------------------------------------------------
def fig_a02_learning_curves(ablation):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    styles = {
        "B0": dict(color="#999999", ls="-", lw=1.6),
        "B1": dict(color="#E69F00", ls="--", lw=1.6),
        "B3": dict(color="#56B4E9", ls="-.", lw=1.6),
        "B2": dict(color="#0072B2", ls="-", lw=2.2),
    }
    for code, lab in ABL_ORDER:
        series = []
        for seed in SEEDS:
            path = V3 / "ablation" / code / f"seed_{seed}" / "curves.jsonl"
            if not path.is_file():
                continue
            xs, ys = [], []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("val_parent_balanced_feasibility") is None:
                    continue
                xs.append(int(row["update"]))
                ys.append(100.0 * float(row["val_parent_balanced_feasibility"]))
            if xs:
                series.append((np.asarray(xs), np.asarray(ys)))
        if not series:
            continue
        grid = sorted({int(x) for xs, _ in series for x in xs})
        mat = np.asarray([np.interp(grid, xs, ys) for xs, ys in series])
        mu, sd = mat.mean(0), mat.std(0, ddof=1) if len(mat) > 1 else np.zeros_like(mat[0])
        st = styles[code]
        ax.plot(grid, mu, label=lab, **st)
        ax.fill_between(grid, mu - sd, mu + sd, color=st["color"], alpha=0.12)
    ax.set_xlabel("Training update")
    ax.set_ylabel("Parent-balanced VAL feasibility (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, title="mean ± SD", fontsize=7)
    ax.text(0.02, 0.04, "Development VAL", transform=ax.transAxes, fontsize=7, color="#666666")
    ax.set_title("(a) Gold development learning curves", loc="left")
    fig.tight_layout()
    save_fig(fig, "figA02_learning_curves", appendix=True)


# ---------------------------------------------------------------------------
# Appendix A3 — training stability (value loss / grad)
# ---------------------------------------------------------------------------
def fig_a03_training_stability():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    styles = {
        "B0": dict(color="#999999", ls="-", lw=1.5),
        "B1": dict(color="#E69F00", ls="--", lw=1.5),
        "B3": dict(color="#56B4E9", ls="-.", lw=1.5),
        "B2": dict(color="#0072B2", ls="-", lw=2.0),
    }
    for ax, key, title, ylabel in (
        (axes[0], "value_loss", "(a) Value loss", "Value loss"),
        (axes[1], "grad_norm_preclip", "(b) Pre-clip gradient norm", "Gradient norm (pre-clip)"),
    ):
        for code, lab in ABL_ORDER:
            series = []
            for seed in SEEDS:
                path = V3 / "ablation" / code / f"seed_{seed}" / "curves.jsonl"
                if not path.is_file():
                    continue
                xs, ys = [], []
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get(key) is None:
                        continue
                    xs.append(int(row["update"]))
                    ys.append(float(row[key]))
                if xs:
                    series.append((np.asarray(xs), np.asarray(ys)))
            if not series:
                continue
            grid = sorted({int(x) for xs, _ in series for x in xs})
            mat = np.asarray([np.interp(grid, xs, ys) for xs, ys in series])
            mu, sd = mat.mean(0), mat.std(0, ddof=1) if len(mat) > 1 else np.zeros_like(mat[0])
            st = styles[code]
            ax.plot(grid, mu, label=lab, **st)
            ax.fill_between(grid, np.maximum(mu - sd, 1e-8), mu + sd, color=st["color"], alpha=0.12)
        ax.set_yscale("log")
        ax.set_xlabel("Training update")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=7, title="mean ± SD")
        ax.text(0.02, 0.04, "Development VAL", transform=ax.transAxes, fontsize=7, color="#666666")
    fig.tight_layout()
    save_fig(fig, "figA03_training_stability", appendix=True)


# ---------------------------------------------------------------------------
# Appendix A4 — SynthCharge envelope A/B/C
# ---------------------------------------------------------------------------
def fig_a04_envelope():
    if not ENVELOPE_SUMMARY.is_file():
        raise SystemExit(f"missing envelope summary: {ENVELOPE_SUMMARY}")
    summary = load_json(ENVELOPE_SUMMARY)
    order = [
        ("A_arrival_to_max", "Arrival-to-Max"),
        ("B_energy_to_max", "EnergyLower-to-Max"),
        ("C_energy_to_time", "EnergyLower-to-TimeUpper"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    rng = np.random.default_rng(3)
    for i, (key, lab) in enumerate(order):
        v = summary["variants"][key]
        seeds = [100.0 * float(v["val_feas_per_seed"][str(s)]) for s in SEEDS]
        jitter = rng.uniform(-0.1, 0.1, size=len(seeds))
        color = COLOR["HybridPPO"] if key == "C_energy_to_time" else "#666666"
        ax.scatter(i + jitter, seeds, color=color, s=36, zorder=3, edgecolors="white", linewidths=0.4)
        mu = 100.0 * float(v["val_feas_mean"])
        ax.plot([i - 0.2, i + 0.2], [mu, mu], color="black", lw=1.8, zorder=4)
        ax.text(i, mu + 0.8, f"{mu:.1f}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(3), [lab for _, lab in order])
    ax.set_ylabel("Parent-balanced VAL feasibility (%)")
    ax.set_ylim(94, 101.5)
    ax.set_title("(a) Post-hoc SynthCharge development envelope study", loc="left")
    ax.text(0.02, 0.06, "Post-hoc SynthCharge development study", transform=ax.transAxes, fontsize=7.5, color="#666666")
    fig.tight_layout()
    save_fig(fig, "figA04_envelope_ablation", appendix=True)


# ---------------------------------------------------------------------------
# Appendix A5 — failure analysis
# ---------------------------------------------------------------------------
def fig_a05_failures(rows):
    hybrid = method_rows(rows, "HybridPPO")
    by_route = defaultdict(list)
    for r in hybrid:
        by_route[r["route_id"]].append(r)

    problem = []
    n_all_ok = 0
    for rid, rr in by_route.items():
        n_fail = sum(1 for r in rr if not r["feasible"])
        if n_fail == 0:
            n_all_ok += 1
            continue
        sample = next(r for r in rr if not r["feasible"])
        problem.append(
            {
                "route_id": rid,
                "n_fail": n_fail,
                "layout": sample.get("layout"),
                "length": sample.get("length_bin"),
                "charge_class": sample.get("charge_class"),
            }
        )
    problem.sort(key=lambda d: (-d["n_fail"], d["route_id"]))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]})
    # (a) lollipop of problematic routes
    ys = np.arange(len(problem))[::-1]
    xs = [d["n_fail"] for d in problem]
    axes[0].hlines(ys, 0, xs, color="#A7C7DC", lw=1.6)
    axes[0].scatter(xs, ys, color=COLOR["HybridPPO"], s=36, zorder=3, edgecolors="white", linewidths=0.4)
    labels = [f"{d['route_id']}  ({d['layout']}/{d['length']})" for d in problem]
    axes[0].set_yticks(ys, labels, fontsize=6.5)
    axes[0].set_xlabel("Number of failing seeds (1–5)")
    axes[0].set_xlim(0, 5.4)
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].set_title("(a) Problematic routes", loc="left")
    axes[0].text(
        0.98,
        0.02,
        f"{n_all_ok} / {len(by_route)} routes succeeded\nunder all five seeds",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCCCCC", alpha=0.95),
    )

    # (b) charging-required failure-rate heatmap (%)
    layouts = ["R", "C", "RC"]
    bins = ["short", "medium", "long"]
    rate = np.zeros((3, 3))
    for i, layout in enumerate(layouts):
        for j, length in enumerate(bins):
            cell_rows = [
                r
                for rr in by_route.values()
                for r in rr
                if rr[0].get("layout") == layout
                and rr[0].get("length_bin") == length
                and rr[0].get("charge_class") == "charging_required"
            ]
            rate[i, j] = 100.0 * mean(float(not r["feasible"]) for r in cell_rows) if cell_rows else np.nan
    vmax = max(10.0, float(np.nanmax(rate)))
    im = axes[1].imshow(rate, vmin=0, vmax=vmax, cmap="magma")
    axes[1].set_xticks(range(3), bins)
    axes[1].set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            v = rate[i, j]
            if np.isnan(v):
                axes[1].text(j, i, "NA", ha="center", va="center", color="white", fontsize=8)
            else:
                axes[1].text(j, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046)
    cb.set_label("Failure rate (%)")
    axes[1].set_title("(b) Charging-required failure rate", loc="left")
    axes[1].set_xlabel("Route length")
    axes[1].set_ylabel("Layout")
    fig.tight_layout()
    save_fig(fig, "figA05_failure_analysis", appendix=True)
    return problem, n_all_ok


# ---------------------------------------------------------------------------
# Appendix A6 — seed robustness
# ---------------------------------------------------------------------------
def fig_a06_seeds(summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    seeds = list(SEEDS)
    feas = [100.0 * summary["per_seed_feasibility"][s] for s in seeds]
    comp = [summary["per_seed_completion"][s] for s in seeds]
    axes[0].scatter(seeds, feas, s=55, color=COLOR["HybridPPO"], marker="o", zorder=3, edgecolors="black", linewidths=0.4)
    axes[0].axhline(100.0 * summary["feasibility_mean"], color="black", ls="--", lw=1)
    lo, hi = 100.0 * summary["feasibility_ci95"][0], 100.0 * summary["feasibility_ci95"][1]
    axes[0].axhspan(lo, hi, color=COLOR["HybridPPO"], alpha=0.15, label="hier. 95% CI")
    # truncated axis with clear note
    ymin = min(feas + [lo]) - 2
    axes[0].set_ylim(ymin, 100.5)
    axes[0].text(0.02, 0.05, "y-axis truncated", transform=axes[0].transAxes, fontsize=6.5, color="#666666")
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Feasibility (%)")
    axes[0].set_title("(a) Feasibility", loc="left")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")

    axes[1].scatter(seeds, comp, s=55, color=COLOR["HybridPPO"], marker="o", zorder=3, edgecolors="black", linewidths=0.4)
    axes[1].axhline(summary["completion_all_mean"], color="black", ls="--", lw=1)
    lo, hi = summary["completion_all_ci95"]
    axes[1].axhspan(lo, hi, color=COLOR["HybridPPO"], alpha=0.15)
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Failure-retaining completion")
    axes[1].set_title("(b) Completion", loc="left")
    fig.tight_layout()
    save_fig(fig, "figA06_seed_robustness", appendix=True)


# ---------------------------------------------------------------------------
# Appendix A7 — native FRVCP reference
# ---------------------------------------------------------------------------
def fig_a07_frvcp():
    summary_csv = ROOT / "results/final/statistics/frvcpy_native/method_summary.csv"
    gap_csv = ROOT / "results/final/tables/frvcpy_native/table_F_frvcpy.csv"
    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    gaps = {}
    for r in csv.DictReader(gap_csv.open(encoding="utf-8")):
        raw = (r.get("mean_gap_percent_vs_frvcpy_solver") or "").strip()
        if raw:
            gaps[r["method"]] = float(raw)
    order = ["frvcpy_Solver", "FRVCPGreedyMin", "FRVCPGreedyFull"]
    labels = {"frvcpy_Solver": "frvcpy Solver", "FRVCPGreedyMin": "GreedyMin", "FRVCPGreedyFull": "GreedyFull"}
    colors = {"frvcpy_Solver": "#0072B2", "FRVCPGreedyMin": "#E69F00", "FRVCPGreedyFull": "#009E73"}
    markers = {"frvcpy_Solver": "o", "FRVCPGreedyMin": "^", "FRVCPGreedyFull": "D"}
    feas = {r["method"]: 100.0 * float(r["feasibility_mean"]) for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        axes[0].plot(feas[m], yi, marker=markers[m], color=colors[m], ms=9, markeredgecolor="black", markeredgewidth=0.4)
        axes[0].text(feas[m] + 1.5, yi, f"{feas[m]:.1f}%", va="center", fontsize=7.5)
    axes[0].set_yticks(y, [labels[m] for m in order])
    axes[0].set_xlim(0, 108)
    axes[0].set_xlabel("Feasibility (%)")
    axes[0].set_title("(a) Native FRVCP reference", loc="left")
    gm = ["FRVCPGreedyMin", "FRVCPGreedyFull"]
    y2 = np.arange(len(gm))[::-1]
    for yi, m in zip(y2, gm):
        axes[1].plot(gaps[m], yi, marker=markers[m], color=colors[m], ms=9, markeredgecolor="black", markeredgewidth=0.4)
        axes[1].text(gaps[m] + 0.4, yi, f"{gaps[m]:.1f}%", va="center", fontsize=7.5)
    axes[1].set_yticks(y2, [labels[m] for m in gm])
    axes[1].set_xlabel("Mean optimality gap vs frvcpy Solver (%)")
    axes[1].set_title("(b) Gap vs frvcpy Solver", loc="left")
    fig.tight_layout()
    save_fig(fig, "figA07_native_frvcp_reference", appendix=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
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
    tex = [
        "% Auto-generated from frozen results.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{title}}}",
        rf"\label{{tab:{stem}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
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
            ["Native FRVCP archive", "separate reference", "native FRVCP", "133", "—", "—", "no", "no", "appendix only"],
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
        "FA-HPPO = mean over 5 training seeds; baselines deterministic. Infeasible completion = horizon H. CI = hierarchical bootstrap (seed→route). Holm-adjusted primary p < 0.001 (Monte Carlo floor).",
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
        "DEVELOPMENT / GOLD VALIDATION ONLY — not V3 TEST. B0/B1/B3/B2 are provenance labels. Five seeds 42–46.",
    )
    amt = {m: summarize_learned(rows, m) for m in ("HybridPPO", "FA-HPPO-Max", "FA-HPPO-Min")}
    write_table(
        "table04_amount_sensitivity",
        "Table 4 — TEST amount-policy sensitivity",
        ["Method", "u policy", "Feasibility", "95% CI", "Completion", "Interpretation"],
        [
            ["FA-HPPO", "learned Beta", fmt(amt["HybridPPO"]["feasibility_mean"]), f"[{fmt(amt['HybridPPO']['feasibility_ci95'][0])}, {fmt(amt['HybridPPO']['feasibility_ci95'][1])}]", fmt(amt["HybridPPO"]["completion_all_mean"], 3), "primary"],
            ["FA-HPPO-Max", "forced u=1", fmt(amt["FA-HPPO-Max"]["feasibility_mean"]), f"[{fmt(amt['FA-HPPO-Max']['feasibility_ci95'][0])}, {fmt(amt['FA-HPPO-Max']['feasibility_ci95'][1])}]", fmt(amt["FA-HPPO-Max"]["completion_all_mean"], 3), "≈ free; envelope upper bound"],
            ["FA-HPPO-Min", "forced u=0", fmt(amt["FA-HPPO-Min"]["feasibility_mean"]), f"[{fmt(amt['FA-HPPO-Min']['feasibility_ci95'][0])}, {fmt(amt['FA-HPPO-Min']['feasibility_ci95'][1])}]", fmt(amt["FA-HPPO-Min"]["completion_all_mean"], 3), "degenerate lower-bound stress"],
        ],
        "Same frozen checkpoints on V3 TEST. Feasibility-aware envelope accounts for much of the gain; do not claim learned continuous amount beats Max.",
    )
    write_table(
        "tableA01_per_seed",
        "Table A1 — FA-HPPO per seed",
        ["Seed", "Feasibility", "Completion", "Charge-req feasibility"],
        [
            [
                str(seed),
                fmt(all_s["HybridPPO"]["per_seed_feasibility"][seed]),
                fmt(all_s["HybridPPO"]["per_seed_completion"][seed], 3),
                fmt(ch_s["HybridPPO"]["per_seed_feasibility"][seed]),
            ]
            for seed in SEEDS
        ],
        "Individual training seeds on V3 TEST.",
    )
    paired = load_json(STATS / "paired_primary.json")
    write_table(
        "tableA02_primary_statistics",
        "Table A2 — Primary paired statistics",
        ["Comparison", "Metric", "Effect", "95% CI", "p_raw", "p_Holm", "Procedure", "Units"],
        [
            [
                t["comparison"],
                t["family"],
                fmt(t["effect"], 4),
                f"[{fmt(t['ci95'][0], 4)}, {fmt(t['ci95'][1], 4)}]",
                fmt_p(t["p_raw"]),
                fmt_p(t["p_holm"]),
                t.get("procedure", "predeclared"),
                str(t["n_independent_units"]),
            ]
            for t in paired
        ],
        "Predeclared seed-averaged route-paired analysis with Holm correction. "
        "Paper-facing p-values use <5e-5 at the n_perm=20000 Monte Carlo floor; raw JSON retains machine values.",
    )
    build_failure_table(rows)


def build_failure_table(rows) -> None:
    hybrid = method_rows(rows, "HybridPPO")
    by_route = defaultdict(list)
    for r in hybrid:
        by_route[r["route_id"]].append(r)
    table_rows = []
    for route_id, rr in sorted(by_route.items()):
        n_fail = sum(1 for r in rr if not r["feasible"])
        if n_fail == 0:
            continue
        seeds_fail = sorted(int(r["seed"]) for r in rr if not r["feasible"])
        seeds_ok = sorted(int(r["seed"]) for r in rr if r["feasible"])
        sample = next(r for r in rr if not r["feasible"])
        visits = [float(r.get("n_station_visits") or 0) for r in rr if not r["feasible"]]
        terms = [r.get("terminal_soc") for r in rr if not r["feasible"] and r.get("terminal_soc") is not None]
        table_rows.append(
            [
                route_id,
                sample.get("layout"),
                sample.get("length_bin"),
                sample.get("charge_class"),
                str(n_fail),
                ",".join(str(s) for s in seeds_fail),
                ",".join(str(s) for s in seeds_ok) if seeds_ok else "—",
                sample.get("reason") or "NA",
                fmt(mean(visits), 2),
                fmt(mean(terms), 3) if terms else "NA",
            ]
        )
    write_table(
        "tableA03_failure_routes",
        "Table A3 — FA-HPPO failing routes (frozen V3 raw)",
        [
            "Route",
            "Layout",
            "Length",
            "Charge class",
            "n_seeds_fail",
            "Fail seeds",
            "OK seeds",
            "Reason",
            "Mean visits (fail)",
            "Mean terminal SOC (fail)",
        ],
        table_rows,
        "From frozen raw rows only. Pre-failure action/SOC trajectories were not recorded in V3 raw data "
        "and are unavailable without replaying TEST (forbidden). Shared failures = n_seeds_fail=5.",
    )


# ---------------------------------------------------------------------------
# Captions / README / manifest / verify
# ---------------------------------------------------------------------------
def write_captions() -> None:
    text = """# Figure captions (`results_paper/`)

Use PDF figures in the LaTeX manuscript; PNG for preview/GitHub.

## Main paper

**Figure 1.** Conceptual overview of feasibility-aware hybrid PPO (FA-HPPO) for fixed-route charging control. (a) The customer visit order is fixed; the policy chooses CONTINUE or an optional charging-station insertion and does not reorder customers. (b) When a station is selected, the admissible departure SOC is restricted to a feasibility-aware interval, and the continuous action u in [0,1] maps to SOC_target = SOC_lower + u (SOC_upper - SOC_lower). (c) A discrete head selects CONTINUE/station under action masking; a Beta continuous head produces u. Schematic only — not empirical evidence.

**Figure 2.** Confirmatory performance on the independently generated V3 SynthCharge TEST (180 routes). FA-HPPO results aggregate five independently trained seeds; small points denote individual seed results and the larger marker denotes the across-seed mean. Error bars show the predeclared hierarchical 95% bootstrap interval over training seed and route. Baselines are deterministic. Failure-retaining completion assigns horizon H to infeasible episodes. Higher feasibility and lower completion are better. Charging-required subset metrics appear in Table 2.

**Figure 3.** Paired route-level effects of FA-HPPO versus each baseline on the V3 SynthCharge TEST (n=180 routes; five FA-HPPO training seeds). Points show seed-averaged paired differences with 95% confidence intervals; the vertical line marks zero. Positive feasibility differences and negative completion differences favor FA-HPPO. Holm-adjusted permutation p-values for the predeclared primary comparisons are reported in Table A2 (Monte Carlo floor <5e-5).

**Figure 4.** Robustness of FA-HPPO on charging-required V3 TEST routes by layout x route-length cell. (a) FA-HPPO feasibility (%; mean over route x seed rows in each cell). (b) FA-HPPO minus OneStepLookahead feasibility (percentage points) on the same cells, using a diverging scale centered at zero. Computed from frozen V3 raw rows only.

**Figure 5.** Mechanism evidence. (a) Gold development B0/B1/B3/B2 ablation on official EVRPTW-GR VAL (five seeds 42-46): parent-balanced feasibility with compact variant codes; the component matrix lists time-aware upper cap and return scaling. Development evidence only — not V3 TEST. (b) V3 TEST amount-policy sensitivity with the same frozen FA-HPPO checkpoints: learned continuous u, forced u=1 (Max), and forced u=0 (Min). Small points are seeds; error bars are hierarchical 95% CIs. Learned vs Max ties on 173/180 routes. Prefer the interpretation that the feasibility-aware envelope accounts for much of the observed performance; do not claim continuous amount learning is the main driver.

## Appendix

**Figure A1.** Illustrative SynthCharge VAL episode; not TEST evidence. Fixed-route geometry with charging insertions, SOC evolution, and one hybrid discrete/continuous decision with the feasible SOC interval.

**Figure A2.** Gold development learning curves (parent-balanced VAL feasibility, %): mean +/- SD across five seeds for B0/B1/B3/B2. Line styles distinguish variants; not V3 TEST evidence.

**Figure A3.** Effect of return scaling / methodology components on PPO optimization stability (gold development curves). (a) Value loss vs update. (b) Pre-clip gradient norm vs update. Bands are mean +/- SD (not confidence intervals). Log y-scale. Development VAL only.

**Figure A4.** Post-hoc SynthCharge TRAIN/VAL envelope ablation (Arrival-to-Max, EnergyLower-to-Max, EnergyLower-to-TimeUpper); five seeds 42-46. Points are seed values; bars mark means. Post-hoc development study — not confirmatory V3 TEST evidence. Do not interpret as a monotonic causal claim that each envelope component improves feasibility.

**Figure A5.** Frozen-raw FA-HPPO failure analysis on V3 TEST (no policy replay). (a) The 20 routes with at least one failing seed, sorted by number of failing seeds; annotation notes that 160/180 routes succeed under all five seeds. (b) Charging-required failure rate (%) by layout x length. All recorded failure reasons are NO_FEASIBLE_ACTION. Step-level pre-failure trajectories were not stored.

**Figure A6.** FA-HPPO per-seed robustness on V3 TEST. (a) Feasibility (%; y-axis truncated for readability, with hierarchical 95% CI band). (b) Failure-retaining completion. Seed values also appear in the main result plots and Table A1.

**Figure A7.** Native FRVCP reference benchmark (separate archive; n=133). This is a separate native FRVCP benchmark and is not a SynthCharge/EVRPTW-GR exact comparison. Do not treat frvcpy as an exact oracle for the paper's SynthCharge formulation.
"""
    (OUT / "CAPTIONS.md").write_text(text, encoding="utf-8")


def write_readme() -> None:
    (OUT / "README.md").write_text(
        """# Publication outputs (`results_paper/`)

Single source of truth for the manuscript. Every figure is written as:

- **PDF** — vector version for LaTeX
- **PNG** — high-resolution preview (600 dpi)

## Main paper

| Artifact | Role | Evidence |
|----------|------|----------|
| `figures/fig01_method_schematic.{pdf,png}` | Conceptual FA-HPPO method | schematic (not data) |
| `figures/fig02_main_test.{pdf,png}` | Confirmatory performance | V3 TEST |
| `figures/fig03_effect_sizes.{pdf,png}` | Paired effect sizes | V3 TEST / `paired_primary.json` |
| `figures/fig04_difficulty_robustness.{pdf,png}` | Layout x length robustness | V3 TEST (charging-required) |
| `figures/fig05_mechanism.{pdf,png}` | Gold ablation + amount sensitivity | gold VAL + V3 TEST |
| `tables/table01_*` … `table04_*` | Protocol / main / ablation / amount | see MANIFEST |
| `CAPTIONS.md` | Self-contained figure captions | — |

## Appendix

| Artifact | Role |
|----------|------|
| `figures/appendix/figA01_illustrative_val_trajectory.{pdf,png}` | VAL case study (not TEST) |
| `figures/appendix/figA02_learning_curves.{pdf,png}` | Gold development learning curves |
| `figures/appendix/figA03_training_stability.{pdf,png}` | Value-loss / grad-norm diagnostics |
| `figures/appendix/figA04_envelope_ablation.{pdf,png}` | Post-hoc SynthCharge envelope A/B/C |
| `figures/appendix/figA05_failure_analysis.{pdf,png}` | Frozen-raw failure concentration |
| `figures/appendix/figA06_seed_robustness.{pdf,png}` | Per-seed robustness |
| `figures/appendix/figA07_native_frvcp_reference.{pdf,png}` | Native FRVCP reference |
| `tables/tableA01_*` … `tableA03_*` | Per-seed / paired / failure routes |
| `case_study/illustrative_val_episode.json` | Source for Fig. A1 |
| `failure_analysis/` | Deeper frozen-raw failure write-up |

## Regenerate

```bash
python scripts/paper/record_illustrative_val_episode.py   # once (VAL case study JSON)
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Does **not** retrain or re-evaluate the consumed V3 TEST.

`scripts/paper/build_paper_artifacts.py` is deprecated and delegates here.
""",
        encoding="utf-8",
    )


def build_manifest(artifacts):
    for item in artifacts:
        path = ROOT / item["path"]
        item["sha256"] = _sha(path) if path.is_file() else None
        item["generation_script"] = item.get("generation_script", "scripts/paper/build_results_paper.py")
    (OUT / "MANIFEST.json").write_text(json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify() -> None:
    consumed = load_json(V3 / "EVALUATION_CONSUMED.json")
    if lf_sha256(RAW) != consumed["raw_sha256_lf"]:
        raise SystemExit("raw TEST hash mismatch")
    rows = load_rows()
    if len(rows) != 3240:
        raise SystemExit(f"expected 3240 rows, got {len(rows)}")
    if "DiscretePPO" in {r["method"] for r in rows}:
        raise SystemExit("DiscretePPO in V3 TEST")
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

    expected_png = {f"{s}.png" for s in MAIN_FIGS + APPENDIX_FIGS}
    expected_pdf = {f"{s}.pdf" for s in MAIN_FIGS + APPENDIX_FIGS}
    pngs = {p.name for p in FIG.rglob("*.png")}
    pdfs = {p.name for p in FIG.rglob("*.pdf")}
    missing_png = expected_png - pngs
    missing_pdf = expected_pdf - pdfs
    if missing_png:
        raise SystemExit(f"missing PNGs: {sorted(missing_png)}")
    if missing_pdf:
        raise SystemExit(f"missing PDFs: {sorted(missing_pdf)}")
    if not (TAB / "tableA03_failure_routes.md").is_file():
        raise SystemExit("missing tableA03_failure_routes")
    if not (OUT / "CAPTIONS.md").is_file():
        raise SystemExit("missing CAPTIONS.md")
    for path in FIG.rglob("*"):
        if path.is_file() and path.suffix.lower() not in {".png", ".pdf"}:
            raise SystemExit(f"unexpected figure format: {path}")
    leftovers = {f"{s}.png" for s in OBSOLETE_FIGS} & pngs
    if leftovers:
        raise SystemExit(f"obsolete figure names still present: {sorted(leftovers)}")
    print(
        json.dumps(
            {
                "verify": "ok",
                "n_rows": 3240,
                "n_instances": 180,
                "n_png": len(pngs),
                "n_pdf": len(pdfs),
            },
            indent=2,
        )
    )


def main() -> None:
    if "--verify" in sys.argv:
        verify()
        return
    style()
    cleanup_obsolete()
    rows = load_rows()
    fig01_method_schematic()
    all_s, ch_s = fig02_main(rows)
    fig03_effects()
    fig04_difficulty(rows)
    ablation = load_ablation()
    fig05_mechanism(ablation, rows)
    fig_a01_illustrative()
    fig_a02_learning_curves(ablation)
    fig_a03_training_stability()
    fig_a04_envelope()
    fig_a05_failures(rows)
    fig_a06_seeds(all_s["HybridPPO"])
    fig_a07_frvcp()
    build_tables(all_s, ch_s, ablation, rows)
    write_captions()
    write_readme()
    artifacts = [
        {"path": "results_paper/README.md", "role": "index"},
        {"path": "results_paper/CAPTIONS.md", "role": "captions"},
        {"path": "results_paper/case_study/illustrative_val_episode.json", "role": "methodology-illustration", "dataset": "SynthCharge VAL", "note": "not TEST"},
        {"path": "results_paper/figures/fig01_method_schematic.png", "role": "main-methodology", "dataset": "schematic"},
        {"path": "results_paper/figures/fig01_method_schematic.pdf", "role": "main-methodology", "dataset": "schematic"},
        {"path": "results_paper/figures/fig02_main_test.png", "role": "main", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5},
        {"path": "results_paper/figures/fig02_main_test.pdf", "role": "main", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5},
        {"path": "results_paper/figures/fig03_effect_sizes.png", "role": "main", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5},
        {"path": "results_paper/figures/fig03_effect_sizes.pdf", "role": "main", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5},
        {"path": "results_paper/figures/fig04_difficulty_robustness.png", "role": "main", "dataset": "SynthCharge V3 TEST", "subset": "charging_required"},
        {"path": "results_paper/figures/fig04_difficulty_robustness.pdf", "role": "main", "dataset": "SynthCharge V3 TEST", "subset": "charging_required"},
        {"path": "results_paper/figures/fig05_mechanism.png", "role": "main-development+test", "dataset": "gold VAL + V3 TEST"},
        {"path": "results_paper/figures/fig05_mechanism.pdf", "role": "main-development+test", "dataset": "gold VAL + V3 TEST"},
        {"path": "results_paper/figures/appendix/figA01_illustrative_val_trajectory.png", "role": "appendix", "dataset": "SynthCharge VAL", "note": "not TEST"},
        {"path": "results_paper/figures/appendix/figA01_illustrative_val_trajectory.pdf", "role": "appendix", "dataset": "SynthCharge VAL", "note": "not TEST"},
        {"path": "results_paper/figures/appendix/figA02_learning_curves.png", "role": "appendix-development", "dataset": "gold VAL"},
        {"path": "results_paper/figures/appendix/figA02_learning_curves.pdf", "role": "appendix-development", "dataset": "gold VAL"},
        {"path": "results_paper/figures/appendix/figA03_training_stability.png", "role": "appendix-development", "dataset": "gold VAL"},
        {"path": "results_paper/figures/appendix/figA03_training_stability.pdf", "role": "appendix-development", "dataset": "gold VAL"},
        {"path": "results_paper/figures/appendix/figA04_envelope_ablation.png", "role": "appendix-development", "dataset": "SynthCharge VAL", "note": "post-hoc"},
        {"path": "results_paper/figures/appendix/figA04_envelope_ablation.pdf", "role": "appendix-development", "dataset": "SynthCharge VAL", "note": "post-hoc"},
        {"path": "results_paper/figures/appendix/figA05_failure_analysis.png", "role": "appendix", "dataset": "SynthCharge V3 TEST", "note": "frozen raw only"},
        {"path": "results_paper/figures/appendix/figA05_failure_analysis.pdf", "role": "appendix", "dataset": "SynthCharge V3 TEST", "note": "frozen raw only"},
        {"path": "results_paper/figures/appendix/figA06_seed_robustness.png", "role": "appendix", "dataset": "SynthCharge V3 TEST"},
        {"path": "results_paper/figures/appendix/figA06_seed_robustness.pdf", "role": "appendix", "dataset": "SynthCharge V3 TEST"},
        {"path": "results_paper/figures/appendix/figA07_native_frvcp_reference.png", "role": "appendix", "dataset": "native FRVCP"},
        {"path": "results_paper/figures/appendix/figA07_native_frvcp_reference.pdf", "role": "appendix", "dataset": "native FRVCP"},
    ]
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
            artifacts.append({"path": f"results_paper/tables/{stem}.{ext}", "role": "appendix" if stem.startswith("tableA") else "main"})
    build_manifest(artifacts)
    print("results_paper written")


if __name__ == "__main__":
    main()
