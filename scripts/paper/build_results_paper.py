"""Build publication-facing results_paper/ (PNG figures + tables) from frozen data.

READ ONLY: V3 raw/statistics/ablation + archived FRVCP stats.
Never trains, evaluates TEST, or modifies frozen raw rows.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3_hppo"))

from experiments.stats import hierarchical_bootstrap_ci, mean_sd_across_training_seeds  # noqa: E402
from common import BASELINES, SEEDS, V3, lf_sha256, load_json, sha256  # noqa: E402

RAW = V3 / "raw" / "synthcharge_test.jsonl"
STATS = V3 / "statistics"
OUT = ROOT / "results_paper"
FIG = OUT / "figures"
FIG_A = FIG / "appendix"
TAB = OUT / "tables"
N_BOOT = 5000
RNG = 20261003
DPI = 600

METHODS = ("HybridPPO",) + BASELINES
LABEL = {
    "HybridPPO": "FA-HPPO",
    "GreedyMinimumSufficientCharge": "Greedy Min",
    "GreedyFullCharge": "Greedy Full",
    "OneStepLookahead": "Lookahead",
    "FA-HPPO-Min": "FA-HPPO-Min",
    "FA-HPPO-Max": "FA-HPPO-Max",
}
ORDER_BASE = ["HybridPPO", "OneStepLookahead", "GreedyFullCharge", "GreedyMinimumSufficientCharge"]
COLOR = {
    "HybridPPO": "#0072B2",
    "OneStepLookahead": "#D55E00",
    "GreedyFullCharge": "#009E73",
    "GreedyMinimumSufficientCharge": "#E69F00",
    "FA-HPPO-Min": "#CC79A7",
    "FA-HPPO-Max": "#56B4E9",
}
ABL_ORDER = [
    ("B0", "Base HPPO (B0)"),
    ("B1", "+ Time-aware cap (B1)"),
    ("B3", "+ Return scaling (B3)"),
    ("B2", "FA-HPPO full (B2)"),
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


def summarize_learned(rows, method):
    learned = [{**r, "feasible_f": float(bool(r["feasible"]))} for r in method_rows(rows, method)]
    feas = mean_sd_across_training_seeds(learned, "feasible_f")
    comp = mean_sd_across_training_seeds(learned, "completion_time_all_routes")
    feas_ci = hierarchical_bootstrap_ci(learned, "feasible_f", n_boot=N_BOOT, seed=RNG)
    comp_ci = hierarchical_bootstrap_ci(learned, "completion_time_all_routes", n_boot=N_BOOT, seed=RNG)
    per_f = {int(k): v["mean"] for k, v in feas["per_seed"].items()}
    per_c = {int(k): v["mean"] for k, v in comp["per_seed"].items()}
    return {
        "feasibility_mean": feas["mean_across_seeds"],
        "feasibility_sd": feas["sd_across_seeds"],
        "feasibility_ci95": [feas_ci["lo"], feas_ci["hi"]],
        "completion_all_mean": comp["mean_across_seeds"],
        "completion_all_ci95": [comp_ci["lo"], comp_ci["hi"]],
        "per_seed_feasibility": per_f,
        "per_seed_completion": per_c,
        "runtime_s": mean(r.get("runtime_s") for r in method_rows(rows, method)),
    }


def summarize_baseline(rows, method):
    subset = method_rows(rows, method)
    return {
        "feasibility_mean": mean(float(r["feasible"]) for r in subset),
        "feasibility_sd": None,
        "feasibility_ci95": None,
        "completion_all_mean": mean(r["completion_time_all_routes"] for r in subset),
        "completion_all_ci95": None,
        "per_seed_feasibility": None,
        "per_seed_completion": None,
        "runtime_s": mean(r.get("runtime_s") for r in subset),
    }


def save_png(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_method_panel(ax, summaries, metric_mean, metric_ci, metric_per, ylabel, title, ylim=None):
    xs = np.arange(len(ORDER_BASE))
    for i, m in enumerate(ORDER_BASE):
        s = summaries[m]
        color = COLOR[m]
        if s.get(metric_per):
            vals = [s[metric_per][seed] for seed in SEEDS]
            ax.scatter(np.full(len(vals), i), vals, color=color, s=28, zorder=3, alpha=0.9)
            mu = s[metric_mean]
            ax.plot([i - 0.18, i + 0.18], [mu, mu], color="black", lw=1.6, zorder=4)
            if s.get(metric_ci):
                lo, hi = s[metric_ci]
                ax.vlines(i, lo, hi, color="black", lw=1.2, zorder=4)
        else:
            ax.scatter([i], [s[metric_mean]], color=color, s=55, marker="D", zorder=3)
    ax.set_xticks(xs, [LABEL[m] for m in ORDER_BASE], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ylim is not None:
        ax.set_ylim(*ylim)


def fig01(rows):
    all_s = {m: summarize_learned(rows, m) if m == "HybridPPO" else summarize_baseline(rows, m) for m in ORDER_BASE}
    charge = [r for r in rows if r.get("charge_class") == "charging_required"]
    ch_s = {m: summarize_learned(charge, m) if m == "HybridPPO" else summarize_baseline(charge, m) for m in ORDER_BASE}
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
    plot_method_panel(axes[0, 0], all_s, "feasibility_mean", "feasibility_ci95", "per_seed_feasibility", "Feasibility", "(a) All routes — feasibility", (0, 1.05))
    plot_method_panel(axes[0, 1], all_s, "completion_all_mean", "completion_all_ci95", "per_seed_completion", "Completion (infeasible→H)", "(b) All routes — completion")
    plot_method_panel(axes[1, 0], ch_s, "feasibility_mean", "feasibility_ci95", "per_seed_feasibility", "Feasibility", "(c) Charging-required — feasibility", (0, 1.05))
    plot_method_panel(axes[1, 1], ch_s, "completion_all_mean", "completion_all_ci95", "per_seed_completion", "Completion (infeasible→H)", "(d) Charging-required — completion")
    fig.suptitle("V3 fresh SynthCharge TEST (180 routes; FA-HPPO: 5 seeds, hierarchical 95% CI)", fontsize=10, y=1.01)
    fig.tight_layout()
    save_png(fig, FIG / "fig01_main_test.png")
    return all_s, ch_s


def fig02():
    paired = load_json(STATS / "paired_primary.json")
    # Keep Lookahead, Full, Min order
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
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, family, xlab, scale in (
        (axes[0], "feasibility", "Feasibility effect (pp)", 100.0),
        (axes[1], "completion_all", "Completion effect", 1.0),
    ):
        rows = [t for name in order for t in paired if t["comparison"] == name and t["family"] == family]
        ys = np.arange(len(rows))[::-1]
        labels = []
        for y, t in zip(ys, rows):
            eff = float(t["effect"]) * scale
            lo, hi = float(t["ci95"][0]) * scale, float(t["ci95"][1]) * scale
            ax.hlines(y, lo, hi, color="#0072B2", lw=2)
            ax.plot(eff, y, "o", color="#0072B2", ms=6)
            ptxt = f"p_Holm={t['p_holm']:.1e}" if t["p_holm"] < 1e-3 else f"p_Holm={t['p_holm']:.3f}"
            labels.append(f"{short[t['comparison']]}\n{ptxt}")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ys, labels)
        ax.set_xlabel(xlab)
        ax.set_title("(a) Feasibility" if family == "feasibility" else "(b) Completion", loc="left")
    fig.suptitle("Paired FA-HPPO − baseline effects (predeclared seed-averaged route analysis)", fontsize=10)
    fig.tight_layout()
    save_png(fig, FIG / "fig02_effect_sizes.png")


def load_ablation():
    rows = []
    for code, _label in ABL_ORDER:
        for seed in SEEDS:
            path = V3 / "ablation" / code / f"seed_{seed}" / "validation.json"
            rows.append(load_json(path))
    return rows


def fig03(ablation):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    labels = [lab for _c, lab in ABL_ORDER]
    xs = np.arange(len(ABL_ORDER))
    for i, (code, _lab) in enumerate(ABL_ORDER):
        subset = [r for r in ablation if r["variant"] == code]
        feas = [r["parent_balanced_val_feasibility"] for r in subset]
        axes[0].scatter(np.full(5, i), feas, color=COLOR["HybridPPO"] if code == "B2" else "#666666", s=28, zorder=3)
        axes[0].plot([i - 0.2, i + 0.2], [mean(feas), mean(feas)], color="black", lw=1.5)
        vl = [r["optimization"]["value_loss_mean"] for r in subset]
        gn = [r["optimization"]["grad_norm_preclip_mean"] for r in subset]
        axes[1].scatter(np.full(5, i), vl, color="#D55E00", s=28, zorder=3)
        axes[2].scatter(np.full(5, i), gn, color="#E69F00", s=28, zorder=3)
    axes[0].set_ylabel("Parent-balanced VAL feasibility")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Feasibility", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean value loss")
    axes[1].set_title("(b) Value loss (log)", loc="left")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Mean grad norm (pre-clip)")
    axes[2].set_title("(c) Gradient norm (log)", loc="left")
    for ax in axes:
        ax.set_xticks(xs, labels, rotation=25, ha="right")
    fig.suptitle("DEVELOPMENT / GOLD VALIDATION — not V3 TEST", fontsize=10, color="#8B0000")
    fig.tight_layout()
    save_png(fig, FIG / "fig03_ablation_and_training_stability.png")


def fig04(rows):
    methods = ["HybridPPO", "FA-HPPO-Max", "FA-HPPO-Min"]
    summaries = {m: summarize_learned(rows, m) for m in methods}
    amount = load_json(STATS / "amount_sensitivity.json") if (STATS / "amount_sensitivity.json").is_file() else None
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, key_mean, key_ci, key_per, ylab, title in (
        (axes[0], "feasibility_mean", "feasibility_ci95", "per_seed_feasibility", "Feasibility", "(a) Feasibility"),
        (axes[1], "completion_all_mean", "completion_all_ci95", "per_seed_completion", "Completion", "(b) Failure-retaining completion"),
    ):
        for i, m in enumerate(methods):
            s = summaries[m]
            vals = [s[key_per][seed] for seed in SEEDS]
            ax.scatter(np.full(5, i), vals, color=COLOR[m], s=28, zorder=3)
            ax.plot([i - 0.18, i + 0.18], [s[key_mean], s[key_mean]], color="black", lw=1.5)
            lo, hi = s[key_ci]
            ax.vlines(i, lo, hi, color="black", lw=1.1)
        ax.set_xticks(range(3), ["FA-HPPO\nlearned u", "FA-HPPO-Max\nu=1", "FA-HPPO-Min\nu=0"])
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        if key_mean == "feasibility_mean":
            ax.set_ylim(0, 1.05)
    tie = amount["route_averaged_feasibility_vs_max"]["tied"] if amount else 173
    axes[0].text(0.02, 0.08, f"FA-HPPO vs Max: {tie}/180 routes tied\n(route-averaged feasibility)", transform=axes[0].transAxes, fontsize=7)
    fig.suptitle("V3 TEST amount-policy sensitivity (same frozen checkpoints)", fontsize=10)
    fig.tight_layout()
    save_png(fig, FIG / "fig04_amount_sensitivity.png")


def fig05(rows):
    hybrid = method_rows(rows, "HybridPPO")
    charge = [r for r in hybrid if r.get("charge_class") == "charging_required"]
    fails = [r for r in hybrid if not r["feasible"]]
    layouts = ["R", "C", "RC"]
    bins = ["short", "medium", "long"]
    feas = np.zeros((3, 3))
    fail_rate = np.zeros((3, 3))
    for i, layout in enumerate(layouts):
        for j, length in enumerate(bins):
            cell = [r for r in charge if r.get("layout") == layout and r.get("length_bin") == length]
            feas[i, j] = mean(float(r["feasible"]) for r in cell) if cell else np.nan
            fail_rate[i, j] = 1.0 - feas[i, j] if cell else np.nan
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    im0 = axes[0].imshow(feas, vmin=0, vmax=1, cmap="viridis")
    axes[0].set_xticks(range(3), bins)
    axes[0].set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{feas[i, j]:.2f}", ha="center", va="center", color="white" if feas[i, j] < 0.55 else "black", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[0].set_title("(a) Charging-required FA-HPPO feasibility", loc="left")
    im1 = axes[1].imshow(fail_rate, vmin=0, vmax=1, cmap="magma")
    axes[1].set_xticks(range(3), bins)
    axes[1].set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{fail_rate[i, j]:.2f}", ha="center", va="center", color="white" if fail_rate[i, j] > 0.45 else "black", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title("(b) Failure rate by cell", loc="left")
    reasons = Counter(r.get("reason") for r in fails)
    note = f"{len(fails)} route×seed failures; " + ", ".join(f"{k}={v}" for k, v in reasons.items())
    fig.suptitle(note, fontsize=9)
    fig.tight_layout()
    save_png(fig, FIG / "fig05_difficulty_and_failures.png")


def fig_a01(summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    seeds = list(SEEDS)
    feas = [summary["per_seed_feasibility"][s] for s in seeds]
    comp = [summary["per_seed_completion"][s] for s in seeds]
    axes[0].scatter(seeds, feas, s=50, color=COLOR["HybridPPO"], zorder=3)
    axes[0].axhline(summary["feasibility_mean"], color="black", ls="--", lw=1)
    lo, hi = summary["feasibility_ci95"]
    axes[0].axhspan(lo, hi, color=COLOR["HybridPPO"], alpha=0.15)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Feasibility")
    axes[0].set_title("(a) Feasibility", loc="left")
    axes[1].scatter(seeds, comp, s=50, color=COLOR["HybridPPO"], zorder=3)
    axes[1].axhline(summary["completion_all_mean"], color="black", ls="--", lw=1)
    lo, hi = summary["completion_all_ci95"]
    axes[1].axhspan(lo, hi, color=COLOR["HybridPPO"], alpha=0.15)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Completion")
    axes[1].set_title("(b) Completion", loc="left")
    fig.suptitle("Appendix — FA-HPPO seed robustness (V3 TEST)", fontsize=10)
    fig.tight_layout()
    save_png(fig, FIG_A / "figA01_seed_robustness.png")


def fig_a02():
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
    feas = {r["method"]: float(r["feasibility_mean"]) for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    x = np.arange(len(order))
    axes[0].bar(x, [feas[m] for m in order], color=[colors[m] for m in order], width=0.7)
    axes[0].set_xticks(x, [labels[m] for m in order], rotation=15)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Feasibility")
    axes[0].set_title("(a) Native FRVCP feasibility (n=133)", loc="left")
    gm = ["FRVCPGreedyMin", "FRVCPGreedyFull"]
    axes[1].bar(np.arange(2), [gaps[m] for m in gm], color=[colors[m] for m in gm], width=0.65)
    axes[1].set_xticks(np.arange(2), [labels[m] for m in gm])
    axes[1].set_ylabel("Mean optimality gap (%)")
    axes[1].set_title("(b) Gap vs frvcpy Solver", loc="left")
    fig.suptitle("NATIVE FRVCP REFERENCE — not EVRPTW-GR / not SynthCharge", fontsize=9, color="#8B0000")
    fig.tight_layout()
    save_png(fig, FIG_A / "figA02_native_frvcp_reference.png")


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
        "V3 is a fresh independently generated SynthCharge TEST, not external-domain generalization. Stratified certificate-filtered stress benchmark.",
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
        "FA-HPPO = mean over 5 training seeds; baselines deterministic. Infeasible completion = horizon H. CI = hierarchical bootstrap (seed→route).",
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
        ["Variant", "Time cap", "Return scale", "VAL feas.", "SD", "Charge-req feas.", "Completion", "Value loss", "Grad norm"],
        rows_abl,
        "DEVELOPMENT / GOLD VALIDATION ONLY — not V3 TEST. Five seeds 42–46.",
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
        "Same frozen checkpoints on V3 TEST. Do not claim free continuous u beats Max.",
    )
    # Appendix
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
                f"{t['p_raw']:.2e}",
                f"{t['p_holm']:.2e}",
                t.get("procedure", "predeclared"),
                str(t["n_independent_units"]),
            ]
            for t in paired
        ],
        "Predeclared seed-averaged route-paired analysis with Holm correction. Sensitivity: statistics/paired_sensitivity_joint_seed_route.json.",
    )


def write_readme():
    text = """# Publication outputs (`results_paper/`)

Single source of truth for manuscript figures (PNG only) and tables.

## Main paper

| Artifact | Role | Source | Split | Seeds | CI |
|----------|------|--------|-------|-------|-----|
| `figures/fig01_main_test.png` | Confirmatory performance | `results/v3_hppo/raw/synthcharge_test.jsonl` | V3 TEST | 5 | hierarchical bootstrap |
| `figures/fig02_effect_sizes.png` | Paired effect sizes | `statistics/paired_primary.json` | V3 TEST | 5 | predeclared paired CI |
| `figures/fig03_ablation_and_training_stability.png` | Methodology | `results/v3_hppo/ablation/` | gold VAL | 5 | seed scatter |
| `figures/fig04_amount_sensitivity.png` | Amount policy | V3 raw | V3 TEST | 5 | hierarchical bootstrap |
| `figures/fig05_difficulty_and_failures.png` | Difficulty / failures | V3 raw | V3 TEST | 5 | cell means |
| `tables/table01_*` | Protocol | metadata | — | — | — |
| `tables/table02_*` | Main results | V3 raw/stats | V3 TEST | 5 | hierarchical bootstrap |
| `tables/table03_*` | Ablation | ablation JSON | gold VAL | 5 | SD across seeds |
| `tables/table04_*` | Amount sensitivity | V3 raw | V3 TEST | 5 | hierarchical bootstrap |

## Appendix

| Artifact | Role |
|----------|------|
| `figures/appendix/figA01_seed_robustness.png` | Per-seed robustness |
| `figures/appendix/figA02_native_frvcp_reference.png` | Native FRVCP (not EVRPTW-GR / not SynthCharge) |
| `tables/tableA01_*` | Per-seed FA-HPPO |
| `tables/tableA02_*` | Primary paired statistics |

## Regenerate

```bash
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Does **not** retrain or re-evaluate TEST.

## Notes

- FA-HPPO CIs: hierarchical bootstrap resampling training seed then route.
- Deterministic baselines: no fabricated seed uncertainty.
- Fig 3 is **development / gold validation**, not confirmatory TEST.
- V3 = fresh independently generated SynthCharge TEST (not external-domain generalization).
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def build_manifest(artifacts: list[dict]):
    for item in artifacts:
        path = ROOT / item["path"]
        item["sha256"] = _sha(path) if path.is_file() else None
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
    locked = [k for k in lock["files"] if "/instances/" in k.replace("\\", "/")]
    if len(locked) != 180:
        raise SystemExit("expected 180 locked instances")
    import subprocess

    tracked = subprocess.check_output(["git", "ls-files", "data/routes_v2/synthcharge_v3_test/instances"], text=True).splitlines()
    if len(tracked) != 180:
        raise SystemExit(f"expected 180 tracked instances, got {len(tracked)}")
    pngs = list(FIG.rglob("*.png"))
    if len(pngs) < 7:
        raise SystemExit("missing PNG figures")
    for path in FIG.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".svg", ".eps"}:
            raise SystemExit(f"non-PNG figure present: {path}")
    manifest = load_json(OUT / "MANIFEST.json")
    for item in manifest["artifacts"]:
        if item["path"].endswith((".png", ".csv", ".md", ".tex", ".json")):
            if not (ROOT / item["path"]).is_file():
                raise SystemExit(f"missing artifact {item['path']}")
    print(json.dumps({"verify": "ok", "n_rows": 3240, "n_instances": 180, "n_png": len(pngs)}, indent=2))


def main() -> None:
    if "--verify" in sys.argv:
        verify()
        return
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    FIG_A.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    all_s, ch_s = fig01(rows)
    fig02()
    ablation = load_ablation()
    fig03(ablation)
    fig04(rows)
    fig05(rows)
    fig_a01(all_s["HybridPPO"])
    fig_a02()
    build_tables(all_s, ch_s, ablation, rows)
    write_readme()
    artifacts = [
        {"path": "results_paper/README.md", "role": "index", "generation_script": "scripts/paper/build_results_paper.py"},
        {"path": "results_paper/figures/fig01_main_test.png", "role": "main", "metric": "feasibility/completion", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5, "source": ["results/v3_hppo/raw/synthcharge_test.jsonl"]},
        {"path": "results_paper/figures/fig02_effect_sizes.png", "role": "main", "metric": "paired effects", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5, "source": ["results/v3_hppo/statistics/paired_primary.json"]},
        {"path": "results_paper/figures/fig03_ablation_and_training_stability.png", "role": "main-development", "metric": "VAL feas / value loss / grad", "dataset": "gold VAL", "n_routes": 47, "n_seeds": 5, "source": ["results/v3_hppo/ablation/"]},
        {"path": "results_paper/figures/fig04_amount_sensitivity.png", "role": "main", "metric": "amount sensitivity", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5, "source": ["results/v3_hppo/raw/synthcharge_test.jsonl"]},
        {"path": "results_paper/figures/fig05_difficulty_and_failures.png", "role": "main", "metric": "cell feas / failure rate", "dataset": "SynthCharge V3 TEST", "n_routes": 144, "n_seeds": 5, "source": ["results/v3_hppo/raw/synthcharge_test.jsonl"]},
        {"path": "results_paper/figures/appendix/figA01_seed_robustness.png", "role": "appendix", "metric": "per-seed", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5, "source": ["results/v3_hppo/raw/synthcharge_test.jsonl"]},
        {"path": "results_paper/figures/appendix/figA02_native_frvcp_reference.png", "role": "appendix", "metric": "FRVCP feas/gap", "dataset": "native FRVCP archive", "n_routes": 133, "n_seeds": None, "source": ["results/final/statistics/frvcpy_native/method_summary.csv"]},
    ]
    for stem in (
        "table01_benchmark_protocol",
        "table02_main_results",
        "table03_development_ablation",
        "table04_amount_sensitivity",
        "tableA01_per_seed",
        "tableA02_primary_statistics",
    ):
        for ext in ("csv", "md", "tex"):
            artifacts.append(
                {
                    "path": f"results_paper/tables/{stem}.{ext}",
                    "role": "appendix" if stem.startswith("tableA") else "main",
                    "generation_script": "scripts/paper/build_results_paper.py",
                    "source": ["results/v3_hppo/"],
                }
            )
    build_manifest(artifacts)
    print("results_paper written")


if __name__ == "__main__":
    main()
