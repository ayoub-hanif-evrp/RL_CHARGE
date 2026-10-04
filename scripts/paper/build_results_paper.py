"""Build publication-facing results_paper/ (PNG figures + tables) from frozen data.

READ ONLY for TEST evidence. Case-study JSON is a VAL illustration (not TEST).
Never trains final models, evaluates TEST, or modifies frozen raw rows.
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
CASE = OUT / "case_study" / "illustrative_val_episode.json"
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
ABL_ORDER = [
    ("B0", "Base HPPO\n(B0)"),
    ("B1", "+ Time-aware cap\n(B1)"),
    ("B3", "+ Return scaling\n(B3)"),
    ("B2", "FA-HPPO full\n(B2)"),
]
MC_P_FLOOR = 5e-5  # (1)/(20000+1) Monte Carlo resolution for n_perm=20000


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
    """Paper-facing p-value; avoid false precision at the Monte Carlo floor."""
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
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_method_panel(ax, summaries, metric_mean, metric_ci, metric_per, ylabel, title, ylim=None):
    xs = np.arange(len(ORDER_BASE))
    rng = np.random.default_rng(1)
    for i, m in enumerate(ORDER_BASE):
        s = summaries[m]
        color = COLOR[m]
        if s.get(metric_per):
            vals = [s[metric_per][seed] for seed in SEEDS]
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(i + jitter, vals, color=color, s=32, zorder=3, edgecolors="white", linewidths=0.4)
            mu = s[metric_mean]
            ax.plot([i - 0.22, i + 0.22], [mu, mu], color="black", lw=2.0, zorder=4)
            if s.get(metric_ci):
                lo, hi = s[metric_ci]
                ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], fmt="none", ecolor="black", elinewidth=1.6, capsize=4.5, zorder=4)
        else:
            ax.scatter([i], [s[metric_mean]], color=color, s=60, marker="D", zorder=3, edgecolors="black", linewidths=0.4)
    ax.set_xticks(xs, [LABEL[m] for m in ORDER_BASE], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ylim is not None:
        ax.set_ylim(*ylim)


def _beta_pdf(u, alpha, beta):
    from math import lgamma

    u = np.clip(np.asarray(u, dtype=float), 1e-6, 1 - 1e-6)
    log_b = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    return np.exp((alpha - 1) * np.log(u) + (beta - 1) * np.log(1 - u) - log_b)


def fig01_case_study():
    if not CASE.is_file():
        raise SystemExit("missing case study JSON; run scripts/paper/record_illustrative_val_episode.py")
    ep = load_json(CASE)
    nodes = ep["nodes"]
    customers = ep["customer_ids"]
    depot = ep["depot_id"]
    stations = ep["station_ids"]
    path = ep["path_nodes"]
    trace = ep.get("soc_trace") or []
    decisions = ep["decisions"]
    visited_stations = [n for n in path if n in stations]

    fig = plt.figure(figsize=(12.2, 4.15))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.28, 1.05], wspace=0.30)
    ax_map = fig.add_subplot(outer[0, 0])
    ax_soc = fig.add_subplot(outer[0, 1])
    right = outer[0, 2].subgridspec(2, 1, hspace=0.55, height_ratios=[1.0, 1.15])
    ax_disc = fig.add_subplot(right[0, 0])
    ax_beta = fig.add_subplot(right[1, 0])

    # --- (a) route map ---
    seq = [depot] + customers + [depot]
    ax_map.plot(
        [nodes[i]["x"] for i in seq],
        [nodes[i]["y"] for i in seq],
        ls="--",
        color="#B0B0B0",
        lw=1.1,
        zorder=1,
        label="Fixed customer order",
    )
    for a, b in zip(path[:-1], path[1:]):
        ax_map.annotate(
            "",
            xy=(nodes[b]["x"], nodes[b]["y"]),
            xytext=(nodes[a]["x"], nodes[a]["y"]),
            arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=1.35, mutation_scale=10),
            zorder=2,
        )
    ax_map.plot(
        [nodes[i]["x"] for i in path],
        [nodes[i]["y"] for i in path],
        color="#D55E00",
        lw=1.5,
        zorder=2,
        label="FA-HPPO path (+ charges)",
    )
    for sid in stations:
        n = nodes[sid]
        used = sid in visited_stations
        if used:
            ax_map.scatter(n["x"], n["y"], marker="*", s=160, color="#0072B2", zorder=6, edgecolors="white", linewidths=0.4, label="Visited station" if sid == visited_stations[0] else None)
            ax_map.text(n["x"], n["y"] + 0.028, sid, ha="center", va="bottom", fontsize=7, color="#0072B2", fontweight="bold")
        else:
            ax_map.scatter(n["x"], n["y"], marker="^", s=55, color="#C8C8C8", zorder=3, label="Unused station" if sid == stations[0] else None)
            ax_map.text(n["x"], n["y"] + 0.022, sid, ha="center", va="bottom", fontsize=6, color="#9A9A9A")
    for i, cid in enumerate(customers, start=1):
        n = nodes[cid]
        ax_map.scatter(n["x"], n["y"], s=62, facecolors="white", edgecolors="#222222", linewidths=1.05, zorder=5, label="Customer (order)" if i == 1 else None)
        ax_map.text(n["x"], n["y"], str(i), ha="center", va="center", fontsize=7, zorder=6)
    nd = nodes[depot]
    ax_map.scatter(nd["x"], nd["y"], marker="s", s=78, color="#000000", zorder=7, label="Depot")
    ax_map.text(nd["x"], nd["y"] - 0.035, "Depot", ha="center", va="top", fontsize=7)
    # Zoom to visited path with padding; unused stations remain if nearby
    vx = [nodes[i]["x"] for i in path]
    vy = [nodes[i]["y"] for i in path]
    pad_x = max(0.04, 0.08 * (max(vx) - min(vx) + 1e-9))
    pad_y = max(0.04, 0.08 * (max(vy) - min(vy) + 1e-9))
    ax_map.set_xlim(min(vx) - pad_x, max(vx) + pad_x)
    ax_map.set_ylim(min(vy) - pad_y, max(vy) + pad_y)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("x (normalized)")
    ax_map.set_ylabel("y (normalized)")
    ax_map.set_title("(a) Fixed route + charging detours", loc="left")
    ax_map.text(0.02, 0.98, "Customer order fixed;\nFA-HPPO inserts charges", transform=ax_map.transAxes, fontsize=6.5, va="top")
    handles, labels = ax_map.get_legend_handles_labels()
    # de-duplicate labels
    seen = set()
    uniq = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            uniq.append((h, lab))
            seen.add(lab)
    ax_map.legend([h for h, _ in uniq], [lab for _, lab in uniq], loc="upper left", bbox_to_anchor=(0.0, -0.18), frameon=False, fontsize=6.2, ncol=2)

    # --- (b) SOC along visit sequence ---
    # Travel = sloping segments; charging = vertical jump at the station index.
    points = [(0.0, 100.0 * float(trace[0]["soc"]) if trace else 100.0, "start", None)]
    cursor = 0
    for d in decisions:
        if d.get("action") == "CONTINUE":
            cursor += 1
            soc_arr = 100 * float(d["soc_after"])
            points.append((float(cursor), soc_arr, "arrive", d))
        elif d.get("action") == "CHARGE":
            cursor += 1
            soc_arr = 100 * float(d.get("soc_arrival", d["soc_before_decision"]))
            points.append((float(cursor), soc_arr, "station_arrive", d))
            soc_dep = 100 * float(d.get("soc_departure", d["soc_after"]))
            points.append((float(cursor), soc_dep, "charge_depart", d))
    for (x0, y0, _k0, _), (x1, y1, k1, _meta) in zip(points[:-1], points[1:]):
        if k1 == "charge_depart":
            ax_soc.plot([x0, x1], [y0, y1], color="#D55E00", lw=2.4, zorder=3)
        else:
            ax_soc.plot([x0, x1], [y0, y1], color="#0072B2", lw=1.8, zorder=3)
    for x, y, kind, meta in points:
        if kind == "station_arrive":
            ax_soc.scatter([x], [y], s=30, color="#0072B2", zorder=5, edgecolors="white", linewidths=0.4)
            if meta:
                lo, hi = 100 * meta["soc_lower"], 100 * meta["soc_upper"]
                ax_soc.fill_between([x - 0.18, x + 0.18], lo, hi, color="#56B4E9", alpha=0.35, zorder=1)
                ax_soc.plot([x, x], [lo, hi], color="#56B4E9", lw=1.0, alpha=0.85, zorder=2)
        elif kind == "charge_depart":
            ax_soc.scatter([x], [y], marker="*", s=95, color="#D55E00", zorder=6)
        elif kind == "arrive":
            ax_soc.scatter([x], [y], s=18, color="#333333", zorder=4)
    min_soc = 100 * float(ep.get("min_soc_fraction", 0.0))
    ax_soc.axhline(min_soc, color="#888888", ls="--", lw=0.8)
    cust_rank = {cid: str(i) for i, cid in enumerate(customers, start=1)}
    xticklabels = []
    for nid in path:
        if nid == depot:
            xticklabels.append("D")
        elif nid in cust_rank:
            xticklabels.append(cust_rank[nid])
        else:
            xticklabels.append(nid)
    ax_soc.set_xticks(range(len(path)), xticklabels, fontsize=7)
    ax_soc.set_ylim(-2, 108)
    ax_soc.set_xlim(-0.3, len(path) - 0.7)
    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_xlabel("Visit order (customer # / station / depot)")
    ax_soc.set_title("(b) SOC evolution along route", loc="left")
    ax_soc.text(
        0.98,
        0.06,
        r"$SOC_{tgt}=SOC_{lo}+u\,(SOC_{hi}-SOC_{lo})$",
        transform=ax_soc.transAxes,
        ha="right",
        fontsize=6.5,
    )
    for x, y, kind, meta in points:
        if kind == "charge_depart" and meta and meta.get("step") == ep.get("representative_charge_step"):
            ax_soc.annotate(
                f"target {y:.0f}%\n[lo,hi]=[{100*meta['soc_lower']:.0f},{100*meta['soc_upper']:.0f}]",
                xy=(x, y),
                xytext=(min(x + 0.7, len(path) - 1.2), min(102, y + 6)),
                fontsize=6,
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7),
            )
            break

    # --- (c) representative hybrid decision ---
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
        colors = []
        for i, name in enumerate(names):
            if not mask[i]:
                colors.append("#DDDDDD")
            elif name == rep.get("station_id") or (name.startswith("S") and name == rep.get("station_id")):
                colors.append("#0072B2")
            elif i == rep["discrete"]:
                colors.append("#0072B2")
            else:
                colors.append("#9E9E9E")
        ax_disc.bar(xs, np.where(mask, probs, 0.0), color=colors, width=0.72, edgecolor="#444444", linewidth=0.4)
        for i, (_name, mflag) in enumerate(zip(names, mask)):
            if not mflag:
                ax_disc.bar(i, max(float(probs.max()) * 0.08, 0.04), color="none", edgecolor="#888888", hatch="////", width=0.72)
                ax_disc.text(i, max(float(probs.max()) * 0.08, 0.04) + 0.02, "masked", ha="center", va="bottom", fontsize=5.5, color="#666666", rotation=90)
        ax_disc.set_xticks(xs, names, fontsize=7)
        ax_disc.set_ylim(0, 1.08)
        ax_disc.set_ylabel("Probability")
        ax_disc.set_title(f"(c) Hybrid decision @ {rep.get('station_id')}", loc="left", fontsize=9)
        ax_disc.text(0.98, 0.92, "softmax(masked logits)", transform=ax_disc.transAxes, ha="right", fontsize=6, color="#555555")

        alpha = float(rep["alpha"])
        beta = float(rep["beta"])
        u = float(rep["u"])
        ugrid = np.linspace(0.001, 0.999, 300)
        dens = _beta_pdf(ugrid, alpha, beta)
        ax_beta.fill_between(ugrid, dens, color="#56B4E9", alpha=0.35)
        ax_beta.plot(ugrid, dens, color="#0072B2", lw=1.5)
        ax_beta.axvline(u, color="#D55E00", lw=1.6)
        ax_beta.axvline(alpha / (alpha + beta), color="#333333", ls=":", lw=1.0)
        ax_beta.set_xlabel(r"$u\in[0,1]$")
        ax_beta.set_ylabel(r"Beta dens.")
        lo, hi, tgt = rep["soc_lower"], rep["soc_upper"], rep["soc_target"]
        ax_beta.set_title(
            rf"Beta$(\alpha={alpha:.1f},\beta={beta:.1f})$  $u={u:.2f}$",
            loc="left",
            fontsize=8,
        )
        ax_beta.text(
            0.02,
            0.95,
            rf"$SOC_{{lo}}={100*lo:.0f}\%\rightarrow SOC_{{tgt}}={100*tgt:.0f}\%\rightarrow SOC_{{hi}}={100*hi:.0f}\%$",
            transform=ax_beta.transAxes,
            va="top",
            fontsize=6.3,
        )
        feats = rep.get("station_features", {}).get(rep.get("station_id"), {})
        if feats:
            ax_beta.text(
                0.98,
                0.55,
                "\n".join(
                    [
                        f"dist→CS={feats.get('dist_to_station', float('nan')):.2f}",
                        f"E→CS={feats.get('energy_to_station', float('nan')):.2f}",
                        f"detour t={feats.get('detour_time', float('nan')):.2f}",
                        f"slack={feats.get('slack_to_next_after_travel', float('nan')):.2f}",
                    ]
                ),
                transform=ax_beta.transAxes,
                ha="right",
                va="top",
                fontsize=5.8,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
            )

    fig.suptitle(
        "Illustrative SynthCharge VAL episode — not TEST evidence\n"
        f"{ep['route_id']}  (geometry-selected RC/medium; FA-HPPO seed 42)",
        fontsize=9,
        color="#8B0000",
        y=1.05,
    )
    save_png(fig, FIG / "fig01_method_case_study.png")


def fig02_main(rows):
    all_s = {m: summarize_learned(rows, m) if m == "HybridPPO" else summarize_baseline(rows, m) for m in ORDER_BASE}
    charge = [r for r in rows if r.get("charge_class") == "charging_required"]
    ch_s = {m: summarize_learned(charge, m) if m == "HybridPPO" else summarize_baseline(charge, m) for m in ORDER_BASE}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))
    plot_method_panel(axes[0, 0], all_s, "feasibility_mean", "feasibility_ci95", "per_seed_feasibility", "Feasibility", "(a) All routes — feasibility", (0, 1.05))
    plot_method_panel(axes[0, 1], all_s, "completion_all_mean", "completion_all_ci95", "per_seed_completion", "Completion (infeasible→H)", "(b) All routes — completion")
    plot_method_panel(axes[1, 0], ch_s, "feasibility_mean", "feasibility_ci95", "per_seed_feasibility", "Feasibility", "(c) Charging-required — feasibility", (0, 1.05))
    plot_method_panel(axes[1, 1], ch_s, "completion_all_mean", "completion_all_ci95", "per_seed_completion", "Completion (infeasible→H)", "(d) Charging-required — completion")
    fig.suptitle("V3 SynthCharge TEST (180 routes; FA-HPPO 5 seeds, hierarchical 95% CI)", fontsize=9.5)
    fig.tight_layout()
    save_png(fig, FIG / "fig02_main_test.png")
    return all_s, ch_s


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
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for ax, family, scale in (
        (axes[0], "feasibility", 100.0),
        (axes[1], "completion_all", 1.0),
    ):
        rows = [t for name in order for t in paired if t["comparison"] == name and t["family"] == family]
        ys = np.arange(len(rows))[::-1]
        for y, t in zip(ys, rows):
            eff = float(t["effect"]) * scale
            lo, hi = float(t["ci95"][0]) * scale, float(t["ci95"][1]) * scale
            ax.hlines(y, lo, hi, color="#0072B2", lw=2.2)
            ax.plot(eff, y, "o", color="#0072B2", ms=6.5)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ys, [short[t["comparison"]] for t in rows])
        ax.set_title("(a) Feasibility" if family == "feasibility" else "(b) Completion", loc="left")
    axes[0].set_xlabel("Feasibility effect (percentage points)\nFA-HPPO better →")
    axes[1].set_xlabel("Completion effect\n← FA-HPPO better")
    fig.suptitle("Paired FA-HPPO − baseline effects (all Holm-adjusted p < 0.001)", fontsize=9.5)
    fig.tight_layout()
    save_png(fig, FIG / "fig03_effect_sizes.png")


def load_ablation():
    rows = []
    for code, _ in ABL_ORDER:
        for seed in SEEDS:
            rows.append(load_json(V3 / "ablation" / code / f"seed_{seed}" / "validation.json"))
    return rows


def fig04_ablation(ablation):
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.5))
    labels = [lab for _c, lab in ABL_ORDER]
    xs = np.arange(len(ABL_ORDER))
    rng = np.random.default_rng(0)
    for i, (code, _lab) in enumerate(ABL_ORDER):
        subset = [r for r in ablation if r["variant"] == code]
        jitter = rng.uniform(-0.12, 0.12, size=len(subset))
        feas = [r["parent_balanced_val_feasibility"] for r in subset]
        axes[0].scatter(i + jitter, feas, color=COLOR["HybridPPO"] if code == "B2" else "#666666", s=30, zorder=3, alpha=0.9)
        axes[0].plot([i - 0.2, i + 0.2], [mean(feas), mean(feas)], color="black", lw=1.8, zorder=4)
        vl = [r["optimization"]["value_loss_mean"] for r in subset]
        gn = [r["optimization"]["grad_norm_preclip_mean"] for r in subset]
        axes[1].scatter(i + jitter, vl, color="#D55E00", s=30, zorder=3)
        axes[1].plot([i - 0.2, i + 0.2], [mean(vl), mean(vl)], color="black", lw=1.8, zorder=4)
        axes[2].scatter(i + jitter, gn, color="#E69F00", s=30, zorder=3)
        axes[2].plot([i - 0.2, i + 0.2], [mean(gn), mean(gn)], color="black", lw=1.8, zorder=4)
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
        ax.set_xticks(xs, labels)
    fig.suptitle("DEVELOPMENT / GOLD VALIDATION — not V3 TEST", fontsize=9.5, color="#8B0000")
    fig.tight_layout()
    save_png(fig, FIG / "fig04_ablation_and_training_stability.png")


def fig05_amount(rows):
    methods = ["HybridPPO", "FA-HPPO-Max", "FA-HPPO-Min"]
    summaries = {m: summarize_learned(rows, m) for m in methods}
    amount = load_json(STATS / "amount_sensitivity.json") if (STATS / "amount_sensitivity.json").is_file() else None
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, key_mean, key_ci, key_per, ylab, title in (
        (axes[0], "feasibility_mean", "feasibility_ci95", "per_seed_feasibility", "Feasibility", "(a) Feasibility"),
        (axes[1], "completion_all_mean", "completion_all_ci95", "per_seed_completion", "Completion", "(b) Failure-retaining completion"),
    ):
        for i, m in enumerate(methods):
            s = summaries[m]
            vals = [s[key_per][seed] for seed in SEEDS]
            ax.scatter(np.full(5, i), vals, color=COLOR[m], s=30, zorder=3, edgecolors="white", linewidths=0.4)
            ax.plot([i - 0.18, i + 0.18], [s[key_mean], s[key_mean]], color="black", lw=1.8)
            lo, hi = s[key_ci]
            ax.errorbar(i, s[key_mean], yerr=[[s[key_mean] - lo], [hi - s[key_mean]]], fmt="none", ecolor="black", elinewidth=1.2, capsize=3.5)
        ax.set_xticks(range(3), ["FA-HPPO\nlearned u", "FA-HPPO-Max\nu=1", "FA-HPPO-Min\nu=0"])
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        if key_mean == "feasibility_mean":
            ax.set_ylim(0, 1.05)
    tie = amount["route_averaged_feasibility_vs_max"]["tied"] if amount else 173
    axes[0].text(0.02, 0.08, f"FA-HPPO vs Max: {tie}/180 routes tied", transform=axes[0].transAxes, fontsize=7)
    fig.suptitle("V3 TEST amount-policy sensitivity (same frozen checkpoints)", fontsize=9.5)
    fig.tight_layout()
    save_png(fig, FIG / "fig05_amount_sensitivity.png")


def fig_a01_difficulty(rows):
    """Feasibility heatmap + mean station visits by cell (not redundant failure-rate panel)."""
    hybrid = method_rows(rows, "HybridPPO")
    charge = [r for r in hybrid if r.get("charge_class") == "charging_required"]
    fails = [r for r in hybrid if not r["feasible"]]
    layouts = ["R", "C", "RC"]
    bins = ["short", "medium", "long"]
    feas = np.zeros((3, 3))
    visits = np.zeros((3, 3))
    for i, layout in enumerate(layouts):
        for j, length in enumerate(bins):
            cell = [r for r in charge if r.get("layout") == layout and r.get("length_bin") == length]
            feas[i, j] = mean(float(r["feasible"]) for r in cell) if cell else np.nan
            visits[i, j] = mean(float(r.get("n_station_visits") or 0) for r in cell) if cell else np.nan
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    im0 = axes[0].imshow(feas, vmin=0, vmax=1, cmap="viridis")
    axes[0].set_xticks(range(3), bins)
    axes[0].set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{feas[i, j]:.2f}", ha="center", va="center", color="white" if feas[i, j] < 0.55 else "black", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[0].set_title("(a) Charging-required feasibility", loc="left")
    im1 = axes[1].imshow(visits, cmap="cividis")
    axes[1].set_xticks(range(3), bins)
    axes[1].set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{visits[i, j]:.2f}", ha="center", va="center", color="white" if visits[i, j] > np.nanmean(visits) else "black", fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title("(b) Mean station visits", loc="left")
    reasons = Counter(r.get("reason") for r in fails)
    note = f"{len(fails)} route×seed failures; " + ", ".join(f"{k}={v}" for k, v in reasons.items())
    fig.suptitle(note, fontsize=8.5)
    fig.tight_layout()
    save_png(fig, FIG_A / "figA01_difficulty_heatmap.png")


def fig_a02_seeds(summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
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
    fig.suptitle("Appendix — FA-HPPO seed robustness (V3 TEST)", fontsize=9.5)
    fig.tight_layout()
    save_png(fig, FIG_A / "figA02_seed_robustness.png")


def fig_a03_frvcp():
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
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    x = np.arange(len(order))
    axes[0].bar(x, [feas[m] for m in order], color=[colors[m] for m in order], width=0.7)
    axes[0].set_xticks(x, [labels[m] for m in order], rotation=12)
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
    save_png(fig, FIG_A / "figA03_native_frvcp_reference.png")


def fig_a04_learning_curves(ablation):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = {"B0": "#999999", "B1": "#E69F00", "B3": "#56B4E9", "B2": "#0072B2"}
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
                ys.append(float(row["val_parent_balanced_feasibility"]))
            if xs:
                series.append((np.asarray(xs), np.asarray(ys)))
        if not series:
            continue
        # interpolate onto common updates
        grid = sorted({int(x) for xs, _ in series for x in xs})
        mat = []
        for xs, ys in series:
            mat.append(np.interp(grid, xs, ys))
        mat = np.asarray(mat)
        mu, sd = mat.mean(0), mat.std(0, ddof=1) if len(mat) > 1 else np.zeros_like(mat[0])
        ax.plot(grid, mu, color=colors[code], lw=1.8, label=lab.replace("\n", " "))
        ax.fill_between(grid, mu - sd, mu + sd, color=colors[code], alpha=0.15)
    ax.set_xlabel("Training update")
    ax.set_ylabel("Parent-balanced VAL feasibility")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("DEVELOPMENT learning curves (gold VAL) — not V3 TEST", loc="left", color="#8B0000", fontsize=9)
    fig.tight_layout()
    save_png(fig, FIG_A / "figA04_learning_curves.png")


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
        "FA-HPPO = mean over 5 training seeds; baselines deterministic. Infeasible completion = horizon H. CI = hierarchical bootstrap (seed→route). All Holm-adjusted primary p < 0.001 (Monte Carlo floor).",
    )
    rows_abl = []
    for code, lab in ABL_ORDER:
        subset = [r for r in ablation if r["variant"] == code]
        rows_abl.append(
            [
                lab.replace("\n", " "),
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
        "DEVELOPMENT / GOLD VALIDATION ONLY — not V3 TEST. Descriptive names primary; B0/B1/B3/B2 are provenance labels. Five seeds 42–46.",
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
    build_failure_analysis(rows)


def build_failure_analysis(rows) -> None:
    """Frozen-raw failure summary. No TEST replay; no pre-failure trajectories."""
    hybrid = method_rows(rows, "HybridPPO")
    by_route = defaultdict(list)
    for r in hybrid:
        by_route[r["route_id"]].append(r)
    fail_counts = []
    table_rows = []
    for route_id, rr in sorted(by_route.items()):
        n_fail = sum(1 for r in rr if not r["feasible"])
        fail_counts.append(n_fail)
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

    # Fig A05: consistency histogram + layout×length failure rate among charging-required
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    counts = Counter(fail_counts)
    xs = np.arange(0, 6)
    ys = [counts.get(int(x), 0) for x in xs]
    axes[0].bar(xs, ys, color="#0072B2", edgecolor="white", width=0.7)
    for x, y in zip(xs, ys):
        axes[0].text(x, y + 0.5, str(y), ha="center", va="bottom", fontsize=7)
    axes[0].set_xlabel("Number of FA-HPPO seeds failing route (0–5)")
    axes[0].set_ylabel("Number of routes")
    axes[0].set_title("(a) Failure consistency across seeds", loc="left")
    axes[0].set_xticks(xs)

    layouts = ["R", "C", "RC"]
    bins = ["short", "medium", "long"]
    rate = np.zeros((3, 3))
    for i, layout in enumerate(layouts):
        for j, length in enumerate(bins):
            cell_routes = [
                rid
                for rid, rr in by_route.items()
                if rr[0].get("layout") == layout
                and rr[0].get("length_bin") == length
                and rr[0].get("charge_class") == "charging_required"
            ]
            if not cell_routes:
                rate[i, j] = np.nan
                continue
            # route-level: fraction of (route×seed) failures
            cell_rows = [r for rid in cell_routes for r in by_route[rid]]
            rate[i, j] = mean(float(not r["feasible"]) for r in cell_rows)
    im = axes[1].imshow(rate, vmin=0, vmax=max(0.25, float(np.nanmax(rate))), cmap="magma")
    axes[1].set_xticks(range(3), bins)
    axes[1].set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            if np.isnan(rate[i, j]):
                axes[1].text(j, i, "NA", ha="center", va="center", color="white", fontsize=8)
            else:
                axes[1].text(j, i, f"{rate[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title("(b) Charging-required failure rate", loc="left")
    fig.suptitle("Frozen V3 raw failure analysis — no TEST replay", fontsize=9, color="#8B0000")
    fig.tight_layout()
    save_png(fig, FIG_A / "figA05_failure_consistency.png")


def write_readme():
    (OUT / "README.md").write_text(
        """# Publication outputs (`results_paper/`)

PNG figures only. Single source of truth for the manuscript.

## Main paper

| Artifact | Role | Evidence |
|----------|------|----------|
| `figures/fig01_method_case_study.png` | Method illustration (route + SOC-along-route + hybrid decision) | **VAL** case study JSON — not TEST |
| `figures/fig02_main_test.png` | Confirmatory performance | V3 TEST |
| `figures/fig03_effect_sizes.png` | Paired effect sizes | V3 TEST / `paired_primary.json` |
| `figures/fig04_ablation_and_training_stability.png` | Methodology components | gold VAL ablation |
| `figures/fig05_amount_sensitivity.png` | Amount-policy sensitivity | V3 TEST |
| `tables/table01_*` … `table04_*` | Protocol / main / ablation / amount | see MANIFEST |

## Appendix

| Artifact | Role |
|----------|------|
| `figures/appendix/figA01_difficulty_heatmap.png` | Cell feasibility + station visits |
| `figures/appendix/figA02_seed_robustness.png` | Per-seed robustness |
| `figures/appendix/figA03_native_frvcp_reference.png` | Native FRVCP reference |
| `figures/appendix/figA04_learning_curves.png` | Development VAL learning curves |
| `figures/appendix/figA05_failure_consistency.png` | Frozen-raw failure consistency |
| `tables/tableA01_*` … `tableA03_*` | Per-seed / paired / failure routes |
| `case_study/illustrative_val_episode.json` | Source for Fig. 1 |

## Regenerate

```bash
python scripts/paper/record_illustrative_val_episode.py
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Does **not** retrain or re-evaluate the consumed V3 TEST.
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
    pngs = list(FIG.rglob("*.png"))
    expected = {
        "fig01_method_case_study.png",
        "fig02_main_test.png",
        "fig03_effect_sizes.png",
        "fig04_ablation_and_training_stability.png",
        "fig05_amount_sensitivity.png",
        "figA01_difficulty_heatmap.png",
        "figA02_seed_robustness.png",
        "figA03_native_frvcp_reference.png",
        "figA04_learning_curves.png",
        "figA05_failure_consistency.png",
    }
    names = {p.name for p in pngs}
    missing = expected - names
    if missing:
        raise SystemExit(f"missing PNGs: {sorted(missing)}")
    if not (TAB / "tableA03_failure_routes.md").is_file():
        raise SystemExit("missing tableA03_failure_routes")
    for path in FIG.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".png":
            raise SystemExit(f"non-PNG under figures: {path}")
    # remove obsolete names if still present
    obsolete = {
        "fig01_main_test.png",
        "fig02_effect_sizes.png",
        "fig03_ablation_and_training_stability.png",
        "fig04_amount_sensitivity.png",
        "fig05_difficulty_and_failures.png",
        "figA02_native_frvcp_reference.png",
    }
    leftovers = obsolete & names
    if leftovers:
        raise SystemExit(f"obsolete figure names still present: {sorted(leftovers)}")
    print(json.dumps({"verify": "ok", "n_rows": 3240, "n_instances": 180, "n_png": len(pngs)}, indent=2))


def main() -> None:
    if "--verify" in sys.argv:
        verify()
        return
    style()
    # Remove obsolete figure filenames from prior numbering
    for old in [
        FIG / "fig01_main_test.png",
        FIG / "fig02_effect_sizes.png",
        FIG / "fig03_ablation_and_training_stability.png",
        FIG / "fig04_amount_sensitivity.png",
        FIG / "fig05_difficulty_and_failures.png",
        FIG_A / "figA01_seed_robustness.png",
        FIG_A / "figA02_native_frvcp_reference.png",
    ]:
        if old.is_file():
            old.unlink()
    rows = load_rows()
    fig01_case_study()
    all_s, ch_s = fig02_main(rows)
    fig03_effects()
    ablation = load_ablation()
    fig04_ablation(ablation)
    fig05_amount(rows)
    fig_a01_difficulty(rows)
    fig_a02_seeds(all_s["HybridPPO"])
    fig_a03_frvcp()
    fig_a04_learning_curves(ablation)
    build_tables(all_s, ch_s, ablation, rows)
    write_readme()
    artifacts = [
        {"path": "results_paper/README.md", "role": "index"},
        {"path": "results_paper/case_study/illustrative_val_episode.json", "role": "methodology-illustration", "dataset": "SynthCharge VAL", "note": "not TEST"},
        {"path": "results_paper/figures/fig01_method_case_study.png", "role": "main-methodology", "dataset": "SynthCharge VAL", "n_seeds": 1},
        {"path": "results_paper/figures/fig02_main_test.png", "role": "main", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5},
        {"path": "results_paper/figures/fig03_effect_sizes.png", "role": "main", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5},
        {"path": "results_paper/figures/fig04_ablation_and_training_stability.png", "role": "main-development", "dataset": "gold VAL", "n_seeds": 5},
        {"path": "results_paper/figures/fig05_amount_sensitivity.png", "role": "main", "dataset": "SynthCharge V3 TEST", "n_routes": 180, "n_seeds": 5},
        {"path": "results_paper/figures/appendix/figA01_difficulty_heatmap.png", "role": "appendix", "dataset": "SynthCharge V3 TEST"},
        {"path": "results_paper/figures/appendix/figA02_seed_robustness.png", "role": "appendix", "dataset": "SynthCharge V3 TEST"},
        {"path": "results_paper/figures/appendix/figA03_native_frvcp_reference.png", "role": "appendix", "dataset": "native FRVCP"},
        {"path": "results_paper/figures/appendix/figA04_learning_curves.png", "role": "appendix-development", "dataset": "gold VAL"},
        {"path": "results_paper/figures/appendix/figA05_failure_consistency.png", "role": "appendix", "dataset": "SynthCharge V3 TEST", "note": "frozen raw only"},
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
