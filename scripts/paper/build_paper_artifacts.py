"""Regenerate V3-HPPO paper statistics, tables, and figures from frozen raw rows.

Does NOT train, select checkpoints, or evaluate TEST.
"""

from __future__ import annotations

import csv
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

from experiments.stats import (  # noqa: E402
    hierarchical_bootstrap_ci,
    holm,
    mean_sd_across_training_seeds,
    paired_parent_diff_hierarchical,
    permutation_pvalue,
)

from common import BASELINES, SEEDS, V3, dump_json, lf_sha256, load_json, sha256  # noqa: E402

RAW = V3 / "raw" / "synthcharge_test.jsonl"
STATS = V3 / "statistics"
TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"
N_BOOT = 5000
N_PERM = 20000
RNG = 20261002
METHODS_MAIN = ("HybridPPO",) + BASELINES
AMOUNT = ("FA-HPPO-Min", "FA-HPPO-Max")
LABEL = {
    "HybridPPO": "FA-HPPO",
    "FA-HPPO-Min": "FA-HPPO-Min",
    "FA-HPPO-Max": "FA-HPPO-Max",
    "GreedyMinimumSufficientCharge": "GreedyMin",
    "GreedyFullCharge": "GreedyFull",
    "OneStepLookahead": "Lookahead",
}
COLORS = {
    "HybridPPO": "#0072B2",
    "GreedyMinimumSufficientCharge": "#E69F00",
    "GreedyFullCharge": "#009E73",
    "OneStepLookahead": "#D55E00",
    "FA-HPPO-Min": "#56B4E9",
    "FA-HPPO-Max": "#CC79A7",
}


def load_rows() -> list[dict]:
    if not RAW.is_file():
        raise SystemExit(f"missing raw rows: {RAW}")
    return [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]


def method_rows(rows, method, seed=None):
    out = [r for r in rows if r["method"] == method]
    if seed is not None:
        out = [r for r in out if r.get("seed") == seed]
    return out


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def sd(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else None


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "NA"
    return f"{v:.{d}f}"


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_md(path: Path, title: str, header, rows, note: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", note, "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path: Path, caption: str, label: str, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = "l" + "c" * (len(header) - 1)
    lines = [
        "% Auto-generated; do not hand-edit numbers.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_fig(fig, stem: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def integrity(rows: list[dict]) -> dict:
    issues = []
    route_ids = sorted({r["route_id"] for r in rows})
    if len(route_ids) != 180:
        issues.append(f"unique routes {len(route_ids)} != 180")
    expected = 180 * (5 + 5 + 5 + 3)
    if len(rows) != expected:
        issues.append(f"row count {len(rows)} != {expected}")
    if "DiscretePPO" in {r["method"] for r in rows}:
        issues.append("DiscretePPO present in V3 TEST")
    for method in ("HybridPPO",) + AMOUNT:
        subset = method_rows(rows, method)
        if len(subset) != 180 * 5:
            issues.append(f"{method} rows {len(subset)} != 900")
    for method in BASELINES:
        subset = method_rows(rows, method)
        if len(subset) != 180:
            issues.append(f"{method} rows {len(subset)} != 180")
    if {r["physics_profile"] for r in rows} != {"synthcharge_linear"}:
        issues.append("bad physics profiles")
    for r in rows:
        if not r["feasible"] and abs(float(r["completion_time_all_routes"]) - float(r["horizon"])) > 1e-9:
            issues.append(f"infeasible completion not H: {r['route_id']}")
            break
    consumed = V3 / "EVALUATION_CONSUMED.json"
    if not consumed.is_file():
        issues.append("EVALUATION_CONSUMED missing")
    else:
        if load_json(consumed).get("raw_sha256_lf") != lf_sha256(RAW):
            issues.append("consumed raw hash mismatch")
    freeze = load_json(V3 / "CHECKPOINT_FREEZE.json")
    for item in freeze["checkpoints"]:
        if sha256(ROOT / item["checkpoint"]) != item["checkpoint_sha256"]:
            issues.append(f"checkpoint changed: {item['checkpoint']}")
    lock = load_json(V3 / "TEST_LOCK.json")
    bad = [rel for rel, digest in lock["files"].items() if lf_sha256(ROOT / rel) != digest]
    if bad:
        issues.append(f"TEST_LOCK mismatch: {bad[:3]}")
    payload = {"n_issues": len(issues), "issues": issues, "n_rows": len(rows), "n_routes": len(route_ids)}
    if issues:
        raise SystemExit(f"integrity failed: {issues}")
    dump_json(STATS / "integrity.json", payload)
    return payload


def summarize_learned(rows, method):
    learned = []
    for r in method_rows(rows, method):
        copy = dict(r)
        copy["feasible_f"] = float(bool(r["feasible"]))
        learned.append(copy)
    feas = mean_sd_across_training_seeds(learned, "feasible_f")
    comp = mean_sd_across_training_seeds(learned, "completion_time_all_routes")
    feas_ci = hierarchical_bootstrap_ci(learned, "feasible_f", n_boot=N_BOOT, seed=RNG)
    comp_ci = hierarchical_bootstrap_ci(learned, "completion_time_all_routes", n_boot=N_BOOT, seed=RNG)
    return {
        "method": method,
        "label": LABEL[method],
        "n_routes": 180,
        "n_seeds": 5,
        "feasibility_mean": feas["mean_across_seeds"],
        "feasibility_sd": feas["sd_across_seeds"],
        "feasibility_ci95": [feas_ci["lo"], feas_ci["hi"]],
        "completion_all_mean": comp["mean_across_seeds"],
        "completion_all_sd": comp["sd_across_seeds"],
        "completion_all_ci95": [comp_ci["lo"], comp_ci["hi"]],
        "per_seed_feasibility": {int(k): v["mean"] for k, v in feas["per_seed"].items()},
        "per_seed_completion": {int(k): v["mean"] for k, v in comp["per_seed"].items()},
        "charging_time": mean(
            mean(r["total_charging_time"] for r in method_rows(rows, method, seed) if r["feasible"]) for seed in SEEDS
        ),
        "station_visits": mean(
            mean(r["n_station_visits"] for r in method_rows(rows, method, seed) if r["feasible"]) for seed in SEEDS
        ),
        "runtime_s": mean(r.get("runtime_s") for r in method_rows(rows, method)),
    }


def summarize_baseline(rows, method):
    subset = method_rows(rows, method)
    feasible = [r for r in subset if r["feasible"]]
    return {
        "method": method,
        "label": LABEL[method],
        "n_routes": len(subset),
        "n_seeds": 1,
        "feasibility_mean": mean(float(r["feasible"]) for r in subset),
        "feasibility_sd": None,
        "feasibility_ci95": None,
        "completion_all_mean": mean(r["completion_time_all_routes"] for r in subset),
        "completion_all_sd": None,
        "completion_all_ci95": None,
        "per_seed_feasibility": None,
        "charging_time": mean(r["total_charging_time"] for r in feasible),
        "station_visits": mean(r["n_station_visits"] for r in feasible),
        "runtime_s": mean(r.get("runtime_s") for r in subset),
    }


def paired_vs_baselines(rows):
    hybrid = []
    for r in method_rows(rows, "HybridPPO"):
        copy = dict(r)
        copy["feasible_f"] = float(bool(r["feasible"]))
        hybrid.append(copy)
    tests = []
    for baseline in BASELINES:
        base = []
        for r in method_rows(rows, baseline):
            copy = dict(r)
            copy["feasible_f"] = float(bool(r["feasible"]))
            # Expand deterministic baseline to pair with each seed average via seed-averaged hybrid.
            base.append(copy)
        for family, key in (("feasibility", "feasible_f"), ("completion_all", "completion_time_all_routes")):
            # Seed-average hybrid per route, then pair with baseline route values.
            by_route_h = defaultdict(list)
            for r in hybrid:
                by_route_h[r["route_id"]].append(float(r[key]))
            h_mean = {rid: float(np.mean(vs)) for rid, vs in by_route_h.items()}
            b_map = {r["route_id"]: float(r[key]) for r in base}
            diffs = [h_mean[rid] - b_map[rid] for rid in sorted(h_mean) if rid in b_map]
            arr = np.asarray(diffs, dtype=float)
            p = permutation_pvalue(diffs, n_perm=N_PERM, seed=RNG)
            rng = np.random.default_rng(RNG)
            boots = [float(arr[rng.integers(0, arr.size, size=arr.size)].mean()) for _ in range(N_BOOT)] if arr.size else []
            lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
            tests.append(
                {
                    "family": family,
                    "comparison": f"HybridPPO - {baseline}",
                    "effect": float(arr.mean()) if arr.size else None,
                    "ci95": [lo, hi],
                    "p_raw": p,
                    "n_independent_units": len(diffs),
                }
            )
    adjusted = holm([(t["comparison"] + "|" + t["family"], t["p_raw"]) for t in tests])
    for t, (_name, _p, adj) in zip(tests, adjusted):
        t["p_holm"] = adj
    return tests


def subset_charge(rows):
    return [r for r in rows if r.get("charge_class") == "charging_required"]


def figure_main(rows, summaries):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    methods = list(METHODS_MAIN)
    x = np.arange(len(methods))
    feas = [summaries[m]["feasibility_mean"] for m in methods]
    axes[0].bar(x, feas, color=[COLORS[m] for m in methods], width=0.7)
    for i, m in enumerate(methods):
        s = summaries[m]
        if s.get("feasibility_ci95"):
            lo, hi = s["feasibility_ci95"]
            axes[0].errorbar(i, s["feasibility_mean"], yerr=[[s["feasibility_mean"] - lo], [hi - s["feasibility_mean"]]], fmt="none", ecolor="black", capsize=3)
        if s.get("per_seed_feasibility"):
            ys = list(s["per_seed_feasibility"].values())
            axes[0].scatter(np.full(len(ys), i), ys, color="black", s=18, zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABEL[m] for m in methods], rotation=15)
    axes[0].set_ylabel("Feasibility")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("A. Feasibility")

    comp = [summaries[m]["completion_all_mean"] for m in methods]
    axes[1].bar(x, comp, color=[COLORS[m] for m in methods], width=0.7)
    for i, m in enumerate(methods):
        s = summaries[m]
        if s.get("completion_all_ci95"):
            lo, hi = s["completion_all_ci95"]
            axes[1].errorbar(i, s["completion_all_mean"], yerr=[[s["completion_all_mean"] - lo], [hi - s["completion_all_mean"]]], fmt="none", ecolor="black", capsize=3)
        if s.get("per_seed_completion"):
            ys = list(s["per_seed_completion"].values())
            axes[1].scatter(np.full(len(ys), i), ys, color="black", s=18, zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABEL[m] for m in methods], rotation=15)
    axes[1].set_ylabel("Failure-retaining completion")
    axes[1].set_title("B. Completion (infeasible → H)")
    fig.suptitle("V3 SynthCharge TEST (180 routes; FA-HPPO 5 seeds)")
    fig.tight_layout()
    save_fig(fig, "fig02_main_test")


def figure_charge_required(rows):
    sub = subset_charge(rows)
    summaries = {}
    for m in METHODS_MAIN:
        if m == "HybridPPO":
            summaries[m] = summarize_learned(sub, m)
            summaries[m]["n_routes"] = len({r["route_id"] for r in sub})
        else:
            summaries[m] = summarize_baseline(sub, m)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    methods = list(METHODS_MAIN)
    x = np.arange(len(methods))
    feas = [summaries[m]["feasibility_mean"] for m in methods]
    axes[0].bar(x, feas, color=[COLORS[m] for m in methods], width=0.7)
    for i, m in enumerate(methods):
        s = summaries[m]
        if s.get("per_seed_feasibility"):
            ys = list(s["per_seed_feasibility"].values())
            axes[0].scatter(np.full(len(ys), i), ys, color="black", s=18, zorder=3)
        if s.get("feasibility_ci95"):
            lo, hi = s["feasibility_ci95"]
            axes[0].errorbar(
                i,
                s["feasibility_mean"],
                yerr=[[s["feasibility_mean"] - lo], [hi - s["feasibility_mean"]]],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABEL[m] for m in methods], rotation=15)
    axes[0].set_ylabel("Feasibility")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("A. Charging-required feasibility")
    comp = [summaries[m]["completion_all_mean"] for m in methods]
    axes[1].bar(x, comp, color=[COLORS[m] for m in methods], width=0.7)
    for i, m in enumerate(methods):
        s = summaries[m]
        if s.get("per_seed_completion"):
            ys = list(s["per_seed_completion"].values())
            axes[1].scatter(np.full(len(ys), i), ys, color="black", s=18, zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABEL[m] for m in methods], rotation=15)
    axes[1].set_ylabel("Failure-retaining completion")
    axes[1].set_title("B. Charging-required completion")
    fig.suptitle("Charging-required subset (~144 routes)")
    fig.tight_layout()
    save_fig(fig, "fig03_charging_required")
    return summaries


def figure_seed_robustness(summary):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    seeds = list(SEEDS)
    feas = [summary["per_seed_feasibility"][s] for s in seeds]
    comp = [summary["per_seed_completion"][s] for s in seeds]
    axes[0].scatter(seeds, feas, s=60, color=COLORS["HybridPPO"], zorder=3)
    axes[0].axhline(summary["feasibility_mean"], color="black", ls="--", lw=1)
    if summary["feasibility_ci95"]:
        lo, hi = summary["feasibility_ci95"]
        axes[0].axhspan(lo, hi, color=COLORS["HybridPPO"], alpha=0.15)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Feasibility")
    axes[0].set_title("Per-seed feasibility")
    axes[1].scatter(seeds, comp, s=60, color=COLORS["HybridPPO"], zorder=3)
    axes[1].axhline(summary["completion_all_mean"], color="black", ls="--", lw=1)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Completion")
    axes[1].set_title("Per-seed failure-retaining completion")
    fig.suptitle("FA-HPPO five-seed robustness (V3 TEST)")
    fig.tight_layout()
    save_fig(fig, "fig04_seed_robustness")


def figure_heatmap(rows):
    hybrid = method_rows(rows, "HybridPPO")
    charge = [r for r in hybrid if r.get("charge_class") == "charging_required"]
    layouts = ["R", "C", "RC"]
    bins = ["short", "medium", "long"]
    mat = np.zeros((3, 3))
    for i, layout in enumerate(layouts):
        for j, length in enumerate(bins):
            cell = [r for r in charge if r.get("layout") == layout and r.get("length_bin") == length]
            mat[i, j] = mean(float(r["feasible"]) for r in cell) if cell else float("nan")
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(3), bins)
    ax.set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="FA-HPPO feasibility")
    ax.set_title("Charging-required difficulty heatmap")
    fig.tight_layout()
    save_fig(fig, "fig07_difficulty_heatmap")


def figure_failures(rows):
    hybrid = [r for r in method_rows(rows, "HybridPPO") if not r["feasible"]]
    reasons = Counter(r.get("reason") or "none" for r in hybrid)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = list(reasons.keys())
    vals = [reasons[k] for k in labels]
    axes[0].barh(labels, vals, color="#D55E00")
    axes[0].set_xlabel("Failed evaluations (route×seed)")
    axes[0].set_title("A. FA-HPPO failure reasons")
    # panel B: failure rate by cell among charging-required
    charge = [r for r in method_rows(rows, "HybridPPO") if r.get("charge_class") == "charging_required"]
    cells = []
    rates = []
    for layout in ("R", "C", "RC"):
        for length in ("short", "medium", "long"):
            cell = [r for r in charge if r.get("layout") == layout and r.get("length_bin") == length]
            if not cell:
                continue
            cells.append(f"{layout}/{length}")
            rates.append(1.0 - mean(float(r["feasible"]) for r in cell))
    axes[1].bar(range(len(cells)), rates, color="#0072B2")
    axes[1].set_xticks(range(len(cells)), cells, rotation=45, ha="right")
    axes[1].set_ylabel("Failure rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("B. Failure rate by cell")
    fig.tight_layout()
    save_fig(fig, "fig08_failure_analysis")
    return dict(reasons)


def figure_ablation():
    """Development ablation figure from results/v3_hppo/ablation if available."""
    rows = []
    for variant in ("B0", "B1", "B3", "B2"):
        for seed in SEEDS:
            path = V3 / "ablation" / variant / f"seed_{seed}" / "validation.json"
            if path.is_file():
                rows.append(load_json(path))
    if len(rows) < 8:
        # Fallback note figure
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"Ablation incomplete ({len(rows)}/20 runs).\nSee results/v3_hppo/ablation/", ha="center", va="center")
        save_fig(fig, "fig05_ablation")
        save_fig(fig, "fig06_training_stability")
        return rows
    # Feasibility 2x2
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for variant, marker, color in (
        ("B0", "o", "#999999"),
        ("B1", "s", "#E69F00"),
        ("B3", "^", "#56B4E9"),
        ("B2", "D", "#0072B2"),
    ):
        subset = [r for r in rows if r["variant"] == variant]
        xs = [1 if r["return_scale_enabled"] else 0 for r in subset]
        # jitter by time_aware
        xs = [x + (0.08 if r["time_aware"] else -0.08) for x, r in zip(xs, subset)]
        ys = [r["parent_balanced_val_feasibility"] for r in subset]
        ax.scatter(xs, ys, marker=marker, color=color, s=55, label=variant, zorder=3)
    ax.set_xticks([0, 1], ["return scale OFF", "return scale ON"])
    ax.set_ylabel("Parent-balanced VAL feasibility")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Variant")
    ax.set_title("B0/B1/B3/B2 development ablation (gold VAL)")
    fig.tight_layout()
    save_fig(fig, "fig05_ablation")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for variant, color in (("B0", "#999999"), ("B1", "#E69F00"), ("B3", "#56B4E9"), ("B2", "#0072B2")):
        subset = [r for r in rows if r["variant"] == variant]
        axes[0].scatter(
            [variant] * len(subset),
            [r["optimization"]["value_loss_mean"] for r in subset],
            color=color,
            s=50,
            label=variant,
        )
        axes[1].scatter(
            [variant] * len(subset),
            [r["optimization"]["grad_norm_preclip_mean"] for r in subset],
            color=color,
            s=50,
            label=variant,
        )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("Mean value loss")
    axes[1].set_title("Mean grad norm (pre-clip)")
    axes[0].legend()
    fig.suptitle("Training stability (development ablation)")
    fig.tight_layout()
    save_fig(fig, "fig06_training_stability")
    return rows


def figure_method_overview():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, "Fixed customer\nroute"),
        (0.18, "Simulator\nstate"),
        (0.34, "Feasibility\nshield"),
        (0.50, "Admissible\nCONTINUE/stations"),
        (0.66, "[SOC_lo, SOC_hi]\namount envelope"),
        (0.82, "HybridPPO\ndiscrete+Beta u"),
    ]
    for x, text in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.14, 0.4, fill=False, lw=1.5, color="#0072B2"))
        ax.text(x + 0.07, 0.55, text, ha="center", va="center", fontsize=9)
        if x < 0.82:
            ax.annotate("", xy=(x + 0.15, 0.55), xytext=(x + 0.14, 0.55), arrowprops=dict(arrowstyle="->", color="black"))
    ax.text(0.5, 0.12, "Does not choose customer order. Charging when / where / how much only.", ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("FA-HPPO method overview")
    save_fig(fig, "fig01_method_overview")


def figure_frvcp_reference():
    src = ROOT / "results" / "final" / "figures" / "frvcpy_native" / "frvcp_gap_hist.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    note = (
        "Native FRVCP reference (separate from SynthCharge/EVRPTW-GR TEST).\n"
        "frvcpy is exact for its FRVCP assumptions, not for full EVRPTW-GR.\n"
    )
    if src.is_file():
        note += f"Historical figure retained at: {src.as_posix()}\nRegenerate from archived V1 tooling if needed."
    else:
        note += "Source figure not found in archive."
    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=10)
    ax.set_title("Figure 9 — Native FRVCP optimization reference")
    save_fig(fig, "fig09_native_frvcp_reference")


def build_tables(summaries, charge_summaries, failures, paired, ablation_rows):
    # Table 1
    header = ["Corpus", "Role", "TRAIN", "VAL", "TEST", "Charge-req TEST", "Physics", "Fresh TEST?"]
    rows = [
        ["SynthCharge V2 train/val", "development", "180", "90", "—", "—", "synthcharge_linear", "no"],
        ["SynthCharge V3 paper TEST", "confirmatory", "—", "—", "180", "144", "synthcharge_linear", "yes"],
        ["Gold official", "ablation TRAIN/VAL", "119", "47", "—", "—", "official_evrptwgr", "no"],
        ["V2 historical (Hybrid+DPPO)", "archived evidence", "—", "—", "90", "—", "mixed", "no (historical)"],
    ]
    write_md(TABLES / "table01_benchmark.md", "Table 1 — Benchmark summary", header, rows, "Split composition.")
    write_csv(TABLES / "table01_benchmark.csv", header, rows)
    write_tex(TABLES / "table01_benchmark.tex", "Benchmark and split summary.", "tab:benchmark", header, rows)

    # Table 2
    header = ["Method", "Feasibility", "95% CI", "Completion", "Charge time", "Visits", "Runtime"]
    rows = []
    for m in METHODS_MAIN:
        s = summaries[m]
        ci = "NA" if not s.get("feasibility_ci95") else f"[{fmt(s['feasibility_ci95'][0])}, {fmt(s['feasibility_ci95'][1])}]"
        rows.append(
            [
                LABEL[m],
                fmt(s["feasibility_mean"]),
                ci,
                fmt(s["completion_all_mean"], 1),
                fmt(s.get("charging_time")),
                fmt(s.get("station_visits"), 2),
                fmt(s.get("runtime_s"), 3),
            ]
        )
    write_md(TABLES / "table02_main_test.md", "Table 2 — Main V3 TEST", header, rows, "Confirmatory SynthCharge TEST.")
    write_csv(TABLES / "table02_main_test.csv", header, rows)
    write_tex(TABLES / "table02_main_test.tex", "Main fresh FA-HPPO TEST.", "tab:main", header, rows)

    # Table 3
    rows = []
    for m in METHODS_MAIN:
        s = charge_summaries[m]
        ci = "NA" if not s.get("feasibility_ci95") else f"[{fmt(s['feasibility_ci95'][0])}, {fmt(s['feasibility_ci95'][1])}]"
        rows.append([LABEL[m], fmt(s["feasibility_mean"]), ci, fmt(s["completion_all_mean"], 1)])
    header = ["Method", "Feasibility", "95% CI", "Completion"]
    write_md(TABLES / "table03_charging_required.md", "Table 3 — Charging-required", header, rows, "Charging-required subset.")
    write_csv(TABLES / "table03_charging_required.csv", header, rows)
    write_tex(TABLES / "table03_charging_required.tex", "Charging-required subset.", "tab:charge", header, rows)

    # Table 4 ablation
    header = ["Variant", "Time cap", "Return scale", "Seeds done", "Mean parent-bal. feas", "Mean value loss"]
    rows = []
    for variant, tw, sc in (("B0", "off", "off"), ("B1", "on", "off"), ("B3", "off", "on"), ("B2", "on", "on")):
        subset = [r for r in ablation_rows if r.get("variant") == variant]
        rows.append(
            [
                variant,
                tw,
                sc,
                str(len(subset)),
                fmt(mean(r["parent_balanced_val_feasibility"] for r in subset)),
                fmt(mean(r["optimization"]["value_loss_mean"] for r in subset), 4) if subset else "NA",
            ]
        )
    for method in AMOUNT:
        if method in summaries:
            s = summaries[method]
            rows.append([method, "on", "on (frozen)", "5 (eval)", fmt(s["feasibility_mean"]), "n/a (eval)"])
    write_md(
        TABLES / "table04_ablation.md",
        "Table 4 — Ablation",
        header,
        rows,
        "Development/validation ablation unless marked eval-on-TEST amount ablation.",
    )
    write_csv(TABLES / "table04_ablation.csv", header, rows)
    write_tex(TABLES / "table04_ablation.tex", "FA-HPPO ablation (development).", "tab:ablation", header, rows)

    # Table 5
    header = ["Reason", "Count (HybridPPO route×seed)"]
    rows = sorted(failures.items(), key=lambda kv: -kv[1])
    write_md(TABLES / "table05_failures.md", "Table 5 — Failures", header, rows, "FA-HPPO failure reasons on V3 TEST.")
    write_csv(TABLES / "table05_failures.csv", header, rows)
    write_tex(TABLES / "table05_failures.tex", "Failure analysis.", "tab:fail", header, rows)

    # Appendix per-seed
    header = ["Seed", "Feasibility", "Completion"]
    s = summaries["HybridPPO"]
    rows = [[str(seed), fmt(s["per_seed_feasibility"][seed]), fmt(s["per_seed_completion"][seed], 1)] for seed in SEEDS]
    write_md(TABLES / "tableA_per_seed.md", "Appendix — FA-HPPO per seed", header, rows, "Individual seeds.")
    write_csv(TABLES / "tableA_per_seed.csv", header, rows)

    dump_json(STATS / "paired_primary.json", paired)
    dump_json(STATS / "summaries.json", {k: summaries[k] for k in summaries})


def main() -> None:
    verify_only = "--verify" in sys.argv
    rows = load_rows()
    STATS.mkdir(parents=True, exist_ok=True)
    integrity(rows)
    summaries = {}
    for m in METHODS_MAIN + AMOUNT:
        if m in ("HybridPPO",) + AMOUNT:
            summaries[m] = summarize_learned(rows, m)
        else:
            summaries[m] = summarize_baseline(rows, m)
    paired = paired_vs_baselines(rows)
    dump_json(STATS / "main_summary.json", {LABEL[k]: summaries[k] for k in METHODS_MAIN})
    charge_summaries = {}
    sub = subset_charge(rows)
    for m in METHODS_MAIN:
        charge_summaries[m] = summarize_learned(sub, m) if m == "HybridPPO" else summarize_baseline(sub, m)
    failures = figure_failures(rows) if not verify_only else Counter()
    if verify_only:
        print(json.dumps({"verify": "ok", "n_rows": len(rows)}, indent=2))
        return
    figure_method_overview()
    figure_main(rows, summaries)
    charge_summaries = figure_charge_required(rows)
    figure_seed_robustness(summaries["HybridPPO"])
    ablation_rows = figure_ablation()
    figure_heatmap(rows)
    failures = figure_failures(rows)
    figure_frvcp_reference()
    build_tables(summaries, charge_summaries, failures, paired, ablation_rows)
    dump_json(
        V3 / "PAPER_ARTIFACTS_MANIFEST.json",
        {
            "figures": sorted(p.name for p in FIGURES.glob("fig*")),
            "tables": sorted(p.name for p in TABLES.glob("table*")),
            "raw_sha256_lf": lf_sha256(RAW),
        },
    )
    print("paper artifacts written")


if __name__ == "__main__":
    main()
