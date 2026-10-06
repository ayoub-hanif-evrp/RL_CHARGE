"""Build standalone V4 paper package (results_v4_paper/) from frozen evidence only.

No retraining. No TEST rerun. No prior-version comparisons in paper-facing outputs.
PNG only. Supports --verify.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_v4_paper"
FIG = OUT / "figures"
TAB = OUT / "tables"
STAT = OUT / "statistics"
CAP = OUT / "captions"
MAN = OUT / "manifests"

RAW = ROOT / "results" / "v4_test" / "raw" / "synthcharge_v4_test.jsonl"
LOCK = ROOT / "results" / "v4_test" / "TEST_LOCK.json"
CONSUMED = ROOT / "results" / "v4_test" / "EVALUATION_CONSUMED.json"
CKPT_FREEZE = ROOT / "results" / "v4_test" / "CHECKPOINT_FREEZE.json"
PROTOCOL = ROOT / "results" / "v4_test" / "PAPER_PROTOCOL.json"
REWARD_FREEZE = ROOT / "results" / "v4_reward" / "FINAL_REWARD_FREEZE.json"
ABLATION = ROOT / "results" / "v4_reward" / "SUMMARY.json"
AUTH = ROOT / "results" / "v4_reward" / "final_authoritative" / "V4_BASE_NO_L_FAIL"

SEEDS = (42, 43, 44, 45, 46)
T_CRIT_5 = 2.776445105
DPI = 600
METHOD_CODE = "V4_FA_HPPO"
METHODS = (
    METHOD_CODE,
    "OneStepLookahead",
    "GreedyFullCharge",
    "GreedyMinimumSufficientCharge",
)
LABEL = {
    METHOD_CODE: "FA-HPPO",
    "OneStepLookahead": "One-step lookahead",
    "GreedyFullCharge": "Greedy full charge",
    "GreedyMinimumSufficientCharge": "Greedy minimum charge",
}
SHORT = {
    METHOD_CODE: "FA-HPPO",
    "OneStepLookahead": "Lookahead",
    "GreedyFullCharge": "Greedy full",
    "GreedyMinimumSufficientCharge": "Greedy min",
}
COLOR = {
    METHOD_CODE: "#0072B2",
    "OneStepLookahead": "#D55E00",
    "GreedyFullCharge": "#009E73",
    "GreedyMinimumSufficientCharge": "#E69F00",
}
REWARD_ORDER = (
    ("V3_TIME", "Time + progress failure"),
    ("V4_BASE", "Normalized equivalent"),
    ("V4_PBRS", "Potential-shaped"),
    ("V4_BASE_NO_L_FAIL", "Time-horizon (selected)"),
)
REWARD_COLOR = {
    "V3_TIME": "#999999",
    "V4_BASE": "#56B4E9",
    "V4_PBRS": "#0072B2",
    "V4_BASE_NO_L_FAIL": "#E69F00",
}

FIG_ENTRIES: list[dict] = []


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig, name: str, *, split: str, sources: list[str]) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIG_ENTRIES.append(
        {
            "filename": path.name,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "source_files": sources,
            "source_hashes": {s: _sha(ROOT / s) if (ROOT / s).is_file() else None for s in sources},
            "builder_script": "scripts/paper/build_v4_results_paper.py",
            "figure_sha256": _sha(path),
            "evidence_split": split,
        }
    )
    return path


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _t_ci(vals: list[float]) -> tuple[float, float, float]:
    mu, sd = _mean_sd(vals)
    n = len(vals)
    if n <= 1:
        return mu, mu, mu
    half = T_CRIT_5 * sd / math.sqrt(n) if n == 5 else (2.776445105 * sd / math.sqrt(n))
    # use T_CRIT_5 for n=5; for other n still use same constant only when n==5 in this package
    if n != 5:
        # small-n table
        tcrit = {2: 12.706204736, 3: 4.3026527299, 4: 3.1824463053}.get(n, T_CRIT_5)
        half = tcrit * sd / math.sqrt(n)
    return mu, mu - half, mu + half


def load_rows() -> list[dict]:
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["method"] in METHODS]


def seed_metric(rows: list[dict], method: str, pred) -> list[float]:
    out = []
    for seed in SEEDS:
        sub = [r for r in rows if r["method"] == method and r.get("seed") == seed]
        if not sub:
            continue
        out.append(float(pred(sub)))
    return out


def feas_rate(sub: list[dict]) -> float:
    return sum(1 for r in sub if r["feasible"]) / len(sub)


def comp_mean(sub: list[dict]) -> float:
    return sum(float(r["completion_time_all_routes"]) for r in sub) / len(sub)


def runtime_mean(sub: list[dict]) -> float:
    return sum(float(r["runtime_s"]) for r in sub) / len(sub)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def build_statistics(rows: list[dict]) -> dict:
    main = {}
    for m in METHODS:
        recs = [r for r in rows if r["method"] == m]
        if m == METHOD_CODE:
            feas = seed_metric(rows, m, feas_rate)
            comp = seed_metric(rows, m, comp_mean)
            rt = seed_metric(rows, m, runtime_mean)
            mu, lo, hi = _t_ci(feas)
            cmu, clo, chi = _t_ci(comp)
            main[LABEL[m]] = {
                "feasibility_per_seed": {str(s): feas[i] for i, s in enumerate(SEEDS)},
                "feasibility_mean": mu,
                "feasibility_sd": _mean_sd(feas)[1],
                "feasibility_t95": [lo, hi],
                "completion_per_seed": {str(s): comp[i] for i, s in enumerate(SEEDS)},
                "completion_mean": cmu,
                "completion_sd": _mean_sd(comp)[1],
                "completion_t95": [clo, chi],
                "runtime_mean": float(np.mean(rt)),
            }
        else:
            main[LABEL[m]] = {
                "feasibility_mean": feas_rate(recs),
                "feasibility_sd": None,
                "completion_mean": comp_mean(recs),
                "runtime_mean": runtime_mean(recs),
                "n_routes": len(recs),
            }

    # paired win/tie/loss vs baselines (post-hoc route×seed for FA-HPPO seeds)
    fa = {(int(r["seed"]), r["route_id"]): r for r in rows if r["method"] == METHOD_CODE}
    paired = {}
    for m in METHODS[1:]:
        base = {r["route_id"]: r for r in rows if r["method"] == m}
        win = tie = loss = 0
        for (seed, rid), a in fa.items():
            b = base[rid]
            fa_ok, b_ok = bool(a["feasible"]), bool(b["feasible"])
            if fa_ok and not b_ok:
                win += 1
            elif fa_ok == b_ok:
                tie += 1
            else:
                loss += 1
        paired[LABEL[m]] = {
            "fa_hppo_only_feasible": win,
            "baseline_only_feasible": loss,
            "tie": tie,
            "n": win + tie + loss,
            "analysis": "post-hoc descriptive counts on locked TEST",
        }

    def stratum(key: str) -> dict:
        out = {}
        for val in sorted({r.get(key) for r in rows if r["method"] == METHOD_CODE}):
            rates = []
            for seed in SEEDS:
                sub = [
                    r
                    for r in rows
                    if r["method"] == METHOD_CODE and r.get("seed") == seed and r.get(key) == val
                ]
                rates.append(feas_rate(sub))
            mu, lo, hi = _t_ci(rates)
            out[str(val)] = {"mean": mu, "sd": _mean_sd(rates)[1], "t95": [lo, hi], "per_seed": rates}
        return out

    charge = {}
    for val in ("charging_required", "no_charge_required"):
        rates = []
        for seed in SEEDS:
            sub = [
                r
                for r in rows
                if r["method"] == METHOD_CODE and r.get("seed") == seed and r.get("charge_class") == val
            ]
            rates.append(feas_rate(sub))
        mu, lo, hi = _t_ci(rates)
        charge[val] = {"mean": mu, "sd": _mean_sd(rates)[1], "t95": [lo, hi], "per_seed": rates}

    # heatmap cells
    heat = {}
    for layout in ("C", "R", "RC"):
        for length in ("short", "medium", "long"):
            rates = []
            for seed in SEEDS:
                sub = [
                    r
                    for r in rows
                    if r["method"] == METHOD_CODE
                    and r.get("seed") == seed
                    and r.get("layout") == layout
                    and r.get("length_bin") == length
                ]
                rates.append(feas_rate(sub))
            heat[f"{layout}|{length}"] = float(np.mean(rates))

    payload = {
        "benchmark_wording": "fresh independently generated held-out SynthCharge TEST",
        "strata": "layout × frozen-route-length bin",
        "n_routes": 180,
        "n_charging_required": 144,
        "n_no_charge_required": 36,
        "methods": main,
        "paired_feasibility_vs_baselines": paired,
        "by_layout": stratum("layout"),
        "by_length_bin": stratum("length_bin"),
        "by_charge_class": charge,
        "heatmap_layout_x_length": heat,
        "prior_version_comparisons": False,
        "test_rerun": False,
    }
    _dump_json(STAT / "main_summary.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_tables(rows: list[dict], stats: dict) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    # Table 1
    t1 = []
    for m in METHODS:
        info = stats["methods"][LABEL[m]]
        t1.append(
            {
                "method": LABEL[m],
                "feasibility": info["feasibility_mean"],
                "feasibility_sd": info.get("feasibility_sd"),
                "completion_all": info["completion_mean"],
                "completion_sd": info.get("completion_sd"),
                "runtime_s": info["runtime_mean"],
            }
        )
    _write_csv(TAB / "table01_main_test.csv", t1)
    lines = [
        "# Table 1 — V4 TEST main results",
        "",
        "Fresh independently generated held-out SynthCharge TEST (180 routes).",
        "",
        "| Method | Feasibility | Failure-retaining completion | Runtime (s) |",
        "|---|---:|---:|---:|",
    ]
    for r in t1:
        if r["feasibility_sd"] is not None:
            feas = f"{100*r['feasibility']:.2f}% ± {100*r['feasibility_sd']:.2f}"
            comp = f"{r['completion_all']:.3f} ± {r['completion_sd']:.3f}"
        else:
            feas = f"{100*r['feasibility']:.2f}%"
            comp = f"{r['completion_all']:.3f}"
        lines.append(f"| {r['method']} | {feas} | {comp} | {r['runtime_s']:.4f} |")
    (TAB / "table01_main_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table 2 charging-required
    t2 = []
    for m in METHODS:
        if m == METHOD_CODE:
            info = stats["by_charge_class"]["charging_required"]
            t2.append({"method": LABEL[m], "subset": "charging_required", "n": 144, "feasibility": info["mean"], "sd": info["sd"]})
            info2 = stats["by_charge_class"]["no_charge_required"]
            t2.append({"method": LABEL[m], "subset": "no_charge_required", "n": 36, "feasibility": info2["mean"], "sd": info2["sd"]})
        else:
            for subset, n in (("charging_required", 144), ("no_charge_required", 36)):
                sub = [r for r in rows if r["method"] == m and r.get("charge_class") == subset]
                t2.append({"method": LABEL[m], "subset": subset, "n": n, "feasibility": feas_rate(sub), "sd": None})
    _write_csv(TAB / "table02_charging_required.csv", t2)
    md = [
        "# Table 2 — Charging-required / no-charge subsets",
        "",
        "| Method | Subset | n | Feasibility |",
        "|---|---|---:|---:|",
    ]
    for r in t2:
        if r["sd"] is not None:
            feas = f"{100*r['feasibility']:.2f}% ± {100*r['sd']:.2f}"
        else:
            feas = f"{100*r['feasibility']:.2f}%"
        md.append(f"| {r['method']} | {r['subset']} | {r['n']} | {feas} |")
    (TAB / "table02_charging_required.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Table 3 regime
    t3 = []
    for layout in ("C", "R", "RC"):
        for length in ("short", "medium", "long"):
            t3.append(
                {
                    "layout": layout,
                    "length_bin": length,
                    "feasibility": stats["heatmap_layout_x_length"][f"{layout}|{length}"],
                }
            )
    _write_csv(TAB / "table03_regime_robustness.csv", t3)
    md = [
        "# Table 3 — Regime robustness (FA-HPPO)",
        "",
        "Balanced strata: layout × frozen-route-length bin (not customer-scale cells).",
        "",
        "| Layout | Length bin | Feasibility |",
        "|---|---|---:|",
    ]
    for r in t3:
        md.append(f"| {r['layout']} | {r['length_bin']} | {100*r['feasibility']:.2f}% |")
    for layout, info in stats["by_layout"].items():
        md.append(f"| {layout} | *(all lengths)* | {100*info['mean']:.2f}% |")
    for length, info in stats["by_length_bin"].items():
        md.append(f"| *(all layouts)* | {length} | {100*info['mean']:.2f}% |")
    (TAB / "table03_regime_robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Table 4 reward ablation
    abl = _load_json(ABLATION)["variants"]
    t4 = []
    for code, label in REWARD_ORDER:
        v = abl[code]
        # best update mean from SUMMARY if present
        t4.append(
            {
                "reward": label,
                "val_feasibility": v["feas_mean"],
                "val_feasibility_sd": v["feas_sd"],
                "val_completion": v["comp_mean"],
                "val_completion_sd": v["comp_sd"],
                "mean_best_update": v.get("best_update_mean"),
                "selected": code == "V4_BASE_NO_L_FAIL",
            }
        )
    _write_csv(TAB / "table04_reward_ablation.csv", t4)
    md = [
        "# Table 4 — Reward development ablation (VALIDATION)",
        "",
        "TRAIN/VAL development evidence only. TEST not used for selection.",
        "",
        "| Reward | VAL feasibility | VAL completion | Mean best update | Selected |",
        "|---|---:|---:|---:|---|",
    ]
    for r in t4:
        md.append(
            f"| {r['reward']} | {100*r['val_feasibility']:.1f}% ± {100*r['val_feasibility_sd']:.1f} | "
            f"{r['val_completion']:.3f} ± {r['val_completion_sd']:.3f} | {r['mean_best_update']:.0f} | "
            f"{'yes' if r['selected'] else ''} |"
        )
    (TAB / "table04_reward_ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Table 5 operational
    ops_fields = [
        ("route_completion_time", "completion_time"),
        ("total_charging_time", "charging_time"),
        ("total_energy_charged", "energy_charged"),
        ("n_station_visits", "n_station_visits"),
        ("terminal_soc", "terminal_soc"),
        ("total_distance", "total_distance"),
        ("runtime_s", "runtime_s"),
    ]
    t5 = []
    for m in METHODS:
        feas = [r for r in rows if r["method"] == m and r.get("feasible")]
        # for FA-HPPO pool all seeds' feasible rows
        for field, name in ops_fields:
            vals = [float(r[field]) for r in feas]
            mu, sd = _mean_sd(vals)
            t5.append(
                {
                    "conditioning": "method_feasible_routes_only",
                    "survivor_bias_warning": True,
                    "method": LABEL[m],
                    "metric": name,
                    "n": len(vals),
                    "mean": mu,
                    "sd": sd,
                }
            )
    # matched common-feasible FA-HPPO vs each baseline (seed-averaged route pairs: use seed 42..46 mean of matched)
    fa_idx = {(int(r["seed"]), r["route_id"]): r for r in rows if r["method"] == METHOD_CODE}
    for m in METHODS[1:]:
        base = {r["route_id"]: r for r in rows if r["method"] == m}
        for field, name in ops_fields:
            deltas = []
            for (seed, rid), a in fa_idx.items():
                b = base[rid]
                if a.get("feasible") and b.get("feasible"):
                    deltas.append(float(a[field]) - float(b[field]))
            mu, sd = _mean_sd(deltas)
            t5.append(
                {
                    "conditioning": "matched_common_feasible",
                    "survivor_bias_warning": False,
                    "method": f"FA-HPPO minus {LABEL[m]}",
                    "metric": name,
                    "n": len(deltas),
                    "mean": mu,
                    "sd": sd,
                }
            )
    _write_csv(TAB / "table05_operational_metrics.csv", t5)
    md = [
        "# Table 5 — Operational metrics",
        "",
        "> Method-feasible rows carry a **survivor-bias warning** and must not be read as",
        "> cross-method efficiency gains. Prefer matched common-feasible deltas for comparisons.",
        "",
        "| Conditioning | Method | Metric | n | Mean | SD |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in t5:
        md.append(
            f"| {r['conditioning']} | {r['method']} | {r['metric']} | {r['n']} | "
            f"{r['mean']:.6g} | {r['sd']:.6g} |"
        )
    (TAB / "table05_operational_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig01_usecase() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    # schematic geometry
    depot = np.array([0.1, 0.5])
    customers = np.array([[0.28, 0.72], [0.45, 0.35], [0.62, 0.68], [0.78, 0.40], [0.90, 0.60]])
    stations = np.array([[0.38, 0.55], [0.70, 0.50]])
    ax.plot([depot[0], *customers[:, 0], depot[0]], [depot[1], *customers[:, 1], depot[1]], "-", color="#0072B2", lw=2.0, zorder=1)
    ax.scatter(*depot, s=160, c="#000000", marker="s", zorder=3, label="Depot")
    ax.scatter(customers[:, 0], customers[:, 1], s=90, c="#0072B2", zorder=3, label="Fixed customers")
    ax.scatter(stations[:, 0], stations[:, 1], s=120, c="#D55E00", marker="^", zorder=3, label="Charging stations")
    for i, (x, y) in enumerate(customers, 1):
        ax.text(x, y + 0.05, f"C{i}", ha="center", fontsize=8)
    for i, (x, y) in enumerate(stations, 1):
        ax.text(x, y - 0.06, f"S{i}", ha="center", fontsize=8, color="#D55E00")
    ax.annotate("learned charge insertion", xy=stations[0], xytext=(0.35, 0.20),
                arrowprops=dict(arrowstyle="->", color="#D55E00"), fontsize=8, color="#D55E00")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Fixed customer order; charging decisions are learned", loc="left")
    _save(fig, "fig01_usecase_route", split="methodology", sources=[])


def fig02_method() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, "Fixed route\nstate"),
        (0.18, "Feasibility\nshield"),
        (0.34, "Features"),
        (0.48, "Hybrid\nPPO"),
        (0.62, "CONTINUE /\nstation"),
        (0.76, "Amount u\n+ SOC map"),
        (0.90, "Simulator\ntransition"),
    ]
    for x, text in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.38), 0.12, 0.28, boxstyle="round,pad=0.01", facecolor="#E8F1F8", edgecolor="#0072B2", lw=1.0))
        ax.text(x + 0.06, 0.52, text, ha="center", va="center", fontsize=7.5)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 0.12
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1, 0.52), xytext=(x0, 0.52), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
    ax.text(
        0.5,
        0.18,
        r"Reward: $r_t=-\Delta t/C_{\mathrm{train}}$,  $r_{\mathrm{fail}}=-(H-t)/C_{\mathrm{train}}$  ($C_{\mathrm{train}}=10$)",
        ha="center",
        fontsize=8,
    )
    ax.text(
        0.5,
        0.08,
        r"$SOC_{\mathrm{target}}=SOC_{\mathrm{lower}}+u(SOC_{\mathrm{upper}}-SOC_{\mathrm{lower}})$",
        ha="center",
        fontsize=8,
    )
    _save(fig, "fig02_method_schematic", split="methodology", sources=["results/v4_reward/FINAL_REWARD_FREEZE.json"])


def fig03_envelope() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.linspace(0, 1, 200)
    lower = 0.25 + 0.05 * np.sin(2 * np.pi * x)
    upper = 0.85 - 0.08 * np.cos(2 * np.pi * x)
    u = 0.55
    target = lower + u * (upper - lower)
    arrival = 0.42 + 0.03 * np.sin(4 * np.pi * x)
    ax.fill_between(x, lower, upper, color="#0072B2", alpha=0.15, label="Feasible envelope")
    ax.plot(x, lower, color="#009E73", lw=1.8, label=r"$SOC_{\mathrm{lower}}$")
    ax.plot(x, upper, color="#D55E00", lw=1.8, label=r"$SOC_{\mathrm{upper}}$")
    ax.plot(x, target, color="#0072B2", lw=2.0, label=r"$SOC_{\mathrm{target}}$")
    ax.plot(x, arrival, color="#666666", ls="--", lw=1.2, label="Arrival SOC")
    ax.set_xlabel("Normalized progress along visit")
    ax.set_ylabel("SOC")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _save(fig, "fig03_soc_envelope", split="methodology", sources=[])


def fig04_soc_traj() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    # Illustrative TRAIN/VAL-style trajectory (not TEST)
    t = np.array([0, 1, 2, 2.4, 3.2, 4.0, 4.5, 5.2, 6.0, 6.8])
    soc = np.array([0.90, 0.72, 0.55, 0.82, 0.60, 0.42, 0.78, 0.55, 0.38, 0.50])
    lower = np.linspace(0.20, 0.25, len(t))
    upper = np.linspace(0.95, 0.90, len(t))
    ax.fill_between(t, lower, upper, color="#0072B2", alpha=0.12)
    ax.plot(t, soc, "-o", color="#0072B2", lw=1.8, ms=5)
    for ti, label in ((0, "Depot"), (1, "C1"), (2, "C2"), (2.4, "S1"), (4.5, "S2"), (6.8, "Depot")):
        ax.axvline(ti, color="#DDDDDD", lw=0.8, zorder=0)
        ax.text(ti, 0.05, label, ha="center", fontsize=7, color="#444444")
    ax.set_xlabel("Time")
    ax.set_ylabel("SOC")
    ax.set_ylim(0, 1)
    ax.set_title("Illustrative TRAIN/VAL SOC trajectory (not TEST)", loc="left", fontsize=9)
    _save(fig, "fig04_illustrative_soc", split="TRAIN/VAL", sources=[])


def _bar_methods(ax, values: dict[str, float], yerr: dict[str, float] | None, ylabel: str, ylim=None) -> None:
    xs = np.arange(len(METHODS))
    means = [values[m] for m in METHODS]
    colors = [COLOR[m] for m in METHODS]
    ax.bar(xs, means, color=colors, width=0.72, edgecolor="white", linewidth=0.4)
    if yerr:
        errs = [yerr.get(m) or 0.0 for m in METHODS]
        ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor="#333333", capsize=3, elinewidth=1.0)
    ax.set_xticks(xs, [SHORT[m] for m in METHODS], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def fig05_feas(rows: list[dict], stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    values = {m: 100 * stats["methods"][LABEL[m]]["feasibility_mean"] for m in METHODS}
    yerr = {}
    for m in METHODS:
        info = stats["methods"][LABEL[m]]
        if info.get("feasibility_t95"):
            lo, hi = info["feasibility_t95"][0], info["feasibility_t95"][1]
            yerr[m] = 100 * max(info["feasibility_mean"] - lo, hi - info["feasibility_mean"])
        else:
            yerr[m] = 0.0
    _bar_methods(ax, values, yerr, "TEST feasibility (%)", ylim=(0, 105))
    _save(fig, "fig05_main_feasibility", split="TEST", sources=["results/v4_test/raw/synthcharge_v4_test.jsonl"])


def fig06_comp(stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    values = {m: stats["methods"][LABEL[m]]["completion_mean"] for m in METHODS}
    yerr = {}
    for m in METHODS:
        info = stats["methods"][LABEL[m]]
        if info.get("completion_t95"):
            lo, hi = info["completion_t95"][0], info["completion_t95"][1]
            yerr[m] = max(info["completion_mean"] - lo, hi - info["completion_mean"])
        else:
            yerr[m] = 0.0
    _bar_methods(ax, values, yerr, "Failure-retaining completion", ylim=(0, None))
    _save(fig, "fig06_main_completion", split="TEST", sources=["results/v4_test/raw/synthcharge_v4_test.jsonl"])


def fig07_charging(rows: list[dict], stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    values = {}
    yerr = {}
    for m in METHODS:
        if m == METHOD_CODE:
            info = stats["by_charge_class"]["charging_required"]
            values[m] = 100 * info["mean"]
            lo, hi = info["t95"]
            yerr[m] = 100 * max(info["mean"] - lo, hi - info["mean"])
        else:
            sub = [r for r in rows if r["method"] == m and r.get("charge_class") == "charging_required"]
            values[m] = 100 * feas_rate(sub)
            yerr[m] = 0.0
    _bar_methods(ax, values, yerr, "Charging-required feasibility (%)", ylim=(0, 105))
    _save(fig, "fig07_charging_required", split="TEST", sources=["results/v4_test/raw/synthcharge_v4_test.jsonl"])


def fig08_ecdf(rows: list[dict]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for m in METHODS:
        if m == METHOD_CODE:
            # pool seed means? Use all route×seed failure-retaining completions
            vals = np.sort([float(r["completion_time_all_routes"]) for r in rows if r["method"] == m])
        else:
            vals = np.sort([float(r["completion_time_all_routes"]) for r in rows if r["method"] == m])
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, color=COLOR[m], lw=1.6, label=SHORT[m])
    ax.set_xlabel("Failure-retaining completion")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig08_performance_ecdf", split="TEST", sources=["results/v4_test/raw/synthcharge_v4_test.jsonl"])


def fig09_layout(rows: list[dict], stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    layouts = ["C", "R", "RC"]
    x = np.arange(len(layouts))
    width = 0.18
    for i, m in enumerate(METHODS):
        means = []
        errs = []
        for layout in layouts:
            if m == METHOD_CODE:
                info = stats["by_layout"][layout]
                means.append(100 * info["mean"])
                lo, hi = info["t95"]
                errs.append(100 * max(info["mean"] - lo, hi - info["mean"]))
            else:
                sub = [r for r in rows if r["method"] == m and r.get("layout") == layout]
                means.append(100 * feas_rate(sub))
                errs.append(0.0)
        ax.bar(x + (i - 1.5) * width, means, width=width, color=COLOR[m], label=SHORT[m],
               yerr=errs if m == METHOD_CODE else None, capsize=2, error_kw={"elinewidth": 0.8})
    ax.set_xticks(x, layouts)
    ax.set_ylabel("TEST feasibility (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save(fig, "fig09_by_layout", split="TEST", sources=["results/v4_test/raw/synthcharge_v4_test.jsonl"])


def fig10_length(stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    bins = ["short", "medium", "long"]
    means = [100 * stats["by_length_bin"][b]["mean"] for b in bins]
    errs = []
    for b in bins:
        info = stats["by_length_bin"][b]
        lo, hi = info["t95"]
        errs.append(100 * max(info["mean"] - lo, hi - info["mean"]))
    ax.bar(np.arange(3), means, color="#0072B2", yerr=errs, capsize=3, width=0.65)
    ax.set_xticks(np.arange(3), bins)
    ax.set_ylabel("FA-HPPO TEST feasibility (%)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Frozen-route-length bin")
    _save(fig, "fig10_by_length", split="TEST", sources=["results/v4_test/raw/synthcharge_v4_test.jsonl"])


def fig11_heatmap(stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    layouts = ["C", "R", "RC"]
    lengths = ["short", "medium", "long"]
    mat = np.array([[100 * stats["heatmap_layout_x_length"][f"{L}|{b}"] for b in lengths] for L in layouts])
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(3), lengths)
    ax.set_yticks(range(3), layouts)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Feasibility (%)")
    ax.set_xlabel("Route-length bin")
    ax.set_ylabel("Layout")
    _save(fig, "fig11_difficulty_heatmap", split="TEST", sources=["results/v4_test/raw/synthcharge_v4_test.jsonl"])


def fig12_ablation() -> None:
    _style()
    abl = _load_json(ABLATION)["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    xs = np.arange(len(REWARD_ORDER))
    feas = [100 * abl[c]["feas_mean"] for c, _ in REWARD_ORDER]
    fsd = [100 * abl[c]["feas_sd"] for c, _ in REWARD_ORDER]
    comp = [abl[c]["comp_mean"] for c, _ in REWARD_ORDER]
    csd = [abl[c]["comp_sd"] for c, _ in REWARD_ORDER]
    colors = [REWARD_COLOR[c] for c, _ in REWARD_ORDER]
    labels = [lab for _, lab in REWARD_ORDER]
    for ax, vals, err, ylabel, title in (
        (axes[0], feas, fsd, "VALIDATION feasibility (%)", "(a) Feasibility"),
        (axes[1], comp, csd, "VALIDATION completion", "(b) Failure-retaining completion"),
    ):
        ax.scatter(xs, vals, c=colors, s=40, zorder=3, edgecolors="white", linewidths=0.4)
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor="#333333", capsize=3, zorder=2)
        for i, (c, _) in enumerate(REWARD_ORDER):
            marker = "D" if c == "V4_BASE_NO_L_FAIL" else "o"
            ax.plot(i, vals[i], marker, color=colors[i], markersize=7, markeredgecolor="black", markeredgewidth=0.4, zorder=4)
        ax.set_xticks(xs, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        if ax is axes[0]:
            ax.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, "fig12_reward_ablation_val", split="VAL", sources=["results/v4_reward/SUMMARY.json"])


def _load_auth_curves() -> dict[int, list[dict]]:
    out = {}
    for seed in SEEDS:
        path = AUTH / f"seed_{seed}" / "curves.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[seed] = rows
    return out


def _common_series(curves: dict[int, list[dict]], key: str):
    per = []
    for rows in curves.values():
        by_u = {int(r["update"]): r.get(key) for r in rows if r.get(key) is not None}
        if by_u:
            per.append(by_u)
    if not per:
        return np.array([]), np.array([]), np.array([]), np.array([])
    common = set(per[0])
    for d in per[1:]:
        common &= set(d)
    xs = sorted(common)
    if not xs:
        return np.array([]), np.array([]), np.array([]), np.array([])
    mat = np.array([[float(d[u]) for u in xs] for d in per], dtype=float)
    mu = mat.mean(axis=0)
    se = mat.std(axis=0, ddof=1) / math.sqrt(mat.shape[0])
    half = T_CRIT_5 * se
    return np.asarray(xs), mu, mu - half, mu + half


def fig13_reward() -> None:
    curves = _load_auth_curves()
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs, mu, lo, hi = _common_series(curves, "objective_episode_return_mean")
    if xs.size == 0:
        xs, mu, lo, hi = _common_series(curves, "base_episode_return_mean")
        mu, lo, hi = mu * 10.0, lo * 10.0, hi * 10.0
    ax.plot(xs, mu, color="#0072B2", lw=1.8)
    ax.fill_between(xs, lo, hi, color="#0072B2", alpha=0.18)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Mean objective episode return (−T / −H)")
    _save(
        fig,
        "fig13_reward_evolution",
        split="TRAIN",
        sources=[f"results/v4_reward/final_authoritative/V4_BASE_NO_L_FAIL/seed_{s}/curves.jsonl" for s in SEEDS],
    )


def fig14_loss() -> None:
    curves = _load_auth_curves()
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for ax, key, ylabel, title, logy in (
        (axes[0], "policy_loss", r"Policy loss $L_\pi$", "(a) Policy loss", False),
        (axes[1], "value_loss", r"Value loss $L_V$", "(b) Value loss", True),
    ):
        xs, mu, lo, hi = _common_series(curves, key)
        ax.plot(xs, mu, color="#0072B2", lw=1.8)
        if logy:
            ax.fill_between(xs, np.maximum(lo, 1e-12), np.maximum(hi, 1e-12), color="#0072B2", alpha=0.18)
            if np.all(mu > 0):
                ax.set_yscale("log")
        else:
            ax.fill_between(xs, lo, hi, color="#0072B2", alpha=0.18)
        ax.set_xlabel("PPO update")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    fig.tight_layout()
    _save(
        fig,
        "fig14_loss_evolution",
        split="TRAIN",
        sources=[f"results/v4_reward/final_authoritative/V4_BASE_NO_L_FAIL/seed_{s}/curves.jsonl" for s in SEEDS],
    )


def fig15_val() -> None:
    curves = _load_auth_curves()
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs, mu, lo, hi = _common_series(curves, "val_parent_balanced_feasibility")
    ax.plot(xs, 100 * mu, color="#0072B2", lw=1.8, label="VALIDATION (parent-balanced)")
    ax.fill_between(xs, 100 * lo, 100 * hi, color="#0072B2", alpha=0.18)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("VALIDATION feasibility (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right")
    _save(
        fig,
        "fig15_validation_learning",
        split="VAL",
        sources=[f"results/v4_reward/final_authoritative/V4_BASE_NO_L_FAIL/seed_{s}/curves.jsonl" for s in SEEDS],
    )


def figA1_grad() -> None:
    curves = _load_auth_curves()
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    xs, mu, lo, hi = _common_series(curves, "grad_norm_preclip")
    ax.plot(xs, mu, color="#D55E00", lw=1.8)
    ax.fill_between(xs, np.maximum(lo, 1e-12), np.maximum(hi, 1e-12), color="#D55E00", alpha=0.18)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Pre-clip gradient norm")
    if np.all(mu > 0):
        ax.set_yscale("log")
    _save(
        fig,
        "figA01_gradient_norm",
        split="TRAIN",
        sources=[f"results/v4_reward/final_authoritative/V4_BASE_NO_L_FAIL/seed_{s}/curves.jsonl" for s in SEEDS],
    )


def figA3_seed_test(stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    feas = stats["methods"]["FA-HPPO"]["feasibility_per_seed"]
    xs = list(range(len(SEEDS)))
    vals = [100 * feas[str(s)] for s in SEEDS]
    ax.bar(xs, vals, color="#0072B2", width=0.65)
    ax.axhline(100 * stats["methods"]["FA-HPPO"]["feasibility_mean"], color="#333333", ls="--", lw=1.0)
    ax.set_xticks(xs, [str(s) for s in SEEDS])
    ax.set_xlabel("Seed")
    ax.set_ylabel("TEST feasibility (%)")
    ax.set_ylim(0, 105)
    _save(fig, "figA03_per_seed_test_feasibility", split="TEST", sources=["results/v4_test/raw/synthcharge_v4_test.jsonl"])


def figA4_failures(rows: list[dict]) -> None:
    _style()
    reasons = defaultdict(int)
    for r in rows:
        if r["method"] == METHOD_CODE and not r.get("feasible"):
            reasons[str(r.get("reason") or "unknown")] += 1
    # average counts over seeds roughly by dividing? show total over 5×180
    items = sorted(reasons.items(), key=lambda kv: -kv[1])[:8]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    if not items:
        ax.text(0.5, 0.5, "No failures", ha="center")
    else:
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        ax.barh(range(len(labels)), vals, color="#D55E00")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Failure count (all seeds pooled)")
    fig.tight_layout()
    _save(fig, "figA04_failure_reasons", split="TEST", sources=["results/v4_test/raw/synthcharge_v4_test.jsonl"])


# ---------------------------------------------------------------------------
# Captions / README / Manifest
# ---------------------------------------------------------------------------


def write_captions() -> None:
    CAP.mkdir(parents=True, exist_ok=True)
    text = r"""# V4 figure captions (standalone)

All TEST figures use the fresh independently generated held-out SynthCharge TEST.
TRAIN/VAL figures are development evidence; TEST was not used for training or checkpoint selection.

**Fig. 1.** Fixed-route EV charging use case: the customer sequence is fixed while charging-station insertion and charge amount are learned.

**Fig. 2.** FA-HPPO control loop with feasibility-aware shielding, hybrid discrete/continuous decisions, SOC envelope mapping, and the normalized time-horizon reward.

**Fig. 3.** Feasibility-aware SOC envelope: continuation lower bound, optimistic time-aware upper bound, and target SOC parameterized by continuous amount \(u\).

**Fig. 4.** Illustrative TRAIN/VAL SOC trajectory with charging stops on a fixed customer route (methodology illustration; not TEST).

**Fig. 5.** Main TEST feasibility for FA-HPPO and three heuristic/planning baselines under the same feasibility-aware environment. FA-HPPO shows five-seed Student-t uncertainty; baselines are deterministic.

**Fig. 6.** Failure-retaining completion on all TEST routes (infeasible episodes retain the route horizon \(H\)). Lower is better.

**Fig. 7.** Feasibility on the charging-required TEST subset (144 routes). FA-HPPO reaches 100% on the complementary 36 no-charge-required routes.

**Fig. 8.** Empirical CDF of failure-retaining completion for FA-HPPO and baselines on all TEST routes.

**Fig. 9.** TEST feasibility by layout (`C`, `R`, `RC`), exposing regime dependence.

**Fig. 10.** FA-HPPO TEST feasibility by balanced frozen-route-length bin (`short`, `medium`, `long`).

**Fig. 11.** Layout × route-length feasibility heatmap for FA-HPPO on the locked TEST (balanced 3×3 strata).

**Fig. 12.** VALIDATION reward-development ablation. Potential-based shaping did not improve validation feasibility; the simpler time-horizon reward was selected.

**Fig. 13.** TRAIN objective episode return (\(G=-T\) on success, \(G=-H\) on failure), mean across seeds with Student-t 95% interval on the common update support.

**Fig. 14.** TRAIN PPO policy and value losses (diagnostics; decreasing loss is not claimed as convergence).

**Fig. 15.** VALIDATION parent-balanced feasibility versus PPO update.

**Fig. A1.** Pre-clip gradient norm during authoritative training.

**Fig. A3.** Per-seed FA-HPPO feasibility on the locked TEST.

**Fig. A4.** Pooled FA-HPPO failure-reason counts on the locked TEST.
"""
    (CAP / "CAPTIONS.md").write_text(text, encoding="utf-8")


def write_readme(stats: dict) -> None:
    mu = stats["methods"]["FA-HPPO"]["feasibility_mean"]
    text = f"""# V4 paper-facing package (standalone)

This directory contains **only** V4 manuscript-facing evidence.

## Primary TEST result

FA-HPPO achieves **{100*mu:.2f}%** mean feasibility (5 seeds) on the fresh
independently generated held-out SynthCharge TEST and substantially outperforms
One-step lookahead, Greedy full charge, and Greedy minimum charge under the same
feasibility-aware environment.

## Reproduce displays (no TEST rerun)

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

## Benchmark strata

Balanced quotas are **layout × frozen-route-length bin** (20 routes/cell).
See `docs/V4_BENCHMARK_WORDING.md`.

## Claims / limitations

See `paper/V4_CLAIMS.md`.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def write_manifest(stats: dict) -> dict:
    reward = _load_json(REWARD_FREEZE)
    lock = _load_json(LOCK)
    consumed = _load_json(CONSUMED)
    fig_manifest = {"figures": FIG_ENTRIES}
    _dump_json(MAN / "FIGURE_MANIFEST.json", fig_manifest)
    tables = sorted(TAB.glob("*.*"))
    payload = {
        "package": "results_v4_paper",
        "standalone": True,
        "prior_version_comparisons": False,
        "reward": reward.get("equations"),
        "c_train": reward.get("c_train"),
        "human_name": reward.get("human_name"),
        "training_git_sha": reward.get("training_git_sha"),
        "checkpoint_hashes": {s: v["checkpoint_sha256"] for s, v in reward.get("seeds", {}).items()},
        "test": {
            "n_routes": lock.get("n_routes"),
            "seed_start": lock.get("seed_start"),
            "raw_sha256_lf": consumed.get("raw_sha256_lf"),
            "test_lock_sha256": _lf_sha(LOCK),
            "evaluation_consumed_sha256": _sha(CONSUMED),
            "n_charging_required": 144,
            "n_no_charge_required": 36,
            "n_per_layout": 60,
            "n_per_length_bin": 60,
        },
        "main_test_feasibility_mean": stats["methods"]["FA-HPPO"]["feasibility_mean"],
        "main_test_completion_mean": stats["methods"]["FA-HPPO"]["completion_mean"],
        "charging_required_feasibility_mean": stats["by_charge_class"]["charging_required"]["mean"],
        "figures": FIG_ENTRIES,
        "tables": [
            {
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha(p),
            }
            for p in tables
        ],
        "frozen_source_hashes": {
            "raw": _lf_sha(RAW),
            "test_lock": _lf_sha(LOCK),
            "evaluation_consumed": _sha(CONSUMED),
            "reward_freeze": _sha(REWARD_FREEZE),
            "ablation_summary": _sha(ABLATION),
        },
        "no_test_rerun": True,
        "no_checkpoint_reselection": True,
        "no_reward_redesign": True,
        "no_post_test_training": True,
        "builder": "scripts/paper/build_v4_results_paper.py",
    }
    _dump_json(OUT / "MANIFEST.json", payload)
    return payload


def verify(manifest: dict) -> None:
    # frozen hashes unchanged
    checks = {
        "raw": (_lf_sha(RAW), manifest["frozen_source_hashes"]["raw"]),
        "test_lock": (_lf_sha(LOCK), manifest["frozen_source_hashes"]["test_lock"]),
        "evaluation_consumed": (_sha(CONSUMED), manifest["frozen_source_hashes"]["evaluation_consumed"]),
        "reward_freeze": (_sha(REWARD_FREEZE), manifest["frozen_source_hashes"]["reward_freeze"]),
    }
    for name, (got, exp) in checks.items():
        if got != exp:
            raise SystemExit(f"--verify failed: {name} hash changed")
    if abs(manifest["main_test_feasibility_mean"] - 0.9455555555555556) > 1e-9:
        raise SystemExit("--verify failed: main feasibility mismatch")
    if manifest["test"]["n_routes"] != 180:
        raise SystemExit("--verify failed: n_routes")
    # no prior-version method labels in paper outputs
    forbidden = ("V3_FA_HPPO", "V3-HPPO", "prior-version", "external-domain")
    for path in list(FIG.glob("*.png")) + list(TAB.glob("*.md")) + [CAP / "CAPTIONS.md", OUT / "README.md"]:
        if not path.is_file() or path.suffix == ".png":
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise SystemExit(f"--verify failed: forbidden token {token!r} in {path}")
    for entry in manifest["figures"]:
        path = ROOT / entry["path"]
        if _sha(path) != entry["figure_sha256"]:
            raise SystemExit(f"--verify failed: figure hash drift {entry['filename']}")
    print("VERIFY OK", json.dumps({"n_figures": len(manifest["figures"]), "feas": manifest["main_test_feasibility_mean"]}, indent=2))


def main() -> None:
    verify_only = "--verify" in sys.argv
    if verify_only:
        if not (OUT / "MANIFEST.json").is_file():
            raise SystemExit("run builder before --verify")
        verify(_load_json(OUT / "MANIFEST.json"))
        return

    for d in (OUT, FIG, TAB, STAT, CAP, MAN):
        d.mkdir(parents=True, exist_ok=True)
    FIG_ENTRIES.clear()

    rows = load_rows()
    if len(rows) != 180 * 4:  # FA-HPPO×5 seeds + 3 baselines? Wait FA is 900 + 540 = 1440
        # V4_FA_HPPO: 180*5=900; baselines: 180*3=540; total 1440
        expected = 180 * 5 + 180 * 3
        if len(rows) != expected:
            raise SystemExit(f"unexpected filtered row count {len(rows)} != {expected}")

    stats = build_statistics(rows)
    build_tables(rows, stats)

    fig01_usecase()
    fig02_method()
    fig03_envelope()
    fig04_soc_traj()
    fig05_feas(rows, stats)
    fig06_comp(stats)
    fig07_charging(rows, stats)
    fig08_ecdf(rows)
    fig09_layout(rows, stats)
    fig10_length(stats)
    fig11_heatmap(stats)
    fig12_ablation()
    fig13_reward()
    fig14_loss()
    fig15_val()
    figA1_grad()
    figA3_seed_test(stats)
    figA4_failures(rows)

    write_captions()
    write_readme(stats)
    manifest = write_manifest(stats)
    print(json.dumps({
        "n_figures": len(FIG_ENTRIES),
        "feasibility": stats["methods"]["FA-HPPO"]["feasibility_mean"],
        "completion": stats["methods"]["FA-HPPO"]["completion_mean"],
        "charging_required": stats["by_charge_class"]["charging_required"]["mean"],
        "out": str(OUT.relative_to(ROOT)).replace("\\", "/"),
    }, indent=2))
    verify(manifest)


if __name__ == "__main__":
    main()
