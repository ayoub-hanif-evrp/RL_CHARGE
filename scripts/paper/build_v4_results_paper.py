"""Build standalone V4 paper package (results/v4/paper/) from frozen evidence only.

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
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v4" / "paper"
FIG = OUT / "figures"
TAB = OUT / "tables"
STAT = OUT / "statistics"
CAP = OUT / "captions"
MAN = OUT / "manifests"

RAW = ROOT / "results" / "v4" / "test" / "raw" / "synthcharge_v4_test.jsonl"
LOCK = ROOT / "results" / "v4" / "test" / "TEST_LOCK.json"
CONSUMED = ROOT / "results" / "v4" / "test" / "EVALUATION_CONSUMED.json"
CKPT_FREEZE = ROOT / "results" / "v4" / "test" / "CHECKPOINT_FREEZE.json"
PROTOCOL = ROOT / "results" / "v4" / "test" / "PAPER_PROTOCOL.json"
REWARD_FREEZE = ROOT / "results" / "v4" / "reward_development" / "FINAL_REWARD_FREEZE.json"
ABLATION = ROOT / "results" / "v4" / "reward_development" / "SUMMARY.json"
AUTH = ROOT / "results" / "v4" / "reward_development" / "final_authoritative" / "V4_BASE_NO_L_FAIL"

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
    ("V3_TIME", "Progress"),
    ("V4_BASE", "Normalized"),
    ("V4_PBRS", "PBRS"),
    ("V4_BASE_NO_L_FAIL", "Time-horizon"),
)
REWARD_FULL = {
    "V3_TIME": "Time + progress failure",
    "V4_BASE": "Normalized equivalent",
    "V4_PBRS": "Potential-shaped",
    "V4_BASE_NO_L_FAIL": "Time-horizon (selected)",
}
TRAJ = OUT / "data" / "illustrative_val_trajectory.json"
EXPECTED_FIGS = (
    "fig01_usecase_route.png",
    "fig02_method_schematic.png",
    "fig03_soc_envelope.png",
    "fig04_illustrative_soc.png",
    "fig05_main_feasibility.png",
    "fig06_main_completion.png",
    "fig07_charging_required.png",
    "fig08_performance_ecdf.png",
    "fig09_by_layout.png",
    "fig10_by_length.png",
    "fig11_difficulty_heatmap.png",
    "fig12_reward_ablation_val.png",
    "fig13_reward_evolution.png",
    "fig14_loss_evolution.png",
    "fig15_validation_learning.png",
    "figA01_gradient_norm.png",
    "figA03_per_seed_test_feasibility.png",
    "figA04_failure_reasons.png",
)
FAILURE_LABEL = {
    "TIME_WINDOW_VIOLATION": "Time-window violation",
    "SOC_VIOLATION": "SOC violation",
    "BATTERY_VIOLATION": "Battery violation",
    "CAPACITY_VIOLATION": "Capacity violation",
    "NO_FEASIBLE_ACTION": "No feasible action",
    "HORIZON_EXCEEDED": "Horizon exceeded",
    "UNKNOWN": "Unknown",
    "unknown": "Unknown",
}
REWARD_COLOR = {
    "V3_TIME": "#999999",
    "V4_BASE": "#56B4E9",
    "V4_PBRS": "#0072B2",
    "V4_BASE_NO_L_FAIL": "#E69F00",
}

# Pinned frozen evidence hashes (must not change in this polish pass)
PINNED_RAW_SHA256_LF = "431014b397eba9191b09312f83ccb7523641e353f3d37ae9c6d57bb2a31b00dc"
PINNED_TEST_LOCK_SHA256_LF = "c0055ba29beaa9b36590ef7e84cc9a60735f52adabaf7ca51dfdde5b89444d58"
PINNED_EVALUATION_CONSUMED_SHA256 = "3cdcef0bb339f63a21466d5dd26d27e8c0d518820ed402de21eec78a02bc54b7"
PINNED_REWARD_FREEZE_SHA256 = "c6a520e1a588445c880056777c8298de5fbbf1ae1cdfeac3865b104e036dc82f"
PINNED_MAIN_FEAS = 0.9455555555555556

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
    # Table 1 — FA-HPPO uses 95% Student-t CI across five seeds; baselines are deterministic
    t1 = []
    for m in METHODS:
        info = stats["methods"][LABEL[m]]
        row = {
            "method": LABEL[m],
            "feasibility": info["feasibility_mean"],
            "feasibility_ci_lo": info.get("feasibility_t95", [None, None])[0] if m == METHOD_CODE else None,
            "feasibility_ci_hi": info.get("feasibility_t95", [None, None])[1] if m == METHOD_CODE else None,
            "completion_all": info["completion_mean"],
            "completion_ci_lo": info.get("completion_t95", [None, None])[0] if m == METHOD_CODE else None,
            "completion_ci_hi": info.get("completion_t95", [None, None])[1] if m == METHOD_CODE else None,
            "runtime_s": info["runtime_mean"],
            "uncertainty": "student_t_95_five_seeds" if m == METHOD_CODE else "deterministic",
        }
        t1.append(row)
    _write_csv(TAB / "table01_main_test.csv", t1)
    lines = [
        "# Table 1 — V4 TEST main results",
        "",
        "Fresh independently generated held-out SynthCharge TEST (180 routes).",
        "FA-HPPO intervals are 95% Student-t confidence intervals across five independently trained seeds.",
        "Deterministic baselines have no uncertainty estimates.",
        "",
        "| Method | Feasibility | Failure-retaining completion | Runtime (s) |",
        "|---|---:|---:|---:|",
    ]
    for r in t1:
        if r["feasibility_ci_lo"] is not None:
            feas = f"{100*r['feasibility']:.2f}% [{100*r['feasibility_ci_lo']:.2f}, {100*r['feasibility_ci_hi']:.2f}]"
            comp = f"{r['completion_all']:.3f} [{r['completion_ci_lo']:.3f}, {r['completion_ci_hi']:.3f}]"
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
            lo, hi = info["t95"]
            t2.append(
                {
                    "method": LABEL[m],
                    "subset": "charging_required",
                    "n_routes": 144,
                    "feasibility": info["mean"],
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "uncertainty": "student_t_95_five_seeds",
                }
            )
            info2 = stats["by_charge_class"]["no_charge_required"]
            lo2, hi2 = info2["t95"]
            t2.append(
                {
                    "method": LABEL[m],
                    "subset": "no_charge_required",
                    "n_routes": 36,
                    "feasibility": info2["mean"],
                    "ci_lo": lo2,
                    "ci_hi": hi2,
                    "uncertainty": "student_t_95_five_seeds",
                }
            )
        else:
            for subset, n in (("charging_required", 144), ("no_charge_required", 36)):
                sub = [r for r in rows if r["method"] == m and r.get("charge_class") == subset]
                t2.append(
                    {
                        "method": LABEL[m],
                        "subset": subset,
                        "n_routes": n,
                        "feasibility": feas_rate(sub),
                        "ci_lo": None,
                        "ci_hi": None,
                        "uncertainty": "deterministic",
                    }
                )
    _write_csv(TAB / "table02_charging_required.csv", t2)
    md = [
        "# Table 2 — Charging-required / no-charge subsets",
        "",
        "`n_routes` is the number of TEST routes in the subset (not seed×route).",
        "FA-HPPO uncertainty is a 95% Student-t interval across five independently trained seeds.",
        "",
        "| Method | Subset | n_routes | Feasibility |",
        "|---|---|---:|---:|",
    ]
    for r in t2:
        if r["ci_lo"] is not None:
            feas = f"{100*r['feasibility']:.2f}% [{100*r['ci_lo']:.2f}, {100*r['ci_hi']:.2f}]"
        else:
            feas = f"{100*r['feasibility']:.2f}%"
        md.append(f"| {r['method']} | {r['subset']} | {r['n_routes']} | {feas} |")
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
        "Uncertainty is mean ± SD across five training seeds (not a 95% Student-t CI).",
        "Full definitions: Progress = Time + progress failure; Normalized = Normalized equivalent;",
        "PBRS = Potential-shaped; Time-horizon = selected normalized time-horizon reward.",
        "",
        "| Reward | VAL feasibility (mean ± SD) | VAL completion (mean ± SD) | Mean best update | Selected |",
        "|---|---:|---:|---:|---|",
    ]
    for r in t4:
        md.append(
            f"| {r['reward']} | {100*r['val_feasibility']:.1f}% ± {100*r['val_feasibility_sd']:.1f} | "
            f"{r['val_completion']:.3f} ± {r['val_completion_sd']:.3f} | {r['mean_best_update']:.0f} | "
            f"{'yes' if r['selected'] else ''} |"
        )
    (TAB / "table04_reward_ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Table 5 operational (supplemental)
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
        n_unit = "route×seed" if m == METHOD_CODE else "route"
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
                    "n_unit": n_unit,
                    "mean": mu,
                    "sd": sd,
                }
            )
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
                    "n_unit": "matched route×seed pairs",
                    "mean": mu,
                    "sd": sd,
                }
            )
    _write_csv(TAB / "table05_operational_metrics.csv", t5)
    md = [
        "# Table 5 — Operational metrics (supplemental)",
        "",
        "> Method-feasible rows carry a **survivor-bias warning** and must not be read as",
        "> cross-method efficiency gains. Prefer matched common-feasible deltas for comparisons.",
        "",
        "Units of `n`:",
        "- FA-HPPO method-feasible rows: route×seed observations;",
        "- deterministic baselines: route observations;",
        "- matched comparisons: matched route×seed pairs.",
        "",
        "| Conditioning | Method | Metric | n | n_unit | Mean | SD |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for r in t5:
        md.append(
            f"| {r['conditioning']} | {r['method']} | {r['metric']} | {r['n']} | {r['n_unit']} | "
            f"{r['mean']:.6g} | {r['sd']:.6g} |"
        )
    (TAB / "table05_operational_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig01_usecase() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    depot = np.array([0.08, 0.50])
    customers = np.array([[0.26, 0.72], [0.44, 0.38], [0.62, 0.70], [0.78, 0.42], [0.92, 0.62]])
    stations = np.array([[0.44, 0.58], [0.70, 0.28]])
    # primary frozen customer sequence (solid)
    route_x = [depot[0], *customers[:, 0], depot[0]]
    route_y = [depot[1], *customers[:, 1], depot[1]]
    ax.plot(route_x, route_y, "-", color="#0072B2", lw=2.0, zorder=1, label="Fixed customer sequence")
    # explicit charging insertion detour: C2 → S1 → C3 (replace direct C2–C3 visually)
    c2, c3, s1 = customers[1], customers[2], stations[0]
    ax.plot([c2[0], s1[0], c3[0]], [c2[1], s1[1], c3[1]], "--", color="#D55E00", lw=2.0, zorder=2, label="Charging insertion")
    ax.scatter(*depot, s=160, c="#000000", marker="s", zorder=4, label="Depot")
    ax.scatter(customers[:, 0], customers[:, 1], s=90, c="#0072B2", zorder=4, label="Customers")
    ax.scatter(stations[:, 0], stations[:, 1], s=120, c="#D55E00", marker="^", zorder=4, label="Stations")
    for i, (x, y) in enumerate(customers, 1):
        ax.text(x, y + 0.05, f"C{i}", ha="center", fontsize=8)
    for i, (x, y) in enumerate(stations, 1):
        ax.text(x + 0.03, y - 0.05, f"S{i}", ha="left", fontsize=8, color="#D55E00")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    _save(fig, "fig01_usecase_route", split="methodology", sources=[])


def fig02_method() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        "Fixed route\nstate",
        "Feasibility\nshield",
        "Features",
        "Hybrid\nPPO",
        "CONTINUE /\nstation",
        "Amount u +\nSOC map",
        "Simulator\ntransition",
    ]
    n = len(labels)
    left, right = 0.015, 0.985
    width = 0.112
    span = right - left - width
    xs = [left + i * span / (n - 1) for i in range(n)]
    for x, text in zip(xs, labels):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.42),
                width,
                0.28,
                boxstyle="round,pad=0.006",
                facecolor="#E8F1F8",
                edgecolor="#0072B2",
                lw=1.0,
                clip_on=False,
            )
        )
        ax.text(x + width / 2, 0.56, text, ha="center", va="center", fontsize=7.0)
    for i in range(n - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1] - 0.002, 0.56),
            xytext=(xs[i] + width + 0.002, 0.56),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
            clip_on=False,
        )
    ax.text(
        0.5,
        0.22,
        r"$r_t=-\Delta t/C_{\mathrm{train}}$,\ \ $r_{\mathrm{fail}}=-(H-t)/C_{\mathrm{train}}$\ \ ($C_{\mathrm{train}}=10$)",
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
    _save(fig, "fig02_method_schematic", split="methodology", sources=["results/v4/reward_development/FINAL_REWARD_FREEZE.json"])


def fig03_envelope() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    # Implementation: SOC_lower = max(SOC_arrival, SOC_continuation)
    # Illustrative values must obey arrival <= lower.
    arrival, lower, upper, u = 0.25, 0.35, 0.90, 0.55
    target = lower + u * (upper - lower)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.axhline(upper, color="#D55E00", lw=2.0)
    ax.axhline(lower, color="#009E73", lw=2.0)
    ax.axhline(target, color="#0072B2", lw=2.2)
    ax.axhline(arrival, color="#666666", ls="--", lw=1.4)
    ax.fill_between([0.30, 0.70], lower, upper, color="#0072B2", alpha=0.10)
    ax.annotate("", xy=(0.50, upper - 0.01), xytext=(0.50, lower + 0.01), arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2))
    ax.text(0.52, (lower + upper) / 2, "admissible\ninterval", ha="left", va="center", fontsize=8, color="#333333")
    ax.annotate("", xy=(0.38, target), xytext=(0.38, lower), arrowprops=dict(arrowstyle="<->", color="#0072B2", lw=1.0))
    ax.text(0.22, (lower + target) / 2, r"$u$ fraction", ha="center", va="center", fontsize=8, color="#0072B2")
    ax.text(0.74, upper, r"$SOC_{\mathrm{upper}}$", va="center", color="#D55E00", fontsize=9)
    ax.text(0.74, lower, r"$SOC_{\mathrm{lower}}$", va="center", color="#009E73", fontsize=9)
    ax.text(0.74, target, r"$SOC_{\mathrm{target}}$", va="center", color="#0072B2", fontsize=9)
    ax.text(0.74, arrival, "Arrival SOC", va="center", color="#666666", fontsize=9)
    ax.text(0.05, 1.02, "Optimistic time-feasibility upper bound", fontsize=7.5, color="#D55E00")
    ax.text(
        0.05,
        0.02,
        r"Target lower bound $=\max($arrival SOC, energy-continuation$)$",
        fontsize=7.0,
        color="#009E73",
    )
    ax.text(0.05, target + 0.04, "Selected target SOC", fontsize=7.5, color="#0072B2")
    ax.set_xticks([])
    ax.set_ylabel("SOC")
    _save(fig, "fig03_soc_envelope", split="methodology", sources=[])


def fig04_soc_traj() -> None:
    if not TRAJ.is_file():
        raise SystemExit("missing VAL trajectory; run scripts/paper/record_v4_illustrative_val_trajectory.py")
    ep = _load_json(TRAJ)
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    tl = ep["soc_timeline"]
    times = [float(p["time"]) for p in tl]
    socs = [float(p["soc"]) for p in tl]
    ax.plot(times, socs, "-o", color="#0072B2", lw=1.8, ms=4.5, zorder=3)
    for p in tl:
        if p.get("kind") == "station_arrive":
            ax.axvline(float(p["time"]), color="#D55E00", ls=":", lw=1.0, alpha=0.8)
            ax.scatter([float(p["time"])], [float(p["soc"])], c="#D55E00", s=36, zorder=4, marker="^")
            if p.get("soc_lower") is not None and p.get("soc_upper") is not None:
                ax.vlines(
                    float(p["time"]),
                    float(p["soc_lower"]),
                    float(p["soc_upper"]),
                    colors="#999999",
                    lw=3,
                    alpha=0.35,
                    zorder=2,
                )
            ax.text(float(p["time"]), min(0.97, float(p["soc"]) + 0.08), p.get("node", "S"), color="#D55E00", fontsize=7, ha="center")
        elif p.get("kind") == "arrive" and str(p.get("node", "")).startswith("C"):
            ax.text(float(p["time"]), float(p["soc"]) - 0.08, p["node"], fontsize=6.5, ha="center", color="#444444")
    ax.set_xlabel("Time")
    ax.set_ylabel("SOC")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"VALIDATION route {ep['route_id']} (seed {ep['seed']}; not TEST)", loc="left", fontsize=8)
    _save(
        fig,
        "fig04_illustrative_soc",
        split="VAL",
        sources=["results/v4/paper/data/illustrative_val_trajectory.json"],
    )


def _bar_methods(
    ax,
    values: dict[str, float],
    yerr: dict[str, float] | None,
    ylabel: str,
    ylim=None,
    *,
    fmt: str = "{:.1f}%",
) -> None:
    xs = np.arange(len(METHODS))
    means = [values[m] for m in METHODS]
    colors = [COLOR[m] for m in METHODS]
    ax.bar(xs, means, color=colors, width=0.72, edgecolor="white", linewidth=0.4)
    # Only FA-HPPO receives error-bar artists
    if yerr and yerr.get(METHOD_CODE):
        ax.errorbar(
            [xs[0]],
            [means[0]],
            yerr=[yerr[METHOD_CODE]],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            elinewidth=1.0,
        )
    y_top = ylim[1] if ylim and ylim[1] is not None else max(means) * 1.15
    y_bot = ylim[0] if ylim else 0.0
    pad = 0.015 * (y_top - y_bot + 1e-9)
    for i, m in enumerate(METHODS):
        ax.text(xs[i], means[i] + pad, fmt.format(means[i]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [SHORT[m] for m in METHODS], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def fig05_feas(rows: list[dict], stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    values = {m: 100 * stats["methods"][LABEL[m]]["feasibility_mean"] for m in METHODS}
    yerr = {}
    info = stats["methods"][LABEL[METHOD_CODE]]
    lo, hi = info["feasibility_t95"]
    yerr[METHOD_CODE] = 100 * max(info["feasibility_mean"] - lo, hi - info["feasibility_mean"])
    _bar_methods(ax, values, yerr, "TEST feasibility (%)", ylim=(0, 105), fmt="{:.1f}%")
    _save(fig, "fig05_main_feasibility", split="TEST", sources=["results/v4/test/raw/synthcharge_v4_test.jsonl"])


def fig06_comp(stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    values = {m: stats["methods"][LABEL[m]]["completion_mean"] for m in METHODS}
    yerr = {}
    info = stats["methods"][LABEL[METHOD_CODE]]
    lo, hi = info["completion_t95"]
    yerr[METHOD_CODE] = max(info["completion_mean"] - lo, hi - info["completion_mean"])
    _bar_methods(ax, values, yerr, "Failure-retaining completion", ylim=(0, max(values.values()) * 1.25), fmt="{:.3f}")
    _save(fig, "fig06_main_completion", split="TEST", sources=["results/v4/test/raw/synthcharge_v4_test.jsonl"])


def fig07_charging(rows: list[dict], stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    values = {}
    yerr = {}
    for m in METHODS:
        if m == METHOD_CODE:
            info = stats["by_charge_class"]["charging_required"]
            values[m] = 100 * info["mean"]
            lo, hi = info["t95"]
            yerr[METHOD_CODE] = 100 * max(info["mean"] - lo, hi - info["mean"])
        else:
            sub = [r for r in rows if r["method"] == m and r.get("charge_class") == "charging_required"]
            values[m] = 100 * feas_rate(sub)
    _bar_methods(ax, values, yerr, "Charging-required feasibility (%)", ylim=(0, 105), fmt="{:.1f}%")
    _save(fig, "fig07_charging_required", split="TEST", sources=["results/v4/test/raw/synthcharge_v4_test.jsonl"])


def fig08_ecdf(rows: list[dict]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    # One value per route for every method (FA-HPPO seed-averaged)
    route_ids = sorted({r["route_id"] for r in rows if r["method"] != METHOD_CODE})
    for m in METHODS:
        if m == METHOD_CODE:
            by_route = defaultdict(list)
            for r in rows:
                if r["method"] == METHOD_CODE:
                    by_route[r["route_id"]].append(float(r["completion_time_all_routes"]))
            vals = np.sort([float(np.mean(by_route[rid])) for rid in route_ids])
        else:
            vals = np.sort([float(r["completion_time_all_routes"]) for r in rows if r["method"] == m])
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, color=COLOR[m], lw=1.6, label=SHORT[m])
    ax.set_xlabel("Failure-retaining completion")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig08_performance_ecdf", split="TEST", sources=["results/v4/test/raw/synthcharge_v4_test.jsonl"])


def fig09_layout(rows: list[dict], stats: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    layouts = ["C", "R", "RC"]
    x = np.arange(len(layouts))
    width = 0.18
    for i, m in enumerate(METHODS):
        means = []
        for layout in layouts:
            if m == METHOD_CODE:
                means.append(100 * stats["by_layout"][layout]["mean"])
            else:
                sub = [r for r in rows if r["method"] == m and r.get("layout") == layout]
                means.append(100 * feas_rate(sub))
        xpos = x + (i - 1.5) * width
        ax.bar(xpos, means, width=width, color=COLOR[m], label=SHORT[m])
        if m == METHOD_CODE:
            errs = []
            for layout in layouts:
                info = stats["by_layout"][layout]
                lo, hi = info["t95"]
                errs.append(100 * max(info["mean"] - lo, hi - info["mean"]))
            ax.errorbar(xpos, means, yerr=errs, fmt="none", ecolor="#333333", capsize=2, elinewidth=0.8)
    ax.set_xticks(x, layouts)
    ax.set_ylabel("TEST feasibility (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save(fig, "fig09_by_layout", split="TEST", sources=["results/v4/test/raw/synthcharge_v4_test.jsonl"])


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
    _save(fig, "fig10_by_length", split="TEST", sources=["results/v4/test/raw/synthcharge_v4_test.jsonl"])


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
            val = mat[i, j]
            color = "white" if val >= 55 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Feasibility (%)")
    ax.set_xlabel("Frozen-route-length bin")
    ax.set_ylabel("Layout")
    _save(fig, "fig11_difficulty_heatmap", split="TEST", sources=["results/v4/test/raw/synthcharge_v4_test.jsonl"])


def fig12_ablation() -> None:
    _style()
    abl = _load_json(ABLATION)["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
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
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor="#333333", capsize=3, zorder=2)
        for i, (c, _) in enumerate(REWARD_ORDER):
            marker = "D" if c == "V4_BASE_NO_L_FAIL" else "o"
            ax.plot(i, vals[i], marker, color=colors[i], markersize=7.5, markeredgecolor="black", markeredgewidth=0.45, zorder=4)
        ax.set_xticks(xs)
        tick_labels = ax.set_xticklabels(labels, rotation=0, ha="center")
        for tick, (_, lab) in zip(tick_labels, REWARD_ORDER):
            if lab == "Time-horizon":
                tick.set_fontweight("bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        if ax is axes[0]:
            ax.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, "fig12_reward_ablation_val", split="VAL", sources=["results/v4/reward_development/SUMMARY.json"])


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
        sources=[f"results/v4/reward_development/final_authoritative/V4_BASE_NO_L_FAIL/seed_{s}/curves.jsonl" for s in SEEDS],
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
        sources=[f"results/v4/reward_development/final_authoritative/V4_BASE_NO_L_FAIL/seed_{s}/curves.jsonl" for s in SEEDS],
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
        sources=[f"results/v4/reward_development/final_authoritative/V4_BASE_NO_L_FAIL/seed_{s}/curves.jsonl" for s in SEEDS],
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
        sources=[f"results/v4/reward_development/final_authoritative/V4_BASE_NO_L_FAIL/seed_{s}/curves.jsonl" for s in SEEDS],
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
    _save(fig, "figA03_per_seed_test_feasibility", split="TEST", sources=["results/v4/test/raw/synthcharge_v4_test.jsonl"])


def figA4_failures(rows: list[dict]) -> None:
    _style()
    reasons = defaultdict(int)
    for r in rows:
        if r["method"] == METHOD_CODE and not r.get("feasible"):
            reasons[str(r.get("reason") or "unknown")] += 1
    items = sorted(reasons.items(), key=lambda kv: -kv[1])[:8]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    if not items:
        ax.text(0.5, 0.5, "No failures", ha="center")
    else:
        codes = [k for k, _ in items]
        labels = [FAILURE_LABEL.get(k, k.replace("_", " ").title()) for k in codes]
        vals = [v for _, v in items]
        ax.barh(range(len(labels)), vals, color="#D55E00")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Failure count (route×seed, pooled)")
    fig.tight_layout()
    _save(fig, "figA04_failure_reasons", split="TEST", sources=["results/v4/test/raw/synthcharge_v4_test.jsonl"])


# ---------------------------------------------------------------------------
# Captions / README / Manifest
# ---------------------------------------------------------------------------


def write_captions() -> None:
    CAP.mkdir(parents=True, exist_ok=True)
    traj = _load_json(TRAJ) if TRAJ.is_file() else {}
    rule = traj.get("selection", {}).get(
        "rule",
        "First charging-required VALIDATION route in stable route_id order that is feasible under the frozen V4 seed-42 checkpoint and contains at least one charging action.",
    )
    route_id = traj.get("route_id", "N/A")
    seed = traj.get("seed", 42)
    text = """# V4 figure captions (standalone)

All TEST figures use the fresh independently generated held-out SynthCharge TEST.
TRAIN/VAL figures are development evidence; TEST was not used for training or checkpoint selection.
Uncertainty conventions are stated explicitly per figure (never as generic "error bars").

**Fig. 1 (methodology).** Fixed-route EV charging use case. The solid path is the frozen customer sequence (depot → customers → depot). Candidate charging stations are shown separately; the dashed orange path illustrates one charging insertion (customer → station → next frozen customer). The customer order itself is unchanged. Charge insertion is a learned decision; the figure only depicts the geometric idea.

**Fig. 2 (methodology).** FA-HPPO control loop: fixed-route state → feasibility shield → features → hybrid PPO → CONTINUE/station → continuous amount $u$ with SOC mapping → simulator transition. Equations below the workflow state the normalized time-horizon reward and the SOC-target map $SOC_{\\mathrm{target}}=SOC_{\\mathrm{lower}}+u(SOC_{\\mathrm{upper}}-SOC_{\\mathrm{lower}})$.

**Fig. 3 (methodology).** Single-decision SOC envelope (not a simulated trajectory). Vertical scale shows arrival SOC below the target lower bound $SOC_{\\mathrm{lower}}=\\max($arrival SOC, energy-continuation requirement$)$, optimistic time-feasibility upper bound $SOC_{\\mathrm{upper}}$, and selected target SOC with $SOC_{\\mathrm{lower}}\\le SOC_{\\mathrm{target}}\\le SOC_{\\mathrm{upper}}$ and $SOC_{\\mathrm{target}}=SOC_{\\mathrm{lower}}+u(SOC_{\\mathrm{upper}}-SOC_{\\mathrm{lower}})$.

**Fig. 4 (VALIDATION).** Real SOC trajectory from frozen V4 evidence (not TEST). Selection rule: __RULE__ Selected route `__ROUTE_ID__`, seed __SEED__. Markers show station arrivals and SOC envelopes at charge decisions.

**Fig. 5 (TEST).** Main TEST feasibility for FA-HPPO and three deterministic baselines under the same feasibility-aware environment. FA-HPPO whiskers are the half-width of the 95% Student-t interval across five seeds; baselines have no uncertainty artists.

**Fig. 6 (TEST).** Failure-retaining completion on all TEST routes (infeasible episodes retain the route horizon $H$). Lower is better. FA-HPPO whiskers are the 95% Student-t half-width across five seeds; baselines are deterministic point values.

**Fig. 7 (TEST).** Feasibility on the charging-required TEST subset (144 routes). FA-HPPO whiskers are the 95% Student-t half-width across five seeds; baselines are deterministic. FA-HPPO reaches 100% on the complementary 36 no-charge-required routes.

**Fig. 8 (TEST).** Empirical CDF of failure-retaining completion. Every method contributes exactly 180 route-level values: FA-HPPO values are seed-averaged per route before constructing the ECDF; deterministic baselines contribute their single value per route.

**Fig. 9 (TEST).** TEST feasibility by layout (`C`, `R`, `RC`). FA-HPPO whiskers are 95% Student-t half-widths across five seeds within each layout; baselines have no uncertainty artists.

**Fig. 10 (TEST; supplemental candidate).** FA-HPPO TEST feasibility by balanced frozen-route-length bin (`short`, `medium`, `long`). Whiskers are 95% Student-t half-widths across five seeds. Bins are frozen-route-length strata, not customer-scale cells.

**Fig. 11 (TEST).** Layout × frozen-route-length feasibility heatmap for FA-HPPO on the locked TEST (balanced 3×3 strata). Color scale is fixed to 0%–100%.

**Fig. 12 (VALIDATION).** Reward-development ablation on VALIDATION. Display labels: Progress, Normalized, PBRS, Time-horizon (selected, diamond marker). Uncertainty is mean ± SD across five training seeds (not a 95% Student-t CI). Potential-based shaping did not improve validation feasibility; the simpler time-horizon reward was selected.

**Fig. 13 (TRAIN; supplemental).** TRAIN objective episode return ($G=-T$ on success, $G=-H$ on failure), mean across seeds with Student-t 95% interval on the common update support (no extrapolation after early stopping).

**Fig. 14 (TRAIN; supplemental).** TRAIN PPO policy and value losses with Student-t 95% intervals on the common update support. Diagnostics only; decreasing loss is not claimed as convergence. Policy loss may be negative.

**Fig. 15 (VALIDATION; supplemental).** VALIDATION parent-balanced feasibility versus PPO update, mean with Student-t 95% interval across five seeds on the common update support.

**Fig. A1 (TRAIN; supplemental).** Pre-clip gradient norm during authoritative training (Student-t 95% interval; log axis only where valid).

**Fig. A3 (TEST; supplemental).** Per-seed FA-HPPO feasibility on the locked TEST.

**Fig. A4 (TEST; supplemental).** Pooled FA-HPPO failure-reason counts on the locked TEST (route×seed). Display labels are human-readable; underlying reason codes are preserved in the raw evidence.
"""
    text = (
        text.replace("__RULE__", str(rule))
        .replace("__ROUTE_ID__", str(route_id))
        .replace("__SEED__", str(seed))
    )
    (CAP / "CAPTIONS.md").write_text(text, encoding="utf-8")


def write_readme(stats: dict) -> None:
    mu = stats["methods"]["FA-HPPO"]["feasibility_mean"]
    lo, hi = stats["methods"]["FA-HPPO"]["feasibility_t95"]
    text = f"""# V4 paper-facing package (standalone)

This directory contains **only** V4 manuscript-facing evidence.
It does not compare to prior experimental versions.

## Primary TEST result

FA-HPPO achieves **{100*mu:.2f}%** [{100*lo:.2f}, {100*hi:.2f}] feasibility
(95% Student-t CI across five seeds) on the fresh independently generated
held-out SynthCharge TEST and substantially outperforms One-step lookahead,
Greedy full charge, and Greedy minimum charge under the same feasibility-aware
environment.

## Recommended main-paper figures

Use a compact subset in the manuscript; keep the rest as a figure library / supplement.

| Priority | Figure | Role |
|---|---|---|
| Main | Fig. 1 `fig01_usecase_route.png` | Fixed-route charging problem |
| Main | Fig. 2 `fig02_method_schematic.png` | FA-HPPO method schematic |
| Main | Fig. 3 `fig03_soc_envelope.png` | SOC envelope |
| Main | Fig. 5 `fig05_main_feasibility.png` | Main TEST feasibility |
| Main | Fig. 6 or Fig. 8 | Completion / distribution |
| Main | Fig. 11 `fig11_difficulty_heatmap.png` | Difficulty heatmap |
| Main | Fig. 12 `fig12_reward_ablation_val.png` | Reward ablation |
| Optional main | Fig. 4 `fig04_illustrative_soc.png` | Real VAL SOC trajectory (if space) |
| Supplement | Figs. 7, 9, 10, 13–15, A1, A3, A4 | Charging subset, strata, training, diagnostics |

Do not force all 18 figures into the main manuscript.

## Reproduce displays (no TEST rerun / no retraining)

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

Illustrative VAL trajectory source (Fig. 4):
`results/v4/paper/data/illustrative_val_trajectory.json`
(regenerate with `python scripts/paper/record_v4_illustrative_val_trajectory.py` only if needed; do not retrain).

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
        "package": "results/v4/paper",
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
    # Pinned frozen evidence must be unchanged (no TEST rerun / reward change)
    pinned = {
        "raw": (_lf_sha(RAW), PINNED_RAW_SHA256_LF),
        "test_lock": (_lf_sha(LOCK), PINNED_TEST_LOCK_SHA256_LF),
        "evaluation_consumed": (_sha(CONSUMED), PINNED_EVALUATION_CONSUMED_SHA256),
        "reward_freeze": (_sha(REWARD_FREEZE), PINNED_REWARD_FREEZE_SHA256),
    }
    for name, (got, exp) in pinned.items():
        if got != exp:
            raise SystemExit(f"--verify failed: pinned {name} hash changed ({got} != {exp})")
    # Manifest must also mirror current frozen files
    checks = {
        "raw": (_lf_sha(RAW), manifest["frozen_source_hashes"]["raw"]),
        "test_lock": (_lf_sha(LOCK), manifest["frozen_source_hashes"]["test_lock"]),
        "evaluation_consumed": (_sha(CONSUMED), manifest["frozen_source_hashes"]["evaluation_consumed"]),
        "reward_freeze": (_sha(REWARD_FREEZE), manifest["frozen_source_hashes"]["reward_freeze"]),
    }
    for name, (got, exp) in checks.items():
        if got != exp:
            raise SystemExit(f"--verify failed: manifest {name} hash mismatch")
    if abs(manifest["main_test_feasibility_mean"] - PINNED_MAIN_FEAS) > 1e-9:
        raise SystemExit("--verify failed: main feasibility mismatch")
    if manifest["test"]["n_routes"] != 180:
        raise SystemExit("--verify failed: n_routes")
    if not manifest.get("standalone", False) or manifest.get("prior_version_comparisons", True):
        raise SystemExit("--verify failed: package must remain standalone")
    if not all(
        (
            manifest.get("no_test_rerun"),
            manifest.get("no_checkpoint_reselection"),
            manifest.get("no_reward_redesign"),
            manifest.get("no_post_test_training"),
        )
    ):
        raise SystemExit("--verify failed: accidental retraining/TEST-rerun flags")

    # All expected figures exist and hashes match
    present = {p.name for p in FIG.glob("*.png")}
    missing = [name for name in EXPECTED_FIGS if name not in present]
    if missing:
        raise SystemExit(f"--verify failed: missing figures {missing}")
    by_name = {e["filename"]: e for e in manifest["figures"]}
    for name in EXPECTED_FIGS:
        if name not in by_name:
            raise SystemExit(f"--verify failed: figure not in manifest {name}")
        entry = by_name[name]
        path = ROOT / entry["path"]
        if not path.is_file():
            raise SystemExit(f"--verify failed: missing figure file {name}")
        if _sha(path) != entry["figure_sha256"]:
            raise SystemExit(f"--verify failed: figure hash drift {name}")
        if "evidence_split" not in entry or "builder_script" not in entry:
            raise SystemExit(f"--verify failed: incomplete figure metadata {name}")
        if "source_files" not in entry or "source_hashes" not in entry:
            raise SystemExit(f"--verify failed: incomplete source metadata {name}")

    # Fig. 4 must reference real VAL trajectory source
    fig4 = by_name["fig04_illustrative_soc.png"]
    if "results/v4/paper/data/illustrative_val_trajectory.json" not in fig4.get("source_files", []):
        raise SystemExit("--verify failed: Fig. 4 missing real VAL trajectory source")
    if not TRAJ.is_file():
        raise SystemExit("--verify failed: illustrative VAL trajectory file missing")
    traj = _load_json(TRAJ)
    if traj.get("selection", {}).get("split") != "validation":
        raise SystemExit("--verify failed: Fig. 4 trajectory is not VALIDATION")
    if traj.get("selection", {}).get("not_test_evidence") is not True:
        raise SystemExit("--verify failed: Fig. 4 must not be TEST evidence")
    if int(traj.get("seed", -1)) != 42:
        raise SystemExit("--verify failed: Fig. 4 must use seed 42")
    if not traj.get("feasible") or int(traj.get("n_charge_actions", 0)) < 1:
        raise SystemExit("--verify failed: Fig. 4 trajectory must be feasible with ≥1 charge")
    traj_hash = fig4["source_hashes"].get("results/v4/paper/data/illustrative_val_trajectory.json")
    if traj_hash != _sha(TRAJ):
        raise SystemExit("--verify failed: Fig. 4 trajectory source hash mismatch")

    # No prior-version labels in paper-facing text outputs
    forbidden = (
        "V3_FA_HPPO",
        "V3-HPPO",
        "prior-version",
        "external-domain",
        "results_v3",
        "results_v2",
        "results_v1",
    )
    text_paths = list(TAB.glob("*.md")) + [CAP / "CAPTIONS.md", OUT / "README.md", OUT / "MANIFEST.json"]
    for path in text_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise SystemExit(f"--verify failed: forbidden token {token!r} in {path}")

    print(
        "VERIFY OK",
        json.dumps(
            {
                "n_figures": len(manifest["figures"]),
                "feas": manifest["main_test_feasibility_mean"],
                "fig04_route": traj.get("route_id"),
                "fig04_seed": traj.get("seed"),
            },
            indent=2,
        ),
    )


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
