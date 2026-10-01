"""Analysis-only final V2 statistics, tables, figures, integrity, and report.

Reads raw rows from evaluate_final_v2.py. Does not retrain or re-evaluate.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v2"))

from experiments.stats import (  # noqa: E402
    hierarchical_bootstrap_ci,
    holm,
    mean_sd_across_training_seeds,
    paired_parent_diff_hierarchical,
    permutation_pvalue,
)

from final_common import (  # noqa: E402
    BASELINES,
    DATASETS,
    FINAL,
    METHOD_FREEZE_SHA,
    METHODS,
    SEEDS,
    TRAINING_EXECUTION_SHA,
    dump_json,
    git_head,
    lf_sha256,
    load_json,
    method_tree_diff,
    sha256,
)

RAW = FINAL / "raw"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"
STATS = FINAL / "statistics"
LEARNED = METHODS
ALL_METHODS = LEARNED + BASELINES
N_BOOT = 5000
N_PERM = 20000
RNG = 20261001


def load_rows(name: str) -> list[dict]:
    path = RAW / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def integrity(rows: list[dict], *, n_routes: int, learned_factor: int, baseline_factor: int, name: str) -> dict:
    issues = []
    route_ids = sorted({r["route_id"] for r in rows})
    if len(route_ids) != n_routes:
        issues.append(f"unique routes {len(route_ids)} != {n_routes}")
    if len(rows) != learned_factor + baseline_factor:
        issues.append(f"row count {len(rows)} != {learned_factor + baseline_factor}")
    for method in LEARNED:
        subset = method_rows(rows, method)
        if len(subset) != n_routes * 5:
            issues.append(f"{method} rows {len(subset)} != {n_routes * 5}")
        pairs = [(r["route_id"], r["seed"]) for r in subset]
        if len(pairs) != len(set(pairs)):
            issues.append(f"{method} duplicate route-seed")
        for seed in SEEDS:
            seed_routes = {r["route_id"] for r in method_rows(rows, method, seed)}
            if seed_routes != set(route_ids):
                issues.append(f"{method} seed {seed} route set mismatch")
    for method in BASELINES:
        subset = method_rows(rows, method)
        if len(subset) != n_routes:
            issues.append(f"{method} rows {len(subset)} != {n_routes}")
        ids = [r["route_id"] for r in subset]
        if len(ids) != len(set(ids)):
            issues.append(f"{method} duplicate route")
        if set(ids) != set(route_ids):
            issues.append(f"{method} route set mismatch")
    forbidden = {"DDQN", "SAC", "AttentionPPO"}
    found = {r["method"] for r in rows} & forbidden
    if found:
        issues.append(f"forbidden methods present: {found}")
    profiles = {r["physics_profile"] for r in rows}
    if name == "synthcharge" and profiles != {"synthcharge_linear"}:
        issues.append(f"bad physics profiles: {profiles}")
    if name == "legacy" and profiles != {"official_evrptwgr"}:
        issues.append(f"bad physics profiles: {profiles}")
    for r in rows:
        if not r["feasible"]:
            if abs(float(r["completion_time_all_routes"]) - float(r["horizon"])) > 1e-9:
                issues.append(f"infeasible completion not H: {r['route_id']} {r['method']}")
                break
    payload = {"benchmark": name, "n_issues": len(issues), "issues": issues, "n_rows": len(rows), "n_routes": len(route_ids)}
    if issues:
        raise SystemExit(f"integrity failed for {name}: {issues}")
    return payload


def summarize_learned(rows, method):
    feas = mean_sd_across_training_seeds(method_rows(rows, method), "feasible")
    comp = mean_sd_across_training_seeds(method_rows(rows, method), "completion_time_all_routes")
    # Convert feasible bools already float-compatible via json.
    for r in method_rows(rows, method):
        r["feasible_f"] = float(bool(r["feasible"]))
    feas_ci = hierarchical_bootstrap_ci(method_rows(rows, method), "feasible_f", n_boot=N_BOOT, seed=RNG)
    for r in method_rows(rows, method):
        r["feasible_f"] = float(bool(r["feasible"]))
    # rebuild with feasible_f key on copies
    learned = []
    for r in method_rows(rows, method):
        copy = dict(r)
        copy["feasible_f"] = float(bool(r["feasible"]))
        learned.append(copy)
    feas_ci = hierarchical_bootstrap_ci(learned, "feasible_f", n_boot=N_BOOT, seed=RNG)
    comp_ci = hierarchical_bootstrap_ci(learned, "completion_time_all_routes", n_boot=N_BOOT, seed=RNG)
    feas_seed = {int(k): v["mean"] for k, v in feas["per_seed"].items()}
    return {
        "method": method,
        "n_routes": len({r["route_id"] for r in method_rows(rows, method)}),
        "n_seeds": 5,
        "feasibility_mean": feas["mean_across_seeds"],
        "feasibility_sd": feas["sd_across_seeds"],
        "feasibility_ci95": [feas_ci["lo"], feas_ci["hi"]],
        "completion_all_mean": comp["mean_across_seeds"],
        "completion_all_sd": comp["sd_across_seeds"],
        "completion_all_ci95": [comp_ci["lo"], comp_ci["hi"]],
        "per_seed_feasibility": feas_seed,
        "mean_feasible_routes_per_seed": feas["mean_across_seeds"] * len({r["route_id"] for r in method_rows(rows, method)}) if feas["mean_across_seeds"] is not None else None,
    }


def summarize_baseline(rows, method):
    subset = method_rows(rows, method)
    feasible = [r for r in subset if r["feasible"]]
    return {
        "method": method,
        "n_routes": len(subset),
        "n_seeds": 1,
        "feasibility_mean": mean(float(r["feasible"]) for r in subset),
        "feasibility_sd": None,
        "feasibility_ci95": None,
        "completion_all_mean": mean(r["completion_time_all_routes"] for r in subset),
        "completion_all_sd": None,
        "completion_all_ci95": None,
        "per_seed_feasibility": None,
        "mean_feasible_routes_per_seed": sum(1 for r in subset if r["feasible"]),
        "feasible_only_completion": mean(r["route_completion_time"] for r in feasible),
        "charging_time": mean(r["total_charging_time"] for r in feasible),
        "station_visits": mean(r["n_station_visits"] for r in feasible),
    }


def operational(rows, method):
    if method in LEARNED:
        per = []
        for seed in SEEDS:
            subset = method_rows(rows, method, seed)
            feasible = [r for r in subset if r["feasible"]]
            per.append(
                {
                    "feasible_only_completion": mean(r["route_completion_time"] for r in feasible),
                    "charging_time": mean(r["total_charging_time"] for r in feasible),
                    "station_visits": mean(r["n_station_visits"] for r in feasible),
                    "travel_time": mean(r["total_travel_time"] for r in feasible),
                    "waiting_time": mean(r["total_waiting_at_customers"] for r in feasible),
                    "terminal_soc": mean(r["terminal_soc"] for r in feasible),
                }
            )
        return {k: mean(p[k] for p in per) for k in per[0]}
    return summarize_baseline(rows, method)


def paired_primary(rows, label: str):
    hybrid = method_rows(rows, "HybridPPO")
    discrete = method_rows(rows, "DiscretePPO")
    for r in hybrid + discrete:
        r["feasible_f"] = float(bool(r["feasible"]))
    tests = []
    for family, key in (("feasibility", "feasible_f"), ("completion_all_routes", "completion_time_all_routes")):
        diffs = paired_parent_diff_hierarchical(
            hybrid, discrete, key, "base_instance", a_mean_over_seeds=True, b_mean_over_seeds=True
        )
        values = list(diffs.values())
        p = permutation_pvalue(values, n_perm=N_PERM, seed=RNG)
        # bootstrap CI of mean paired difference
        arr = np.asarray(values, dtype=float)
        rng = np.random.default_rng(RNG)
        if arr.size:
            boots = [float(arr[rng.integers(0, arr.size, size=arr.size)].mean()) for _ in range(N_BOOT)]
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        else:
            lo = hi = None
        tests.append(
            {
                "benchmark": label,
                "family": family,
                "comparison": "HybridPPO - DiscretePPO",
                "effect": float(np.mean(arr)) if arr.size else None,
                "ci95": [lo, hi],
                "ci_type": "paired parent/route percentile bootstrap of seed-averaged differences",
                "p_raw": p,
                "n_independent_units": len(values),
                "test": "paired sign-flip / permutation on seed-averaged parent/route differences",
            }
        )
    adjusted = holm([(t["family"], t["p_raw"]) for t in tests])
    for t, (_name, _p, adj) in zip(tests, adjusted):
        t["p_holm"] = adj
        t["holm_family"] = f"{label}: HybridPPO vs DiscretePPO (feasibility, completion_all)"
    return tests


def failure_analysis(rows):
    out = {}
    for method in ALL_METHODS:
        fails = [r for r in method_rows(rows, method) if not r["feasible"]]
        reasons = Counter(r.get("reason") or "none" for r in fails)
        out[method] = {
            "n_failed_evaluations": len(fails),
            "failure_reasons": dict(reasons),
            "examples": [
                {
                    "route_id": r["route_id"],
                    "seed": r.get("seed"),
                    "reason": r.get("reason"),
                    "dead_end": r.get("dead_end"),
                    "n_customers": r.get("n_customers"),
                    "charge_class": r.get("charge_class"),
                    "layout": r.get("layout"),
                    "length_bin": r.get("length_bin"),
                }
                for r in fails[:50]
            ],
        }
    return out


def cell_breakdown(rows):
    cells = []
    layouts = sorted({r["layout"] for r in rows if r.get("layout")})
    bins = ["short", "medium", "long"]
    classes = ["charging_required", "no_charge_required"]
    for layout in layouts:
        for length in bins:
            for charge in classes:
                subset = [
                    r
                    for r in rows
                    if r.get("layout") == layout and r.get("length_bin") == length and r.get("charge_class") == charge
                ]
                if not subset:
                    continue
                n_routes = len({r["route_id"] for r in subset})
                for method in ALL_METHODS:
                    if method in LEARNED:
                        s = summarize_learned(subset, method)
                        feas = s["feasibility_mean"]
                        comp = s["completion_all_mean"]
                    else:
                        s = summarize_baseline(subset, method)
                        feas = s["feasibility_mean"]
                        comp = s["completion_all_mean"]
                    cells.append(
                        [
                            layout,
                            length,
                            charge,
                            method,
                            n_routes,
                            fmt(feas),
                            fmt(comp),
                            fmt(operational(subset, method).get("charging_time")),
                            fmt(operational(subset, method).get("station_visits"), 2),
                        ]
                    )
    return cells


def figures(synth, legacy):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "synthcharge").mkdir(exist_ok=True)
    (FIGURES / "legacy").mkdir(exist_ok=True)

    def bar_feas(rows, title, path):
        vals, errs, labels = [], [], []
        for method in ALL_METHODS:
            if method in LEARNED:
                s = summarize_learned(rows, method)
                vals.append(s["feasibility_mean"])
                errs.append(s["feasibility_sd"] or 0.0)
            else:
                s = summarize_baseline(rows, method)
                vals.append(s["feasibility_mean"])
                errs.append(0.0)
            labels.append(method.replace("GreedyMinimumSufficientCharge", "GreedyMin").replace("GreedyFullCharge", "GreedyFull").replace("OneStepLookahead", "Lookahead"))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(labels)), vals, yerr=errs, capsize=4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Feasibility (learned: mean±SD across 5 seeds)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    bar_feas(synth, "SynthCharge fresh TEST feasibility", FIGURES / "synthcharge" / "feasibility.png")
    bar_feas(legacy, "Legacy same-domain challenge feasibility (NOT fresh TEST)", FIGURES / "legacy" / "feasibility.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for j, method in enumerate(LEARNED):
        ys = [summarize_learned(synth, method)["per_seed_feasibility"][s] for s in SEEDS]
        ax.scatter(np.full(5, j) + np.linspace(-0.12, 0.12, 5), ys, label=method)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LEARNED)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-seed feasibility")
    ax.set_title("SynthCharge TEST per-seed feasibility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "synthcharge" / "per_seed_feasibility.png", dpi=150)
    plt.close(fig)

    # heatmap HybridPPO feasibility by layout x length for charging_required
    layouts = ["R", "C", "RC"]
    bins = ["short", "medium", "long"]
    mat = np.zeros((3, 3))
    for i, layout in enumerate(layouts):
        for j, length in enumerate(bins):
            subset = [
                r
                for r in synth
                if r.get("layout") == layout
                and r.get("length_bin") == length
                and r.get("charge_class") == "charging_required"
                and r["method"] == "HybridPPO"
            ]
            mat[i, j] = mean(float(r["feasible"]) for r in subset) if subset else float("nan")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(3))
    ax.set_xticklabels(bins)
    ax.set_yticks(range(3))
    ax.set_yticklabels(layouts)
    ax.set_title("HybridPPO charging-required feasibility (SynthCharge TEST)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "synthcharge" / "hybrid_charge_required_heatmap.png", dpi=150)
    plt.close(fig)

    reasons = sorted({r.get("reason") or "none" for r in synth if not r["feasible"]})
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(ALL_METHODS))
    for reason in reasons:
        heights = []
        for method in ALL_METHODS:
            fails = [r for r in method_rows(synth, method) if not r["feasible"] and (r.get("reason") or "none") == reason]
            denom = 5 if method in LEARNED else 1
            heights.append(len(fails) / denom)
        ax.bar(range(len(ALL_METHODS)), heights, bottom=bottom, label=reason)
        bottom += np.array(heights)
    ax.set_xticks(range(len(ALL_METHODS)))
    ax.set_xticklabels([m.replace("GreedyMinimumSufficientCharge", "GreedyMin").replace("GreedyFullCharge", "GreedyFull").replace("OneStepLookahead", "Lookahead") for m in ALL_METHODS], rotation=15)
    ax.set_ylabel("Failed routes (learned: mean per seed)")
    ax.set_title("SynthCharge TEST failure reasons")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "synthcharge" / "failure_reasons.png", dpi=150)
    plt.close(fig)


def write_report(synth_sum, legacy_sum, paired, freeze):
    hybrid = synth_sum["HybridPPO"]
    discrete = synth_sum["DiscretePPO"]
    charge_h = synth_sum["charging_required"]["HybridPPO"]
    charge_d = synth_sum["charging_required"]["DiscretePPO"]
    lines = [
        "# FINAL V2 EXPERIMENT REPORT",
        "",
        "This report distinguishes archived V1, V2 development evidence, frozen B2 methodology,",
        "final gold training, the fresh SynthCharge TEST, and the non-fresh legacy EVRPTW-GR challenge.",
        "",
        "## Provenance",
        "",
        f"- method freeze SHA: `{METHOD_FREEZE_SHA}`",
        f"- training execution SHA: `{TRAINING_EXECUTION_SHA}`",
        f"- evaluation/analysis git HEAD: `{git_head()}`",
        f"- CHECKPOINT_FREEZE sha256 (LF): `{lf_sha256(FINAL / 'CHECKPOINT_FREEZE.json')}`",
        f"- TEST_LOCK sha256 (LF): `{lf_sha256(FINAL / 'TEST_LOCK.json')}`",
        f"- environment: Python {platform.python_version()}, Torch CPU, PyVRP 0.14.0, frvcpy 0.1.1",
        "",
        "## Method",
        "",
        "Proposed method: **Hybrid-Action PPO (HybridPPO)**, development condition **B2**:",
        "energy-continuation SOC lower bound + optimistic time-feasibility SOC upper bound +",
        "one global TRAIN-only return scale.",
        "",
        "Learned comparator: DiscretePPO (same shield, same return scale, same budget; discrete charge amounts).",
        "Excluded from the final matrix: DDQN, SAC, AttentionPPO.",
        "",
        "Reward documentation (unchanged): successful training return = `-T_completion`;",
        "failed training return = `-(H - t0) - L_remaining`. Evaluation infeasible completion = `H`.",
        "",
        "## Development evidence (not TEST)",
        "",
        "Pre-freeze B0/B1/B2/B3 and B2_pyvrp diagnostics remain under `results/v2/diagnostics/`.",
        "They used TRAIN/VAL only and motivated freezing B2. They are not fresh TEST results.",
        "",
        "## Fresh SynthCharge TEST (90 routes, linear physics)",
        "",
        "Physics profile: `synthcharge_linear` (`energy_law=linear_distance`).",
        "This does **not** validate EVRPTW-GR gradient physics.",
        "",
        f"- HybridPPO feasibility (mean±SD across 5 seeds): {fmt(hybrid['feasibility_mean'])} ± {fmt(hybrid['feasibility_sd'])}",
        f"  95% hierarchical bootstrap CI: [{fmt(hybrid['feasibility_ci95'][0])}, {fmt(hybrid['feasibility_ci95'][1])}]",
        f"  per-seed: {hybrid['per_seed_feasibility']}",
        f"- DiscretePPO feasibility: {fmt(discrete['feasibility_mean'])} ± {fmt(discrete['feasibility_sd'])}",
        f"  95% CI: [{fmt(discrete['feasibility_ci95'][0])}, {fmt(discrete['feasibility_ci95'][1])}]",
        f"  per-seed: {discrete['per_seed_feasibility']}",
        f"- HybridPPO failure-retaining completion: {fmt(hybrid['completion_all_mean'])} ± {fmt(hybrid['completion_all_sd'])}",
        f"  95% CI: [{fmt(hybrid['completion_all_ci95'][0])}, {fmt(hybrid['completion_all_ci95'][1])}]",
        f"- DiscretePPO failure-retaining completion: {fmt(discrete['completion_all_mean'])} ± {fmt(discrete['completion_all_sd'])}",
        f"  95% CI: [{fmt(discrete['completion_all_ci95'][0])}, {fmt(discrete['completion_all_ci95'][1])}]",
        "",
        "### Charging-required subset (72 routes)",
        "",
        f"- HybridPPO feasibility: {fmt(charge_h['feasibility_mean'])} ± {fmt(charge_h['feasibility_sd'])}",
        f"- DiscretePPO feasibility: {fmt(charge_d['feasibility_mean'])} ± {fmt(charge_d['feasibility_sd'])}",
        "",
        "### Primary paired comparison (HybridPPO − DiscretePPO)",
        "",
    ]
    for t in paired["synthcharge"]:
        lines.append(
            f"- {t['family']}: effect={fmt(t['effect'], 4)}, "
            f"95% CI=[{fmt(t['ci95'][0], 4)}, {fmt(t['ci95'][1], 4)}], "
            f"p_raw={t['p_raw']:.4g}, p_Holm={t['p_holm']:.4g}, n={t['n_independent_units']}"
        )
    lines += [
        "",
        "Do not claim statistical superiority unless Holm-adjusted evidence supports it.",
        "",
        "## Legacy same-domain challenge (26 routes, NOT a fresh TEST)",
        "",
        "Gold-trained checkpoints only. Six historically consumed V1 TEST parents.",
        "Not used for model selection, tuning, or fresh-generalization claims.",
        "",
        f"- HybridPPO feasibility: {fmt(legacy_sum['HybridPPO']['feasibility_mean'])} ± {fmt(legacy_sum['HybridPPO']['feasibility_sd'])}",
        f"- DiscretePPO feasibility: {fmt(legacy_sum['DiscretePPO']['feasibility_mean'])} ± {fmt(legacy_sum['DiscretePPO']['feasibility_sd'])}",
        "",
        "## Limitations",
        "",
        "- SynthCharge uses linear energy; no EVRPTW-GR terrain/payload energy coupling.",
        "- Certificate search is a positive witness generator; timeout/exhaustion is unverified, not infeasible.",
        "- Legacy challenge has only six parents; low statistical power.",
        "- Five training seeds; seed variability is reported and must not be pooled as independent routes.",
        "- Stateless baselines are secondary references under the same V2 shield where applicable.",
        "",
        "## Artifacts",
        "",
        "- raw: `results/v2/final/raw/`",
        "- statistics: `results/v2/final/statistics/`",
        "- tables: `results/v2/final/tables/`",
        "- figures: `results/v2/final/figures/`",
        "- checkpoint freeze: `results/v2/final/CHECKPOINT_FREEZE.json`",
        "",
    ]
    (FINAL / "FINAL_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if method_tree_diff().strip():
        raise SystemExit("method tree differs from freeze")
    if not (FINAL / "CHECKPOINT_FREEZE.json").is_file():
        raise SystemExit("CHECKPOINT_FREEZE missing")
    if not (FINAL / "EVALUATION_CONSUMED.json").is_file():
        raise SystemExit("evaluation not yet consumed")

    synth = load_rows("synthcharge_test")
    legacy = load_rows("legacy_evrptwgr_challenge")
    freeze = load_json(FINAL / "CHECKPOINT_FREEZE.json")

    synth_audit = integrity(synth, n_routes=90, learned_factor=900, baseline_factor=270, name="synthcharge")
    legacy_audit = integrity(legacy, n_routes=26, learned_factor=260, baseline_factor=78, name="legacy")
    dump_json(STATS / "synthcharge" / "integrity_audit.json", synth_audit)
    dump_json(STATS / "legacy" / "integrity_audit.json", legacy_audit)

    synth_sum = {m: summarize_learned(synth, m) if m in LEARNED else summarize_baseline(synth, m) for m in ALL_METHODS}
    charge = [r for r in synth if r["charge_class"] == "charging_required"]
    nocharge = [r for r in synth if r["charge_class"] == "no_charge_required"]
    synth_sum["charging_required"] = {m: summarize_learned(charge, m) if m in LEARNED else summarize_baseline(charge, m) for m in ALL_METHODS}
    synth_sum["no_charge_required"] = {m: summarize_learned(nocharge, m) if m in LEARNED else summarize_baseline(nocharge, m) for m in ALL_METHODS}
    legacy_sum = {m: summarize_learned(legacy, m) if m in LEARNED else summarize_baseline(legacy, m) for m in ALL_METHODS}
    paired = {"synthcharge": paired_primary(synth, "SynthCharge fresh TEST"), "legacy": paired_primary(legacy, "legacy challenge")}

    # Tables
    header_a = [
        "method",
        "n_routes",
        "n_seeds",
        "mean_feasible_routes_per_seed",
        "feasibility_mean",
        "feasibility_sd",
        "feasibility_ci95_lo",
        "feasibility_ci95_hi",
        "completion_all_mean",
        "completion_all_sd",
        "completion_all_ci95_lo",
        "completion_all_ci95_hi",
    ]
    rows_a = []
    for method in ALL_METHODS:
        s = synth_sum[method]
        ci_f = s.get("feasibility_ci95") or [None, None]
        ci_c = s.get("completion_all_ci95") or [None, None]
        rows_a.append(
            [
                method,
                s["n_routes"],
                s["n_seeds"],
                fmt(s.get("mean_feasible_routes_per_seed"), 2),
                fmt(s["feasibility_mean"]),
                fmt(s.get("feasibility_sd")),
                fmt(ci_f[0]),
                fmt(ci_f[1]),
                fmt(s["completion_all_mean"]),
                fmt(s.get("completion_all_sd")),
                fmt(ci_c[0]),
                fmt(ci_c[1]),
            ]
        )
    write_csv(TABLES / "synthcharge" / "table_a_primary.csv", header_a, rows_a)
    write_md(
        TABLES / "synthcharge" / "table_a_primary.md",
        "SynthCharge Table A — Primary performance",
        header_a,
        rows_a,
        "Fresh TEST, 90 routes. Learned means/SD across 5 seeds. CIs are hierarchical bootstrap (seed then base_instance).",
    )

    header_b = ["method", "n_routes", "feasibility_mean", "feasibility_sd", "completion_all_mean", "charging_time", "station_visits"]
    rows_b = []
    for method in ALL_METHODS:
        s = synth_sum["charging_required"][method]
        op = operational(charge, method)
        rows_b.append(
            [
                method,
                72,
                fmt(s["feasibility_mean"]),
                fmt(s.get("feasibility_sd")),
                fmt(s["completion_all_mean"]),
                fmt(op.get("charging_time")),
                fmt(op.get("station_visits"), 2),
            ]
        )
    write_csv(TABLES / "synthcharge" / "table_b_charging_required.csv", header_b, rows_b)
    write_md(TABLES / "synthcharge" / "table_b_charging_required.md", "SynthCharge Table B — Charging-required", header_b, rows_b, "72 charging-required TEST routes.")

    header_c = ["method", "feasible_only_completion", "charging_time", "station_visits", "travel_time", "waiting_time", "terminal_soc"]
    rows_c = []
    for method in ALL_METHODS:
        op = operational(synth, method)
        rows_c.append([method, fmt(op.get("feasible_only_completion")), fmt(op.get("charging_time")), fmt(op.get("station_visits"), 2), fmt(op.get("travel_time")), fmt(op.get("waiting_time")), fmt(op.get("terminal_soc"))])
    write_csv(TABLES / "synthcharge" / "table_c_feasible_only.csv", header_c, rows_c)
    write_md(TABLES / "synthcharge" / "table_c_feasible_only.md", "SynthCharge Table C — Feasible-only operational", header_c, rows_c, "Secondary metrics; failures dropped.")

    header_d = ["layout", "length_bin", "charge_class", "method", "n_routes", "feasibility", "completion_all", "charging_time", "station_visits"]
    rows_d = cell_breakdown(synth)
    write_csv(TABLES / "synthcharge" / "table_d_cell_breakdown.csv", header_d, rows_d)
    write_md(TABLES / "synthcharge" / "table_d_cell_breakdown.md", "SynthCharge Table D — Cell breakdown", header_d, rows_d, "Descriptive only; no post-hoc significance claims.")

    header_e = ["benchmark", "family", "comparison", "effect", "ci95_lo", "ci95_hi", "p_raw", "p_holm", "n_units", "test"]
    rows_e = []
    for t in paired["synthcharge"]:
        rows_e.append([t["benchmark"], t["family"], t["comparison"], fmt(t["effect"], 4), fmt(t["ci95"][0], 4), fmt(t["ci95"][1], 4), f"{t['p_raw']:.4g}", f"{t['p_holm']:.4g}", t["n_independent_units"], t["test"]])
    write_csv(TABLES / "synthcharge" / "table_e_paired.csv", header_e, rows_e)
    write_md(TABLES / "synthcharge" / "table_e_paired.md", "SynthCharge Table E — Paired HybridPPO vs DiscretePPO", header_e, rows_e, "Primary family; Holm within two comparisons.")

    header_l = header_a
    rows_l = []
    for method in ALL_METHODS:
        s = legacy_sum[method]
        ci_f = s.get("feasibility_ci95") or [None, None]
        ci_c = s.get("completion_all_ci95") or [None, None]
        rows_l.append(
            [
                method,
                s["n_routes"],
                s["n_seeds"],
                fmt(s.get("mean_feasible_routes_per_seed"), 2),
                fmt(s["feasibility_mean"]),
                fmt(s.get("feasibility_sd")),
                fmt(ci_f[0]),
                fmt(ci_f[1]),
                fmt(s["completion_all_mean"]),
                fmt(s.get("completion_all_sd")),
                fmt(ci_c[0]),
                fmt(ci_c[1]),
            ]
        )
    write_csv(TABLES / "legacy" / "table_legacy_challenge.csv", header_l, rows_l)
    write_md(
        TABLES / "legacy" / "table_legacy_challenge.md",
        "Legacy same-domain challenge (NOT a fresh TEST)",
        header_l,
        rows_l,
        "26 official skeletons of six historically consumed V1 TEST parents. Gold-trained models only.",
    )

    # per-seed tables
    for method in LEARNED:
        rows = [[seed, fmt(synth_sum[method]["per_seed_feasibility"][seed])] for seed in SEEDS]
        write_csv(STATS / "synthcharge" / f"per_seed_{method}.csv", ["seed", "feasibility"], rows)
        dump_json(STATS / "synthcharge" / f"per_seed_{method}.json", synth_sum[method])
        rows = [[seed, fmt(legacy_sum[method]["per_seed_feasibility"][seed])] for seed in SEEDS]
        write_csv(STATS / "legacy" / f"per_seed_{method}.csv", ["seed", "feasibility"], rows)
        dump_json(STATS / "legacy" / f"per_seed_{method}.json", legacy_sum[method])

    dump_json(STATS / "synthcharge" / "method_summary.json", {k: v for k, v in synth_sum.items() if k in ALL_METHODS})
    dump_json(STATS / "legacy" / "method_summary.json", legacy_sum)
    write_csv(STATS / "synthcharge" / "paired_holm.csv", header_e, rows_e)
    dump_json(STATS / "synthcharge" / "paired_holm.json", paired["synthcharge"])
    write_csv(STATS / "synthcharge" / "cell_breakdown.csv", header_d, rows_d)
    dump_json(STATS / "synthcharge" / "charging_required_summary.json", synth_sum["charging_required"])
    dump_json(STATS / "synthcharge" / "failure_reasons.json", failure_analysis(synth))
    dump_json(STATS / "legacy" / "failure_reasons.json", failure_analysis(legacy))

    figures(synth, legacy)
    write_report(synth_sum, legacy_sum, paired, freeze)

    # ENVIRONMENT + PROVENANCE (results archive; no self-hash)
    import importlib.metadata as metadata

    def ver(name):
        try:
            return metadata.version(name)
        except Exception:
            return "missing"

    dump_json(
        FINAL / "ENVIRONMENT.json",
        {
            "python": platform.python_version(),
            "os": platform.platform(),
            "machine": platform.machine(),
            "torch": ver("torch"),
            "numpy": ver("numpy"),
            "pyvrp": ver("pyvrp"),
            "frvcpy": ver("frvcpy"),
            "matplotlib": ver("matplotlib"),
            "cuda": False,
            "device": "cpu",
            "git_head": git_head(),
            "method_freeze_sha": METHOD_FREEZE_SHA,
            "training_execution_sha": TRAINING_EXECUTION_SHA,
        },
    )
    dump_json(
        FINAL / "PROVENANCE.json",
        {
            "v1_paper_code_sha": "a175ee43548a5d2d5154a9ae0731a01f7642a7f6",
            "v1_final_analysis_sha": "23d3d9c7de87a9486160c9937f8b385dd4833d57",
            "v2_development_sha": "aaa4a1a322086410ff359834861704dca19138ac",
            "v2_method_freeze_sha": METHOD_FREEZE_SHA,
            "v2_data_freeze_sha": freeze["data_freeze_sha"],
            "v2_training_execution_sha": TRAINING_EXECUTION_SHA,
            "v2_results_archive_sha": "commit that adds this file; not self-hashed",
            "checkpoint_freeze_sha256_lf": lf_sha256(FINAL / "CHECKPOINT_FREEZE.json"),
            "test_lock_sha256_lf": lf_sha256(FINAL / "TEST_LOCK.json"),
            "n_learned_checkpoints": 20,
            "synthcharge_raw_rows": len(synth),
            "legacy_raw_rows": len(legacy),
            "absolute_windows_paths": False,
        },
    )

    # concise PAPER_FACTS
    (FINAL / "PAPER_FACTS.md").write_text(
        "\n".join(
            [
                "# PAPER FACTS (claims supported by final evidence only)",
                "",
                "## DEVELOPMENT (not TEST)",
                "- B2 was frozen from gold TRAIN/VAL diagnostics (seeds 42/43).",
                "",
                "## FRESH SynthCharge TEST",
                f"- HybridPPO feasibility mean across 5 seeds: {fmt(synth_sum['HybridPPO']['feasibility_mean'])}",
                f"- DiscretePPO feasibility mean across 5 seeds: {fmt(synth_sum['DiscretePPO']['feasibility_mean'])}",
                f"- HybridPPO charging-required feasibility: {fmt(synth_sum['charging_required']['HybridPPO']['feasibility_mean'])}",
                f"- DiscretePPO charging-required feasibility: {fmt(synth_sum['charging_required']['DiscretePPO']['feasibility_mean'])}",
                f"- HybridPPO failure-retaining completion: {fmt(synth_sum['HybridPPO']['completion_all_mean'])}",
                f"- DiscretePPO failure-retaining completion: {fmt(synth_sum['DiscretePPO']['completion_all_mean'])}",
                "- See table_e_paired.md for Holm-adjusted HybridPPO vs DiscretePPO tests.",
                "",
                "## LEGACY same-domain challenge (not fresh)",
                f"- HybridPPO feasibility: {fmt(legacy_sum['HybridPPO']['feasibility_mean'])}",
                f"- DiscretePPO feasibility: {fmt(legacy_sum['DiscretePPO']['feasibility_mean'])}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("analysis complete")


if __name__ == "__main__":
    main()
