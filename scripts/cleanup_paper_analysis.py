"""Analysis-only cleanup of final paper artifacts. Never retrains or reopens TEST."""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.batch import read_jsonl_dicts  # noqa: E402
from experiments.provenance import frozen_hashes  # noqa: E402
from experiments.stats import (  # noqa: E402
    attach_feas_rate,
    exclude_non_paper_methods,
    logical_checkpoint_path,
    map_ablation_method,
    matched_terrain_route_ids,
    parent_balanced_mean,
    route_weighted_mean,
    seed_feasibility_summary,
)
from routing.serialize import canonical_dumps  # noqa: E402

PAPER_CODE_SHA = "a175ee43548a5d2d5154a9ae0731a01f7642a7f6"
RESULTS_ANALYSIS_CODE_SHA = "7735be30a874517d26b977051159102a3c1190b9"
FINAL = ROOT / "results" / "final"
MAIN_METHODS = (
    "GreedyMinimumSufficientCharge",
    "GreedyFullCharge",
    "OneStepLookahead",
    "HybridPPO",
    "DiscretePPO",
    "AttentionPPO",
)
PINNED_HASHES = {
    "corpus.jsonl": "1796d817ff058fe0559021ab79ca97087ef3646c57744c2c883ecc37e85451e0",
    "manifest.csv": "41bb8be3245f8f3b0800db4ba74cb19a2003236b6fbd0dbc1237e7cdfaee104d",
    "corpus_metadata.json": "1ca8725199842f0a8d4a4e6fac2a31931ea3c9aaca833c2d253c6147469d9a96",
    "train.json": "3aecd73e9472b8572e899a7e2da3f1f874f0e5a13250f5cd6d76d812002913ce",
    "validation.json": "1379aef7e92c308f90ab78674dec9994db72078c96cb1020ecc49d33eef69af4",
    "test.json": "9bf39fbf1f27aa57146879fb71922b0216f24f1eb37a4fd9a9a43b44b6c4b6d0",
    "split_metadata.json": "4740ab71737bfd9c04d37f258e57bfa4854a526da7c03b218ffe877818128900",
}


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def quarantine_nonlinear() -> None:
    src = FINAL / "raw" / "final_montoya_nonlinear.jsonl"
    dest_dir = FINAL / "excluded" / "invalid_external_sensitivity"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "README.md").write_text(
        "# Excluded from the manuscript\n\n"
        "Native Montoya/FRVCP nonlinear charging output is **invalid_external_sensitivity**.\n"
        "Records include `feasible=true`, `reason=optimal`, and non-finite durations.\n"
        "Do not treat this as completed scientific sensitivity analysis.\n"
        "The original bytes are preserved as `final_montoya_nonlinear.jsonl.rawdebug`.\n"
        "The sanitized copy serializes non-finite numbers as null and sets `duration_nonfinite=true`.\n",
        encoding="utf-8",
    )
    if src.is_file():
        rawdebug = dest_dir / "final_montoya_nonlinear.jsonl.rawdebug"
        shutil.copy2(src, rawdebug)
        sanitized = []
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=lambda token: token)
            for key in ("route_completion_time", "completion_time_all_routes", "optimality_gap_percent"):
                value = row.get(key)
                nonfinite = value in {"Infinity", "-Infinity", "NaN"} or (
                    isinstance(value, float) and (value != value or value in (float("inf"), float("-inf")))
                )
                if nonfinite:
                    row[key] = None
                    row["duration_nonfinite"] = True
                    row["scientific_status"] = "invalid_external_sensitivity"
            sanitized.append(canonical_dumps(row))
        (dest_dir / "final_montoya_nonlinear.sanitized.jsonl").write_text(
            "\n".join(sanitized) + "\n", encoding="utf-8"
        )
        src.unlink()
    tables = FINAL / "tables" / "nonlinear_sensitivity"
    figures = FINAL / "figures" / "nonlinear_sensitivity"
    stats = FINAL / "statistics" / "nonlinear_sensitivity"
    for path in (tables, figures, stats):
        if path.is_dir():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "EXCLUDED.md").write_text(
            "Excluded from the manuscript: invalid_external_sensitivity.\n",
            encoding="utf-8",
        )


def sanitize_paths() -> None:
    freeze_path = FINAL / "manifests" / "checkpoint_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for rec in freeze.get("records", []):
        if rec.get("folder"):
            rec["folder"] = logical_checkpoint_path(rec["folder"])
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    for manifest in (FINAL / "training").rglob("manifest.json"):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("checkpoint"):
            payload["checkpoint"] = logical_checkpoint_path(payload["checkpoint"])
        manifest.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")


def _load_raw() -> list[dict]:
    rows = []
    for path in sorted((FINAL / "raw").glob("*.jsonl")):
        for row in read_jsonl_dicts(path):
            row["_source"] = path.name
            rows.append(row)
    return exclude_non_paper_methods(rows)


def known_feasible_diagnostic(rows: list[dict]) -> dict:
    exact = [row for row in rows if row.get("scenario") == "exact_small"]
    proven = [row for row in exact if row.get("reason") == "optimal_for_action_set"]
    ids = {row.get("route_id") for row in proven}
    main = [row for row in rows if row.get("scenario") == "main_test" and row.get("route_id") in ids]
    by_method = defaultdict(list)
    for row in main:
        by_method[str(row.get("method"))].append(row)
    methods = {}
    for method, group in by_method.items():
        tagged = attach_feas_rate(group)
        seeds = seed_feasibility_summary(group)
        methods[method] = {
            "n_episode_rows": len(group),
            "n_known_feasible_routes": len(ids),
            "route_weighted_feasibility": route_weighted_mean(tagged, "feas_rate"),
            "mean_feasible_routes_per_seed": seeds.get("mean_feasible_routes_per_seed"),
            "per_seed": seeds.get("per_seed"),
        }
    payload = {
        "status": "diagnostic_only",
        "exact_global": False,
        "action_set": "CONTINUE + {continuation-minimum SOC, maximum SOC}",
        "n_eligible_small_routes": len(exact),
        "n_proven_feasible_restricted": len(proven),
        "known_feasible_route_ids": sorted(ids),
        "methods": methods,
        "note": (
            "17/34 eligible <=5-customer routes have a proven feasible solution under the "
            "restricted action set. A miss on those routes is policy failure. "
            "The remaining 17 are not established as infeasible for continuous Hybrid PPO "
            "(timeout or restricted-infeasible only)."
        ),
    }
    dest = FINAL / "statistics" / "exact_small"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "known_feasible_diagnostic.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_report(rows: list[dict], diagnostic: dict, n_groups: int) -> None:
    main = [row for row in rows if row.get("scenario") == "main_test"]
    by_method = defaultdict(list)
    for row in main:
        by_method[str(row.get("method"))].append(row)
    ablation = [{**row, "method": map_ablation_method(row.get("method"))} for row in rows if row.get("scenario") == "ablation"]
    ab_groups = defaultdict(list)
    for row in ablation:
        ab_groups[row["method"]].append(row)
    hybrid_seed = seed_feasibility_summary(by_method["HybridPPO"])
    large = [row for row in by_method["HybridPPO"] if row.get("network_group") == "Large_Network"]
    large_feas = route_weighted_mean(attach_feas_rate(large), "feas_rate")
    paired = (FINAL / "statistics" / "main_test" / "paired_holm.csv").read_text(encoding="utf-8")
    holm_all_one = all(
        (parts[7] == "1.0" if len(parts) > 7 else False)
        for line in paired.splitlines()[1:]
        if (parts := line.split(","))
    )
    lines = [
        "# Final experiment report",
        "",
        "Generated from tracked raw records after analysis-only cleanup. No retraining. TEST was not reopened for tuning.",
        "",
        "## A. Frozen methodology and provenance",
        "",
        f"- paper_code_sha: `{PAPER_CODE_SHA}`",
        f"- results_analysis_code_sha: `{RESULTS_ANALYSIS_CODE_SHA}`",
        "- cleanup_commit_sha: the git commit that contains this cleanup; newer than `results_analysis_code_sha`; not written into this file.",
        "- Device: CPU. DDQN excluded. Methodology frozen. TEST is consumed.",
        "",
        "## B–P. See generated tables",
        "",
        "Main TEST population remains 49 routes × declared methods. Failures retained. H used for infeasible completion.",
        f"Matched terrain sibling groups on TEST: **{n_groups}** complete L/NL/VG groups. Unmatched NL-only Medium/Large routes are excluded from terrain-effect claims.",
        "",
        "### HybridPPO seed 46",
        "",
        "Seed 46 exhibited late-training collapse (validation feasibility reached 0 at updates 220–260). "
        "The selected checkpoint is update 60, which had nonzero validation feasibility. "
        "Predeclared validation checkpoint selection prevented the collapsed final policy from being used for TEST. Seed 46 was not retrained.",
        "",
        "### Per-seed HybridPPO TEST feasibility (49 routes each; not a pooled unique-route count)",
        "",
    ]
    for seed, stats in hybrid_seed["per_seed"].items():
        lines.append(
            f"- seed {seed}: {stats['n_feasible_routes']}/49 route-weighted feas. {stats['route_weighted_feasibility']:.4f}; "
            f"parent-balanced {stats['parent_balanced_feasibility']:.4f}"
        )
    lines += [
        "",
        f"Mean feasible routes per seed: {hybrid_seed['mean_feasible_routes_per_seed']:.2f}/49 "
        f"(SD {hybrid_seed['sd_feasible_routes_per_seed']:.2f}). "
        f"Pooled 25/245 is 25 seed-route episodes, not 25 unique TEST routes.",
        "",
        "### Known-feasible small-route diagnostic (not model selection)",
        "",
        f"{diagnostic['n_proven_feasible_restricted']} / {diagnostic['n_eligible_small_routes']} eligible ≤5-customer TEST routes "
        "have a proven feasible solution under the restricted {continuation-minimum, full} action set. "
        "This is not globally exact for continuous Hybrid PPO.",
        "",
    ]
    for method in MAIN_METHODS:
        block = diagnostic["methods"].get(method, {})
        if not block:
            continue
        lines.append(
            f"- {method} on those {diagnostic['n_proven_feasible_restricted']} routes: "
            f"route-weighted feas. {block.get('route_weighted_feasibility')}; "
            f"mean feasible routes/seed {block.get('mean_feasible_routes_per_seed')}."
        )
    lines += [
        "",
        "A miss on a restricted-proven-feasible route is **policy failure**. "
        "Timeout or restricted-infeasible rows do **not** establish continuous-action infeasibility.",
        "",
        "### Nonlinear sensitivity",
        "",
        "The Montoya/native nonlinear experiment is **excluded_from_paper** / `invalid_external_sensitivity`. "
        "It is not a completed scientific sensitivity analysis.",
        "",
        f"Large_Network HybridPPO TEST feasibility: {large_feas}.",
        "",
        "## Q. Limitations that must appear in the manuscript",
        "",
        "- HybridPPO does not show a statistically significant advantage over the declared comparators.",
        "- All Holm-adjusted main-comparison p-values are non-significant"
        + (" (all Holm p = 1.0)." if holm_all_one else "."),
        "- Feasibility is low across all methods.",
        "- All learned methods have 0% feasibility on Large_Network TEST routes.",
        "- Some restricted-search-proven feasible small routes are still missed by HybridPPO.",
        "- Ablations do not show clear support for every architectural component.",
        "- A5/pooling is weaker descriptively, but evidence is limited.",
        "- TEST has only six parent clusters.",
        "- Do not describe HybridPPO as best, superior, or state of the art.",
        "",
        "## R. Manuscript-ready tables",
        "",
        "- `results/final/tables/main_test/table_A_completion_time.md`",
        "- `results/final/tables/main_test/table_A_per_seed_feasibility.md`",
        "- `results/final/tables/main_test/table_B_feasible_only.md`",
        "- `results/final/tables/main_test/table_C_terrain.md`",
        "- `results/final/tables/main_test/table_D_size_family.md`",
        "- `results/final/tables/ablation/table_E_ablations.md`",
        "- `results/final/tables/frvcpy_native/table_F_frvcpy.md`",
        "- `results/final/tables/main_test/table_G_runtime.md`",
        "- `results/final/tables/soc_reserve/table_H_soc_reserve.md`",
        "- `results/final/tables/main_test/table_I_failure_reasons.md`",
        "- `results/final/statistics/main_test/paired_holm.csv`",
        "- `results/final/statistics/exact_small/known_feasible_diagnostic.json`",
        "",
        "## S. Manuscript-ready figures",
        "",
        "- `results/final/figures/main_test/hybrid_ppo_val_curves.png`",
        "- `results/final/figures/main_test/feasibility_bars.png`",
        "- `results/final/figures/main_test/completion_all_bars.png`",
        "- `results/final/figures/main_test/terrain_comparison.png`",
        "- `results/final/figures/main_test/hybrid_failure_reasons.png`",
        "- `results/final/figures/ablation/ablation_comparison.png`",
        "- `results/final/figures/soc_reserve/soc_reserve.png`",
        "- `results/final/figures/frvcpy_native/frvcp_gap_hist.png`",
        "",
        "Nonlinear figures are excluded.",
        "",
        "## T. Warnings",
        "",
        "1. No post-TEST tuning is allowed. TEST is consumed.",
        "2. Restricted search is not globally exact.",
        "3. frvcpy is not exact for EVRPTW-GR.",
        "4. SOC reserve is not part of main_test sample sizes.",
        "5. Nonlinear Montoya output is invalid and excluded.",
        "6. Terrain claims use matched L/NL/VG siblings only.",
        "7. Parent-balanced CIs must not be labeled as route-weighted.",
        "8. Legacy DDQN is not a paper method.",
        "",
    ]
    (FINAL / "FINAL_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance() -> None:
    import torch

    configs = {}
    for name in ("hybrid_ppo.toml", "discrete_ppo.toml", "attention_ppo.toml"):
        path = ROOT / "configs" / "rl" / name
        import hashlib

        configs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    freeze = json.loads((FINAL / "manifests" / "checkpoint_freeze.json").read_text(encoding="utf-8"))
    ckpt = {}
    for rec in freeze.get("records", []):
        ckpt[f"{rec['method']}/seed_{rec['seed']}"] = {
            "logical_folder": rec.get("folder"),
            "best_pt_sha256": rec.get("best_pt_sha256") or rec.get("checkpoint_sha256"),
            "best_update": rec.get("best_update"),
        }
    payload = {
        "paper_code_sha": PAPER_CODE_SHA,
        "results_analysis_code_sha": RESULTS_ANALYSIS_CODE_SHA,
        "note": "cleanup_commit_sha is the later commit that contains this cleanup file and is not recorded here.",
        "corpus_hashes": frozen_hashes(),
        "pinned_hashes": PINNED_HASHES,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "pyvrp_version": _pkg_version("pyvrp"),
        "frvcpy_version": _pkg_version("frvcpy"),
        "device": "cpu",
        "paper_seeds": [42, 43, 44, 45, 46],
        "ablation_seeds": [42, 43, 44],
        "configs_sha256": configs,
        "checkpoint_sha256": ckpt,
        "ddqn_included": False,
        "nonlinear_sensitivity": "invalid_external_sensitivity / excluded_from_paper",
        "command_log": [
            "analysis-only cleanup; no training; no TEST re-evaluation",
            "python scripts/cleanup_paper_analysis.py",
        ],
    }
    (FINAL / "PROVENANCE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def assert_tracked_json_finite() -> None:
    bad = []
    for path in FINAL.rglob("*"):
        if not path.is_file():
            continue
        if "excluded" in path.parts and path.suffix == ".rawdebug":
            continue
        if path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bInfinity\b|\bNaN\b", text):
            bad.append(str(path.relative_to(ROOT)))
    if bad:
        raise SystemExit("non-finite tokens remain in tracked JSON:\n" + "\n".join(bad))


def main() -> int:
    quarantine_nonlinear()
    sanitize_paths()
    scenarios = ("main_test", "ablation", "soc_reserve", "exact_small", "frvcpy_native")
    for scenario in scenarios:
        subprocess.check_call(
            [sys.executable, "scripts/analyze_results.py", "--raw", str(FINAL / "raw"), "--scenario", scenario, "--out", str(FINAL / "statistics" / scenario)],
            cwd=ROOT,
        )
        subprocess.check_call(
            [sys.executable, "scripts/make_tables.py", "--raw", str(FINAL / "raw"), "--scenario", scenario, "--out", str(FINAL / "tables")],
            cwd=ROOT,
        )
        fig = [sys.executable, "scripts/make_figures.py", "--raw", str(FINAL / "raw"), "--scenario", scenario, "--out", str(FINAL / "figures")]
        if scenario == "main_test":
            fig += ["--curves-root", str(FINAL / "training" / "HybridPPO")]
        subprocess.check_call(fig, cwd=ROOT)
    subprocess.check_call(
        [sys.executable, "scripts/make_tables.py", "--raw", str(FINAL / "raw"), "--scenario", "nonlinear_sensitivity", "--out", str(FINAL / "tables")],
        cwd=ROOT,
    )
    rows = _load_raw()
    diagnostic = known_feasible_diagnostic(rows)
    _, n_groups = matched_terrain_route_ids(split="test")
    write_report(rows, diagnostic, n_groups)
    write_provenance()
    assert_tracked_json_finite()
    print(f"CLEANUP_DONE matched_terrain_groups={n_groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
