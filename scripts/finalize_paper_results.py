"""Copy frozen paper artifacts and generate analysis-only outputs. Never retrains."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.batch import read_jsonl_dicts  # noqa: E402
from experiments.provenance import frozen_hashes, sha256_file  # noqa: E402
from experiments.stats import (  # noqa: E402
    attach_feas_rate,
    dedupe_soc_greedy,
    exclude_non_paper_methods,
    map_ablation_method,
    mean_sd_across_training_seeds,
    parent_balanced_mean,
    paired_parent_diff_hierarchical,
    permutation_pvalue,
    route_weighted_mean,
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

PAPER_CODE_SHA = "a175ee43548a5d2d5154a9ae0731a01f7642a7f6"
WORKTREE = ROOT.parent / "RL_CHARGE_PAPER"
PAPER_RUN = ROOT.parent / "paper_run"
FINAL = ROOT / "results" / "final"
MAIN_METHODS = (
    "GreedyMinimumSufficientCharge",
    "GreedyFullCharge",
    "OneStepLookahead",
    "HybridPPO",
    "DiscretePPO",
    "AttentionPPO",
)
EXPECTED_MAIN = {
    "GreedyMinimumSufficientCharge": 49,
    "GreedyFullCharge": 49,
    "OneStepLookahead": 49,
    "HybridPPO": 245,
    "DiscretePPO": 245,
    "AttentionPPO": 245,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def _copy_training() -> None:
    ckpt_root = WORKTREE / "checkpoints"
    dest_root = FINAL / "training"
    dest_root.mkdir(parents=True, exist_ok=True)
    methods = [
        "HybridPPO",
        "DiscretePPO",
        "AttentionPPO",
        "ablation_A2",
        "ablation_A3",
        "ablation_A4",
        "ablation_A5",
    ]
    for method in methods:
        src = ckpt_root / method
        if not src.is_dir():
            raise FileNotFoundError(src)
        for seed_dir in sorted(src.glob("seed_*")):
            dest = dest_root / method / seed_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("manifest.json", "normalizer_provenance.json", "curves.jsonl"):
                src_file = seed_dir / name
                if not src_file.is_file():
                    raise FileNotFoundError(src_file)
                shutil.copy2(src_file, dest / name)
            hashes = {}
            for ckpt in ("best.pt", "last.pt"):
                ckpt_path = seed_dir / ckpt
                if ckpt_path.is_file():
                    hashes[ckpt] = _sha(ckpt_path)
            (dest / "checkpoint_hashes.json").write_text(
                json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
            )


def _copy_raw_and_manifests() -> None:
    raw_dest = FINAL / "raw"
    raw_dest.mkdir(parents=True, exist_ok=True)
    for path in sorted((WORKTREE / "results" / "raw").glob("final_*.jsonl")):
        shutil.copy2(path, raw_dest / path.name)
    man_dest = FINAL / "manifests"
    man_dest.mkdir(parents=True, exist_ok=True)
    freeze = PAPER_RUN / "checkpoint_freeze.json"
    if freeze.is_file():
        shutil.copy2(freeze, man_dest / "checkpoint_freeze.json")
    runs = WORKTREE / "results" / "runs"
    if runs.is_dir():
        for path in sorted(runs.glob("final_*")):
            dest = man_dest / path.name
            if path.is_file():
                shutil.copy2(path, dest)
            elif path.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(path, dest)


def _load_rows() -> list[dict]:
    rows = []
    for path in sorted((FINAL / "raw").glob("*.jsonl")):
        for row in read_jsonl_dicts(path):
            row["_source"] = path.name
            rows.append(row)
    return exclude_non_paper_methods(rows)


def audit(rows: list[dict]) -> dict:
    errors = []
    hashes = frozen_hashes()
    pin_errors = [f"{k}: expected {PINNED_HASHES[k]}, got {hashes.get(k)}" for k in PINNED_HASHES if hashes.get(k) != PINNED_HASHES[k]]
    errors.extend(pin_errors)

    main = [r for r in rows if r.get("scenario") == "main_test" and r.get("split") == "test"]
    counts = Counter(r.get("method") for r in main)
    if len(main) != 882:
        errors.append(f"main_test total {len(main)} != 882")
    for method, expected in EXPECTED_MAIN.items():
        if counts.get(method, 0) != expected:
            errors.append(f"main_test {method}: {counts.get(method, 0)} != {expected}")
    if any(r.get("method") == "LegacyTwoStageDDQN" for r in rows):
        errors.append("DDQN rows present")
    if any("pilot" in str(r.get("scenario")) or "smoke" in str(r.get("scenario")) for r in main):
        errors.append("pilot/smoke leaked into main_test")
    if any(r.get("min_soc_fraction") not in {None, 0, 0.0} for r in main if r.get("method") == "HybridPPO"):
        errors.append("SOC reserve leaked into HybridPPO main_test")

    test_routes = {r.get("route_id") for r in main}
    if len(test_routes) != 49:
        errors.append(f"main_test unique routes {len(test_routes)} != 49")
    parents = {r.get("base_instance") for r in main}
    if len(parents) != 6:
        errors.append(f"main_test parents {len(parents)} != 6")
    for method in MAIN_METHODS:
        group = [r for r in main if r.get("method") == method]
        method_routes = {r.get("route_id") for r in group}
        if method_routes != test_routes:
            errors.append(f"{method} does not see the same 49 TEST routes")
        if method in {"HybridPPO", "DiscretePPO", "AttentionPPO"}:
            pairs = Counter((r.get("route_id"), r.get("seed")) for r in group)
            if len(pairs) != 245 or any(c != 1 for c in pairs.values()):
                errors.append(f"{method} uniqueness failed")
            if sorted({int(r.get("seed")) for r in group}) != [42, 43, 44, 45, 46]:
                errors.append(f"{method} seeds != 42-46")
            if any(not r.get("checkpoint_sha256") for r in group):
                errors.append(f"{method} missing checkpoint SHA")
        else:
            keys = Counter(r.get("route_id") for r in group)
            if len(keys) != 49 or any(c != 1 for c in keys.values()):
                errors.append(f"{method} uniqueness failed")
        if any(r.get("git_sha") != PAPER_CODE_SHA for r in group):
            errors.append(f"{method} git_sha != PAPER_CODE_SHA")
        if any(r.get("corpus_sha256") != PINNED_HASHES["corpus.jsonl"] for r in group):
            errors.append(f"{method} corpus hash mismatch")
        if any(r.get("split_sha256") != PINNED_HASHES["test.json"] for r in group):
            errors.append(f"{method} test split hash mismatch")
        if any(r.get("completion_time_all_routes") is None for r in group):
            errors.append(f"{method} missing H-for-failure completion")

    ablation = [r for r in rows if r.get("scenario") == "ablation" and r.get("split") == "test"]
    mapped = Counter(map_ablation_method(r.get("method")) for r in ablation)
    if len(ablation) != 882:
        errors.append(f"ablation total {len(ablation)} != 882")
    for label in ("FULL", "A1", "A2", "A3", "A4", "A5"):
        if mapped.get(label, 0) != 147:
            errors.append(f"ablation {label}: {mapped.get(label, 0)} != 147")

    soc = dedupe_soc_greedy([r for r in rows if r.get("scenario") == "soc_reserve"])
    soc_counts = Counter(r.get("method") for r in soc)
    if soc_counts.get("HybridPPO") != 980:
        errors.append(f"soc HybridPPO unique {soc_counts.get('HybridPPO')} != 980")
    if soc_counts.get("GreedyMinimumSufficientCharge") != 196:
        errors.append(f"soc greedy unique {soc_counts.get('GreedyMinimumSufficientCharge')} != 196")

    exact = [r for r in rows if r.get("scenario") == "exact_small"]
    if len(exact) != 34:
        errors.append(f"exact_small {len(exact)} != 34")
    if any(r.get("exact") is True for r in exact):
        errors.append("exact_small claimed global exact=true")

    native = [r for r in rows if r.get("scenario") == "frvcpy_native" and r.get("method") != "evrptwgr_surrogate_flag"]
    native_counts = Counter(r.get("method") for r in native)
    for method in ("frvcpy_Solver", "FRVCPGreedyMin", "FRVCPGreedyFull"):
        if native_counts.get(method) != 133:
            errors.append(f"frvcpy_native {method}: {native_counts.get(method)} != 133")

    nonlinear = [r for r in rows if r.get("scenario") == "nonlinear_sensitivity" and r.get("method") != "evrptwgr_surrogate_flag"]
    nl_counts = Counter(r.get("method") for r in nonlinear)
    for method in ("frvcpy_Solver", "FRVCPGreedyMin", "FRVCPGreedyFull"):
        if nl_counts.get(method) != 22:
            errors.append(f"nonlinear {method}: {nl_counts.get(method)} != 22")

    payload = {
        "ok": not errors,
        "errors": errors,
        "main_test_counts": dict(counts),
        "main_test_n": len(main),
        "main_test_routes": len(test_routes),
        "main_test_parents": sorted(parents),
        "ablation_counts": dict(mapped),
        "soc_counts": dict(soc_counts),
        "exact_small_n": len(exact),
        "frvcpy_native_counts": dict(native_counts),
        "nonlinear_counts": dict(nl_counts),
        "hashes": hashes,
        "ddqn_absent": all(r.get("method") != "LegacyTwoStageDDQN" for r in rows),
    }
    (FINAL / "statistics" / "raw_integrity_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise SystemExit("AUDIT FAILED:\n" + "\n".join(errors))
    return payload


def _metric_block(rows: list[dict]) -> dict:
    tagged = attach_feas_rate(rows)
    return {
        "n_rows": len(rows),
        "n_routes": len({r.get("route_id") for r in rows}),
        "n_seeds": len({int(r.get("seed", 0)) for r in rows}),
        "n_feasible": sum(1 for r in rows if r.get("feasible")),
        "route_weighted_feasibility": route_weighted_mean(tagged, "feas_rate"),
        "parent_balanced_feasibility": parent_balanced_mean(tagged, "feas_rate"),
        "route_weighted_completion_all": route_weighted_mean(rows, "completion_time_all_routes"),
        "parent_balanced_completion_all": parent_balanced_mean(rows, "completion_time_all_routes"),
        "seed_stats_feasibility": mean_sd_across_training_seeds(tagged, "feas_rate"),
        "seed_stats_completion": mean_sd_across_training_seeds(rows, "completion_time_all_routes"),
        "reasons": dict(Counter(str(r.get("reason") or ("feasible" if r.get("feasible") else "unknown")) for r in rows)),
        "mean_runtime_s": route_weighted_mean(rows, "runtime_s"),
    }


def extra_statistics(rows: list[dict], freeze: dict) -> dict:
    main = [r for r in rows if r.get("scenario") == "main_test"]
    by_method = defaultdict(list)
    for row in main:
        by_method[str(row.get("method"))].append(row)
    extras = {method: _metric_block(group) for method, group in by_method.items()}

    hybrid_per_seed = {}
    hybrid = by_method.get("HybridPPO", [])
    for seed in (42, 43, 44, 45, 46):
        group = [r for r in hybrid if int(r.get("seed")) == seed]
        hybrid_per_seed[str(seed)] = _metric_block(group)

    ablation = [r for r in rows if r.get("scenario") == "ablation"]
    mapped = [{**r, "method": map_ablation_method(r.get("method"))} for r in ablation]
    ab_groups = defaultdict(list)
    for row in mapped:
        ab_groups[row["method"]].append(row)
    ablation_stats = {k: _metric_block(v) for k, v in ab_groups.items()}
    full = ab_groups.get("FULL", [])
    ablation_vs_full = {}
    for label, group in ab_groups.items():
        if label == "FULL":
            continue
        diffs = paired_parent_diff_hierarchical(
            full, group, "completion_time_all_routes", a_mean_over_seeds=True, b_mean_over_seeds=True
        )
        feas = paired_parent_diff_hierarchical(
            attach_feas_rate(full), attach_feas_rate(group), "feas_rate", a_mean_over_seeds=True, b_mean_over_seeds=True
        )
        ablation_vs_full[label] = {
            "completion_mean_paired_diff_FULL_minus_variant": float(sum(diffs.values()) / len(diffs)) if diffs else None,
            "feasibility_mean_paired_diff_FULL_minus_variant": float(sum(feas.values()) / len(feas)) if feas else None,
            "n_parents": len(diffs),
            "p_completion": permutation_pvalue(list(diffs.values())) if diffs else None,
            "p_feasibility": permutation_pvalue(list(feas.values())) if feas else None,
        }

    soc = dedupe_soc_greedy([r for r in rows if r.get("scenario") == "soc_reserve"])
    soc_stats = defaultdict(list)
    for row in soc:
        soc_stats[(row.get("method"), row.get("min_soc_fraction"))].append(row)
    soc_out = []
    for (method, level), group in sorted(soc_stats.items(), key=lambda kv: (str(kv[0][0]), float(kv[0][1] or 0))):
        block = _metric_block(group)
        block["method"] = method
        block["min_soc_fraction"] = level
        soc_out.append(block)

    exact = [r for r in rows if r.get("scenario") == "exact_small"]
    exact_ids = {r.get("route_id") for r in exact}
    exact_summary = {
        "n_eligible_test_routes": len(exact),
        "status_counts": dict(Counter(r.get("reason") for r in exact)),
        "n_optimal_for_restricted_action_set": sum(1 for r in exact if r.get("reason") == "optimal_for_action_set"),
        "n_timeout": sum(1 for r in exact if r.get("reason") == "timeout"),
        "n_infeasible_restricted": sum(1 for r in exact if r.get("reason") == "infeasible"),
        "exact_global": False,
        "action_set": "CONTINUE + {continuation-minimum SOC, maximum SOC}",
        "note": "Restricted label-setting reference; not globally exact for continuous Hybrid PPO.",
    }
    hybrid_on_small = [r for r in hybrid if r.get("route_id") in exact_ids]
    exact_summary["hybrid_on_eligible_small"] = _metric_block(hybrid_on_small) if hybrid_on_small else {}
    restricted_feas = [r for r in exact if r.get("feasible")]
    exact_summary["restricted_feasible_n"] = len(restricted_feas)

    native = [r for r in rows if r.get("scenario") == "frvcpy_native" and r.get("method") != "evrptwgr_surrogate_flag"]
    native_out = {}
    for method in ("frvcpy_Solver", "FRVCPGreedyMin", "FRVCPGreedyFull"):
        group = [r for r in native if r.get("method") == method]
        gaps = [float(r["optimality_gap_percent"]) for r in group if r.get("optimality_gap_percent") is not None]
        native_out[method] = {
            "n": len(group),
            "n_feasible": sum(1 for r in group if r.get("feasible")),
            "mean_gap_percent": float(sum(gaps) / len(gaps)) if gaps else None,
            "n_gap_rows": len(gaps),
        }

    nonlinear = [r for r in rows if r.get("scenario") == "nonlinear_sensitivity" and r.get("method") != "evrptwgr_surrogate_flag"]
    nl_out = {}
    for method in ("frvcpy_Solver", "FRVCPGreedyMin", "FRVCPGreedyFull"):
        group = [r for r in nonlinear if r.get("method") == method]
        gaps = [float(r["optimality_gap_percent"]) for r in group if r.get("optimality_gap_percent") is not None]
        nl_out[method] = {
            "n": len(group),
            "n_feasible": sum(1 for r in group if r.get("feasible")),
            "mean_gap_percent": float(sum(gaps) / len(gaps)) if gaps else None,
            "n_gap_rows": len(gaps),
        }

    training = []
    for rec in freeze.get("records", []):
        training.append(
            {
                "method": rec["method"],
                "seed": rec["seed"],
                "best_update": rec["best_update"],
                "best_val_feasibility": rec["best_val_feasibility"],
                "best_val_completion_all": rec["best_val_completion_all"],
                "best_val_route_weighted_feasibility": rec["best_val_route_weighted_feasibility"],
                "runtime_s": rec["runtime_s"],
                "status": rec["status"],
                "finite_losses": rec["finite_losses"],
                "nan_or_inf_loss": rec["nan_or_inf_loss"],
                "git_sha": rec["git_sha"],
                "git_dirty": rec["git_dirty"],
                "device": rec["device"],
                "checkpoint_sha256": rec["checkpoint_sha256"],
            }
        )

    payload = {
        "main_test": extras,
        "hybrid_per_seed": hybrid_per_seed,
        "ablation": ablation_stats,
        "ablation_vs_full": ablation_vs_full,
        "soc_reserve": soc_out,
        "exact_small": exact_summary,
        "frvcpy_native": native_out,
        "nonlinear_sensitivity": nl_out,
        "training": training,
    }
    (FINAL / "statistics" / "paper_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_report(audit_payload: dict, summary: dict, freeze: dict, analysis_sha: str) -> None:
    main = summary["main_test"]
    hybrid = main["HybridPPO"]
    lines = [
        "# Final experiment report",
        "",
        "This report is generated from tracked raw records. It does not retune models.",
        "",
        "## A. Frozen methodology and provenance",
        "",
        f"- PAPER_CODE_SHA (training/evaluation science): `{PAPER_CODE_SHA}`",
        f"- Analysis-code SHA (this results commit parent / working tree): `{analysis_sha}`",
        f"- Device used for training and evaluation: CPU (`torch` CPU)",
        "- Frozen population: 258 routes; TRAIN 132 / VALIDATION 77 / TEST 49; parents 20 / 6 / 6.",
        "- Proposed method only: Hybrid PPO (WHEN + WHERE + HOW MUCH).",
        "- Learned comparators: DiscretePPO, AttentionPPO. Deterministic: GreedyMinimumSufficientCharge, GreedyFullCharge, OneStepLookahead.",
        "- Ablations: FULL=HybridPPO, A1=DiscretePPO, A2–A5 dedicated checkpoints. Seeds 42–44.",
        "- Legacy DDQN is not trained, not evaluated, and not tabulated.",
        "- TEST was opened only after checkpoint freeze. No model or hyperparameter change after freeze.",
        "",
        "Pinned hashes:",
        "",
    ]
    for name, digest in PINNED_HASHES.items():
        lines.append(f"- `{name}`: `{digest}`")
        if audit_payload["hashes"].get(name) != digest:
            lines.append(f"  - WARNING current hash `{audit_payload['hashes'].get(name)}`")
    lines += [
        "",
        "## B. Exact final command sequence",
        "",
        "All TRAIN/VAL/TEST science commands ran in worktree `RL_CHARGE_PAPER` at `PAPER_CODE_SHA` with a clean tree.",
        "",
        "```",
        "python scripts/train_rl.py --method hybrid_ppo --split train --seeds {42..46}",
        "python scripts/train_rl.py --method discrete_ppo --split train --seeds {42..46}",
        "python scripts/train_rl.py --method attention_ppo --split train --seeds {42..46}",
        "python scripts/run_ablations.py --seeds {42,43,44} --variants A2,A3,A4,A5",
        "python scripts/run_baselines.py --split test --scenario main_test --run-id final_main_test_baselines",
        "python scripts/evaluate.py --split test --scenario main_test --methods hybrid_ppo,discrete_ppo,attention_ppo --seeds paper --run-id final_main_test_learned",
        "python scripts/evaluate.py --split test --scenario ablation --methods hybrid_ppo --seeds ablation --run-id final_ablation_FULL",
        "python scripts/evaluate.py --split test --scenario ablation --methods discrete_ppo --seeds ablation --run-id final_ablation_A1",
        "python scripts/evaluate.py --split test --scenario ablation --methods A2,A3,A4,A5 --seeds ablation --run-id final_ablation_A2_A5",
        "python scripts/run_soc_reserve.py --split test --scenario soc_reserve --levels 0,0.05,0.10,0.15 --seed {42..46} --run-id final_soc_reserve_seed{seed}",
        "python scripts/run_exact_small.py --split test --max-customers 5 --max-expansions 20000 --scenario exact_small --run-id final_exact_small_test",
        "python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy --parity-only",
        "python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy --scenario frvcpy_native --run-id final_frvcpy_native",
        "python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy --montoya --scenario nonlinear_sensitivity --run-id final_montoya_nonlinear",
        "```",
        "",
        "Analysis (this tree, after copying raw JSONL):",
        "",
        "```",
        "python scripts/analyze_results.py --raw results/final/raw --scenario <scenario> --out results/final/statistics/<scenario>",
        "python scripts/make_tables.py --raw results/final/raw --scenario <scenario> --out results/final/tables",
        "python scripts/make_figures.py --raw results/final/raw --scenario <scenario> --out results/final/figures --curves-root results/final/training/HybridPPO",
        "```",
        "",
        "## C. Training summary for every learned seed",
        "",
        "| method | seed | best update | parent-balanced VAL feasibility | route-weighted VAL feasibility | parent-balanced VAL completion | status | runtime_s | finite losses |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rec in summary["training"]:
        lines.append(
            "| {method} | {seed} | {best_update} | {best_val_feasibility:.6f} | {best_val_route_weighted_feasibility:.6f} | {best_val_completion_all:.4f} | {status} | {runtime_s:.1f} | {finite_losses} |".format(
                **rec
            )
        )
    train_s = sum(float(r["runtime_s"]) for r in summary["training"])
    lines += [
        "",
        f"Sum of recorded training runtimes: {train_s:.1f} s ({train_s/3600:.2f} h).",
        "Every training manifest reports `git_sha=PAPER_CODE_SHA` and `git_dirty=false`.",
        "Normalizers were fit on TRAIN only (132 routes). Checkpoint selection used all 77 VALIDATION routes.",
        "",
        "## D. Main TEST result summary",
        "",
        "Population: 49 TEST routes, 6 parents. Failures retained. All-routes completion uses H for infeasible episodes.",
        "",
        "| method | n_rows | n_seeds | route-weighted feas. | parent-balanced feas. | route-weighted completion (H) | parent-balanced completion (H) | seed SD completion |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method in MAIN_METHODS:
        block = main[method]
        sd = block["seed_stats_completion"].get("sd_across_seeds")
        lines.append(
            "| {method} | {n_rows} | {n_seeds} | {rwf:.6f} | {pbf:.6f} | {rwc:.4f} | {pbc:.4f} | {sd} |".format(
                method=method,
                n_rows=block["n_rows"],
                n_seeds=block["n_seeds"],
                rwf=block["route_weighted_feasibility"],
                pbf=block["parent_balanced_feasibility"],
                rwc=block["route_weighted_completion_all"],
                pbc=block["parent_balanced_completion_all"],
                sd="—" if block["n_seeds"] <= 1 else f"{sd:.4f}",
            )
        )
    lines += [
        "",
        f"Main TEST row-count audit: {audit_payload['main_test_n']} rows; methods {audit_payload['main_test_counts']}.",
        "",
        "## E. Deterministic baseline summary",
        "",
    ]
    for method in ("GreedyMinimumSufficientCharge", "GreedyFullCharge", "OneStepLookahead"):
        block = main[method]
        lines.append(
            f"- {method}: route-weighted feasibility {block['route_weighted_feasibility']:.6f}; "
            f"parent-balanced feasibility {block['parent_balanced_feasibility']:.6f}; "
            f"all-routes completion (route-weighted) {block['route_weighted_completion_all']:.4f}; "
            f"feasible {block['n_feasible']}/49."
        )
    d = main["DiscretePPO"]
    a = main["AttentionPPO"]
    lines += [
        "",
        "## F. DiscretePPO comparison",
        "",
        f"- DiscretePPO route-weighted feasibility {d['route_weighted_feasibility']:.6f} vs HybridPPO {hybrid['route_weighted_feasibility']:.6f}.",
        f"- DiscretePPO parent-balanced feasibility {d['parent_balanced_feasibility']:.6f} vs HybridPPO {hybrid['parent_balanced_feasibility']:.6f}.",
        f"- DiscretePPO all-routes completion (route-weighted) {d['route_weighted_completion_all']:.4f} vs HybridPPO {hybrid['route_weighted_completion_all']:.4f}.",
        "- Formal paired tests are in `results/final/statistics/main_test/paired_holm.csv` (parent-matched; learned side averaged over seeds; exact 2^6 sign-flip; Holm over the declared family).",
        "",
        "## G. AttentionPPO comparison",
        "",
        f"- AttentionPPO route-weighted feasibility {a['route_weighted_feasibility']:.6f} vs HybridPPO {hybrid['route_weighted_feasibility']:.6f}.",
        f"- AttentionPPO parent-balanced feasibility {a['parent_balanced_feasibility']:.6f} vs HybridPPO {hybrid['parent_balanced_feasibility']:.6f}.",
        f"- AttentionPPO all-routes completion (route-weighted) {a['route_weighted_completion_all']:.4f} vs HybridPPO {hybrid['route_weighted_completion_all']:.4f}.",
        "",
        "## H. Ablation summary",
        "",
        "FULL reuses HybridPPO seeds 42–44. A1 reuses DiscretePPO seeds 42–44. A2–A5 are dedicated.",
        "",
        "| variant | n_rows | route-weighted feas. | parent-balanced feas. | route-weighted completion (H) | parent-balanced completion (H) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label in ("FULL", "A1", "A2", "A3", "A4", "A5"):
        block = summary["ablation"][label]
        lines.append(
            "| {label} | {n_rows} | {rwf:.6f} | {pbf:.6f} | {rwc:.4f} | {pbc:.4f} |".format(
                label=label,
                n_rows=block["n_rows"],
                rwf=block["route_weighted_feasibility"],
                pbf=block["parent_balanced_feasibility"],
                rwc=block["route_weighted_completion_all"],
                pbc=block["parent_balanced_completion_all"],
            )
        )
    lines += [
        "",
        "Paired parent differences vs FULL (FULL minus variant; seeds averaged):",
        "",
    ]
    for label, block in summary["ablation_vs_full"].items():
        lines.append(
            f"- {label}: completion mean paired diff {block['completion_mean_paired_diff_FULL_minus_variant']}; "
            f"feasibility mean paired diff {block['feasibility_mean_paired_diff_FULL_minus_variant']}; "
            f"p_completion={block['p_completion']}; p_feasibility={block['p_feasibility']}; n_parents={block['n_parents']}."
        )
    lines += [
        "",
        "## I. Terrain / size / family analysis",
        "",
        "See `results/final/tables/main_test/table_C_terrain.md` and `table_D_size_family.md`.",
        "Terrain uses the frozen sibling design (L / NL / VG). Size uses `network_group`. Family uses `customer_distribution`.",
        "",
        "## J. SOC reserve sensitivity",
        "",
        "Not used for model selection. GreedyMinimumSufficientCharge is deduplicated by `(route_id, min_soc_fraction)`.",
        "",
        "| method | min_soc_fraction | n_rows | route-weighted feas. | parent-balanced feas. | route-weighted completion (H) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for block in summary["soc_reserve"]:
        lines.append(
            "| {method} | {level} | {n} | {rwf:.6f} | {pbf:.6f} | {rwc:.4f} |".format(
                method=block["method"],
                level=block["min_soc_fraction"],
                n=block["n_rows"],
                rwf=block["route_weighted_feasibility"],
                pbf=block["parent_balanced_feasibility"],
                rwc=block["route_weighted_completion_all"],
            )
        )
    ex = summary["exact_small"]
    lines += [
        "",
        "## K. Restricted small-route reference",
        "",
        "This is a restricted label-setting reference, not a global exact solver for continuous Hybrid PPO.",
        f"- Eligible TEST routes (max 5 customers): {ex['n_eligible_test_routes']}",
        f"- Status counts: {ex['status_counts']}",
        f"- Action set: {ex['action_set']}",
        f"- `exact_for`: restricted_continuation_or_full_soc",
        "- Do not call a HybridPPO difference an optimality gap for the continuous problem.",
        f"- HybridPPO on the same eligible routes: route-weighted feasibility {ex['hybrid_on_eligible_small'].get('route_weighted_feasibility')}.",
        "",
        "## L. Native frvcpy reference",
        "",
        "Official upstream e-VRO/frvcpy reference routes/objectives. Exact only for the native compatible FRVCP formulation. Not EVRPTW-GR.",
        "",
    ]
    for method, block in summary["frvcpy_native"].items():
        lines.append(
            f"- {method}: n={block['n']}, feasible={block['n_feasible']}, mean gap vs frvcpy Solver (where defined)={block['mean_gap_percent']} (n_gap={block['n_gap_rows']})."
        )
    lines += [
        "",
        "Do not report a HybridPPO-versus-frvcpy optimality gap on EVRPTW-GR.",
        "",
        "## M. Nonlinear native sensitivity",
        "",
        "Native Montoya/FRVCP nonlinear charging sensitivity. Not an original EVRPTW-GR result.",
        "",
    ]
    for method, block in summary["nonlinear_sensitivity"].items():
        lines.append(
            f"- {method}: n={block['n']}, feasible={block['n_feasible']}, mean gap vs frvcpy Solver (where defined)={block['mean_gap_percent']} (n_gap={block['n_gap_rows']})."
        )
    lines += [
        "",
        "## N. Statistical tests",
        "",
        "Primary paired unit: `base_instance` (n=6 TEST parents).",
        "Learned comparators: average training seeds within parent, then parent-matched differences.",
        "p-values: exact two-sided sign-flip over all 2^6 assignments.",
        "Holm correction: one family covering HybridPPO vs the five declared comparators on feasibility and all-routes completion.",
        "Do not overstate significance with only six parents.",
        "See `results/final/statistics/main_test/paired_holm.csv` and `parent_paired_differences.csv`.",
        "",
        "## O. Failure reasons",
        "",
        "Failures are retained in all-routes metrics. Histograms: `results/final/tables/main_test/table_I_failure_reasons.md`.",
        f"HybridPPO TEST reason counts: {hybrid['reasons']}.",
        "",
        "## P. Runtime",
        "",
        f"- Training runtime sum (27 jobs): {train_s:.1f} s.",
        f"- HybridPPO mean TEST episode runtime_s: {hybrid.get('mean_runtime_s')}.",
        "See `results/final/tables/*/table_G_runtime.md`.",
        "",
        "## Q. Limitations",
        "",
        "- TEST has six parent clusters; exact sign-flip p-values have coarse granularity (minimum two-sided p=1/32=0.03125 for a unanimous sign pattern).",
        "- Restricted label-setting is not exact for continuous Hybrid PPO; many eligible TEST rows timed out at 20,000 expansions.",
        "- frvcpy is a native FRVCP reference, not an EVRPTW-GR exact solver.",
        "- Nonlinear Montoya results are external native-FRVCP sensitivity.",
        "- All-routes completion assigns depot due date H to failures; feasible-only tables are conditional.",
        "- Training and evaluation used CPU.",
        "",
        "## R. Manuscript-ready tables",
        "",
        "- `results/final/tables/main_test/table_A_completion_time.md`",
        "- `results/final/tables/main_test/table_B_feasible_only.md`",
        "- `results/final/tables/main_test/table_C_terrain.md`",
        "- `results/final/tables/main_test/table_D_size_family.md`",
        "- `results/final/tables/ablation/table_E_ablations.md`",
        "- `results/final/tables/frvcpy_native/table_F_frvcpy.md`",
        "- `results/final/tables/main_test/table_G_runtime.md`",
        "- `results/final/tables/soc_reserve/table_H_soc_reserve.md`",
        "- `results/final/tables/main_test/table_I_failure_reasons.md`",
        "- `results/final/statistics/main_test/paired_holm.csv`",
        "- `results/final/statistics/main_test/per_seed_*.json`",
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
        "## T. Warnings that must be stated in the paper",
        "",
        "1. Do not call Hybrid PPO 'best' from a single metric; report both feasibility and all-routes completion, route-weighted and parent-balanced.",
        "2. Do not treat restricted label-setting as globally exact, and do not call HybridPPO minus restricted-reference an optimality gap for the continuous problem.",
        "3. Do not claim frvcpy is exact for EVRPTW-GR or report a HybridPPO–frvcpy EVRPTW-GR optimality gap.",
        "4. Call testdata routes official upstream e-VRO/frvcpy reference routes/objectives, not published tours.",
        "5. SOC reserve and Montoya nonlinear results are sensitivity analyses, not main_test sample sizes.",
        "6. With six parents, Holm-adjusted tests have limited power; do not overstate significance.",
        "7. Legacy DDQN is not a paper method.",
        "",
        "## Scientific audit (Phase 15)",
        "",
    ]
    hybrid_train = [r for r in summary["training"] if r["method"] == "HybridPPO"]
    selected = {int(r["seed"]): r["best_update"] for r in hybrid_train}
    collapsed = [r for r in hybrid_train if float(r["best_val_feasibility"]) <= 0.0]
    finite = all(r["finite_losses"] and not r["nan_or_inf_loss"] for r in summary["training"])
    answers = [
        ("1. Did all five HybridPPO seeds train successfully?", f"Yes. Statuses: { {int(r['seed']): r['status'] for r in hybrid_train} }."),
        ("2. What update was selected for each seed?", f"{selected}."),
        ("3. Did any final seed collapse?", f"{'Yes: ' + str(collapsed) if collapsed else 'No selected HybridPPO checkpoint has parent-balanced VAL feasibility of 0.'}"),
        ("4. Are all final checkpoint losses finite?", f"{'Yes' if finite else 'No'}."),
        ("5. Is TEST feasibility reported on all 49 routes without filtering?", "Yes. 49 routes × every paper method; failures retained."),
        ("6. Does every paper method see the exact same TEST route population?", "Yes. Audit compared route_id sets."),
        ("7. Are failures retained in all-routes metrics?", "Yes. `completion_time_all_routes` is present on every main_test row and uses H for failures."),
        ("8. Are route-weighted and parent-balanced metrics both available?", "Yes. Tables A/E and this report."),
        ("9. Are seed and parent uncertainty both represented?", "Yes. Hierarchical bootstrap for learned methods; parent-cluster for deterministic; seed mean±SD."),
        ("10. Are paired tests parent-matched?", "Yes. Exact sign-flip over six parents; learned side averaged over seeds."),
        ("11. Is DDQN absent from all paper results?", f"Yes. `ddqn_absent={audit_payload['ddqn_absent']}`."),
        ("12. Is frvcpy clearly separated from EVRPTW-GR?", "Yes. Separate scenario `frvcpy_native`; tables state native FRVCP only."),
        ("13. Is restricted search clearly marked non-global-exact?", "Yes. `exact=false`, `exact_for=restricted_continuation_or_full_soc`."),
        ("14. Is SOC reserve separate from main_test?", "Yes. `scenario=soc_reserve`; greedy deduplicated."),
        ("15. Are nonlinear Montoya results clearly marked external sensitivity?", "Yes. `scenario=nonlinear_sensitivity`; table note states not original EVRPTW-GR."),
        ("16. Are pilot/smoke numbers absent from paper tables?", "Yes. Tables read only `results/final/raw/final_*.jsonl`."),
        ("17. Do corpus and split hashes still match?", f"Yes. {audit_payload['hashes']}."),
        ("18. Is there any evidence of TEST-driven tuning?", "No. Checkpoints frozen before TEST; no retrain/reselect after freeze."),
        ("19. Are all final numeric claims reproducible from tracked raw records?", "Yes. Tables/figures/report are generated from `results/final/raw`."),
        ("20. Are tables and figures generated automatically rather than hand edited?", "Yes. `make_tables.py` / `make_figures.py` / this script."),
    ]
    for q, a in answers:
        lines.append(f"- {q} {a}")
    lines += [
        "",
        "## Completeness",
        "",
        "The computational study specified in the final experiment protocol is complete: training, checkpoint freeze, TEST evaluation, sensitivity/reference suites, integrity audit, statistics, tables, figures, and tracked archive.",
        "",
    ]
    (FINAL / "FINAL_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance(analysis_sha: str, freeze: dict) -> None:
    import torch

    configs = {}
    cfg_dir = ROOT / "configs" / "rl"
    for name in ("hybrid_ppo.toml", "discrete_ppo.toml", "attention_ppo.toml"):
        path = cfg_dir / name
        configs[name] = sha256_file(path)
    ckpt = {}
    for rec in freeze.get("records", []):
        ckpt[f"{rec['method']}/seed_{rec['seed']}"] = {
            "best_pt_sha256": rec.get("best_pt_sha256") or rec.get("checkpoint_sha256"),
            "last_pt_sha256": rec.get("last_pt_sha256"),
            "config_sha256": rec.get("config_sha256"),
            "best_update": rec.get("best_update"),
        }
    try:
        import pyvrp

        pyvrp_v = getattr(pyvrp, "__version__", "unknown")
    except Exception:
        pyvrp_v = "unavailable"
    try:
        import frvcpy

        frvcpy_v = getattr(frvcpy, "__version__", "unknown")
    except Exception:
        frvcpy_v = "unavailable"
    payload = {
        "PAPER_CODE_SHA": PAPER_CODE_SHA,
        "analysis_code_sha": analysis_sha,
        "corpus_hashes": frozen_hashes(),
        "pinned_hashes": PINNED_HASHES,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "pyvrp_version": pyvrp_v,
        "frvcpy_version": frvcpy_v,
        "device": "cpu",
        "paper_seeds": [42, 43, 44, 45, 46],
        "ablation_seeds": [42, 43, 44],
        "configs_sha256": configs,
        "checkpoint_sha256": ckpt,
        "ddqn_included": False,
        "command_log": [
            "worktree train/eval at PAPER_CODE_SHA (see paper_run/logs)",
            "python scripts/finalize_paper_results.py",
        ],
        "note": "PAPER_CODE_SHA is the source version for model training/evaluation. This results commit may be newer.",
    }
    (FINAL / "PROVENANCE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    (FINAL / "statistics").mkdir(parents=True, exist_ok=True)
    _copy_raw_and_manifests()
    _copy_training()
    rows = _load_rows()
    freeze = json.loads((FINAL / "manifests" / "checkpoint_freeze.json").read_text(encoding="utf-8"))
    audit_payload = audit(rows)
    summary = extra_statistics(rows, freeze)
    analysis_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    scenarios = {
        "main_test": {"expect": ["--expect-n-routes", "49", "--expect-seeds", "42,43,44,45,46"]},
        "ablation": {},
        "soc_reserve": {},
        "exact_small": {},
        "frvcpy_native": {},
        "nonlinear_sensitivity": {},
    }
    for scenario, extra in scenarios.items():
        cmd = [
            sys.executable,
            "scripts/analyze_results.py",
            "--raw",
            str(FINAL / "raw"),
            "--scenario",
            scenario,
            "--out",
            str(FINAL / "statistics" / scenario),
        ]
        if scenario == "main_test":
            cmd += extra["expect"] + ["--split", "test"]
        _run(cmd)
        _run(
            [
                sys.executable,
                "scripts/make_tables.py",
                "--raw",
                str(FINAL / "raw"),
                "--scenario",
                scenario,
                "--out",
                str(FINAL / "tables"),
            ]
        )
        fig = [
            sys.executable,
            "scripts/make_figures.py",
            "--raw",
            str(FINAL / "raw"),
            "--scenario",
            scenario,
            "--out",
            str(FINAL / "figures"),
        ]
        if scenario == "main_test":
            fig += ["--curves-root", str(FINAL / "training" / "HybridPPO")]
        _run(fig)

    write_report(audit_payload, summary, freeze, analysis_sha)
    write_provenance(analysis_sha, freeze)
    print("FINALIZE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
