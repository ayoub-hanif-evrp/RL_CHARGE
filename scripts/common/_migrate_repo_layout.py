"""One-shot repository layout migration (hash-preserving moves).

Run from repo root:
  python scripts/common/_migrate_repo_layout.py

Does not retrain, does not rerun TEST, does not alter file contents.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "docs" / "reproducibility"
MIGRATION_JSON = REPORT_DIR / "REPOSITORY_MIGRATION.json"
CLEANUP_PLAN = REPORT_DIR / "CLEANUP_PLAN.md"
PATH_MIGRATION_MD = REPORT_DIR / "PATH_MIGRATION.md"
MODEL_INVENTORY = REPORT_DIR / "MODEL_INVENTORY.md"

moves: list[dict] = []
deletes: list[dict] = []
archives: list[dict] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def move_path(src: Path, dst: Path, *, kind: str = "move") -> None:
    if not src.exists():
        moves.append({"old_path": rel(src) if src.is_absolute() else str(src), "new_path": rel(dst), "status": "missing_source", "kind": kind})
        return
    if dst.exists():
        raise SystemExit(f"destination exists: {dst}")
    ensure_parent(dst)
    # Hash critical files before move
    file_hashes: list[tuple[str, str]] = []
    if src.is_file():
        before = sha256(src)
        file_hashes.append((rel(src), before))
    else:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                # Sample: hash all .pt, .json, .jsonl, freeze files; skip huge bulk optionally
                if p.suffix.lower() in {".pt", ".json", ".jsonl", ".md", ".csv", ".png"} or p.name.endswith("_FREEZE.json") or "LOCK" in p.name or "CONSUMED" in p.name or "PROTOCOL" in p.name:
                    file_hashes.append((rel(p), sha256(p)))

    shutil.move(str(src), str(dst))

    after_entries = []
    for old_rel, before in file_hashes:
        # map old relative under src to new under dst
        old_path = ROOT / old_rel
        # After move, reconstruct new path
        try:
            suffix = Path(old_rel).relative_to(rel(src) if src.is_absolute() else src)
        except Exception:
            # src was moved; compute from recorded rel
            src_rel = moves[-1]["old_path"] if False else None
            # Use path surgery: old_rel starts with old src rel
            pass
        # Better: recompute from stored mapping
        src_rel = rel(src) if not str(src).endswith(str(dst.name)) else None
    # Re-hash after using dst tree
    verified = []
    before_map = {old: before for old, before in file_hashes}
    src_rel = None
    # Reconstruct src_rel from first hash keys
    if file_hashes:
        # Find common prefix of hashed files that was the moved root
        # We recorded rel(p) before move; after move those paths don't exist.
        # Compute new path: dst / relative_to(old_src)
        pass

    # Simpler verification: walk destination and match by relative suffix
    # Store mapping using precomputed relative paths from src root.
    # Re-do move logic with proper relative tracking:
    raise RuntimeError("internal: use move_tree")


def move_tree(src: Path, dst: Path, *, kind: str = "move") -> None:
    """Move a file or directory; verify selected content hashes unchanged."""
    if not src.exists():
        moves.append(
            {
                "old_path": rel(src),
                "new_path": rel(dst),
                "sha256_before": None,
                "sha256_after": None,
                "status": "missing_source",
                "kind": kind,
            }
        )
        return
    if dst.exists():
        moves.append(
            {
                "old_path": rel(src),
                "new_path": rel(dst),
                "sha256_before": None,
                "sha256_after": None,
                "status": "skipped_dest_exists",
                "kind": kind,
            }
        )
        return

    records: list[tuple[str, str, str]] = []  # old_rel, rel_within, before_hash
    if src.is_file():
        before = sha256(src)
        records.append((rel(src), src.name, before))
        ensure_parent(dst)
        shutil.move(str(src), str(dst))
        after = sha256(dst)
        status = "ok" if after == before else "HASH_MISMATCH"
        moves.append(
            {
                "old_path": rel(src),
                "new_path": rel(dst),
                "sha256_before": before,
                "sha256_after": after,
                "status": status,
                "kind": kind,
            }
        )
        if status != "ok":
            raise SystemExit(f"hash mismatch moving {src} -> {dst}")
        return

    # directory
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        within = p.relative_to(src).as_posix()
        # Always hash scientifically important suffixes; also hash everything under freeze/test dirs
        important = (
            p.suffix.lower() in {".pt", ".json", ".jsonl", ".md", ".csv", ".png", ".toml", ".tex"}
            or "FREEZE" in p.name
            or "LOCK" in p.name
            or "CONSUMED" in p.name
            or "PROTOCOL" in p.name
            or "MANIFEST" in p.name
        )
        if important:
            records.append((rel(p), within, sha256(p)))

    ensure_parent(dst)
    shutil.move(str(src), str(dst))

    mismatches = 0
    for old_rel, within, before in records:
        new_path = dst / within
        after = sha256(new_path) if new_path.is_file() else None
        status = "ok" if after == before else "HASH_MISMATCH"
        if status != "ok":
            mismatches += 1
        moves.append(
            {
                "old_path": old_rel,
                "new_path": rel(new_path),
                "sha256_before": before,
                "sha256_after": after,
                "status": status,
                "kind": kind,
            }
        )
    if mismatches:
        raise SystemExit(f"{mismatches} hash mismatches after moving {src} -> {dst}")
    # Also record the directory-level move
    moves.append(
        {
            "old_path": rel(src),
            "new_path": rel(dst),
            "sha256_before": None,
            "sha256_after": None,
            "status": "directory_moved",
            "kind": kind,
            "n_hashed_files": len(records),
        }
    )


def delete_path(path: Path, reason: str) -> None:
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            if path.exists():
                deletes.append({"path": rel(path), "reason": reason, "status": "partial_permission_denied"})
                return
        else:
            path.unlink()
    except OSError as exc:
        deletes.append({"path": rel(path), "reason": reason, "status": f"failed:{exc}"})
        return
    deletes.append({"path": rel(path), "reason": reason, "status": "deleted"})


def archive_md(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    move_tree(src, dst, kind="archive")
    archives.append({"old_path": rel(src), "new_path": rel(dst)})


def write_cleanup_plan() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = """# Cleanup plan (pre-deletion inventory)

## KEEP / MOVE

| Item | Action |
|------|--------|
| `results/v4_reward/` | → `results/v4/reward_development/` |
| `results/v4_test/` | → `results/v4/test/` |
| `results_v4_paper/` | → `results/v4/paper/` |
| `results_v4/` | → `results/v4/training_figures/` |
| `results_paper/` | → `results/v3/paper/` |
| `results/v3_hppo/` | → `results/v3/` (scientific tree) |
| `results/final/` | → `results/v1/` |
| `results/v2/` | keep under `results/v2/` |
| `checkpoints*/` | → `models/v{1,2,3,4}/` (+ development) |
| V4 authoritative `best.pt` | → `models/v4/fa_hppo/seed_*/best.pt` |
| Active docs (`physics.md`, `V4_*` design, etc.) | → `docs/methodology/` or `docs/reproducibility/` |
| Historical cleanup/status reports | → `docs/archive/` |
| Scripts by version | → `scripts/{common,v1,v2,v3,v4,paper}/` |

## ARCHIVE

- Old cleanup / finalization / scientific-fix reports under `docs/` and `docs/history/`
- `context.md` (session dump)
- Development envelope ablation models under `models/development/`

## DELETE

| Item | Reason |
|------|--------|
| `.ipynb_checkpoints/` | Generated Jupyter noise |
| `.pytest_cache/` | Generated test cache |
| `__pycache__/` / `*.pyc` | Generated bytecode |
| Empty `paper/figures/`, `paper/tables/` | Empty placeholders |
| Exact duplicate `.pt` copies under superseded paths after canonical model placement | Same SHA256 as canonical (recorded) |
| `Thumbs.db` / `.DS_Store` | OS junk |

Ambiguous frozen evidence is kept or archived, never deleted.
"""
    CLEANUP_PLAN.write_text(text, encoding="utf-8")


def migrate_models() -> None:
    models = ROOT / "models"
    models.mkdir(exist_ok=True)

    # V1–V3 wholesale
    move_tree(ROOT / "checkpoints", models / "v1", kind="models")
    move_tree(ROOT / "checkpoints_v2", models / "v2", kind="models")
    move_tree(ROOT / "checkpoints_v3", models / "v3", kind="models")
    move_tree(ROOT / "checkpoints_development", models / "development", kind="models")

    # V4: create structure
    v4 = models / "v4"
    v4.mkdir(exist_ok=True)
    src_auth = ROOT / "checkpoints_v4" / "final_authoritative" / "V4_BASE_NO_L_FAIL"
    if src_auth.exists():
        move_tree(src_auth, v4 / "fa_hppo", kind="models")
    move_tree(ROOT / "checkpoints_v4" / "reward_ablation", v4 / "reward_ablation", kind="models")
    move_tree(ROOT / "checkpoints_v4" / "reward_ablation_smoke", v4 / "reward_ablation_smoke", kind="models")

    # final_reward: keep only if not exact duplicate of fa_hppo best.pt
    src_final = ROOT / "checkpoints_v4" / "final_reward" / "V4_BASE_NO_L_FAIL"
    if src_final.exists() and (v4 / "fa_hppo").exists():
        dup_ok = True
        for seed_dir in sorted(src_final.glob("seed_*")):
            best = seed_dir / "best.pt"
            canon = v4 / "fa_hppo" / seed_dir.name / "best.pt"
            if best.is_file() and canon.is_file() and sha256(best) == sha256(canon):
                deletes.append(
                    {
                        "path": rel(best),
                        "reason": "exact duplicate of models/v4/fa_hppo canonical best.pt",
                        "canonical_path": rel(canon),
                        "shared_sha256": sha256(best),
                    }
                )
            else:
                dup_ok = False
        if dup_ok:
            # Still keep sidecar manifests/json by moving remaining tree excluding best.pt duplicates
            move_tree(src_final, v4 / "final_reward_sidecars", kind="models")
            # Remove duplicate best.pt from sidecars
            for seed_dir in sorted((v4 / "final_reward_sidecars").glob("seed_*")):
                bp = seed_dir / "best.pt"
                if bp.is_file():
                    canon = v4 / "fa_hppo" / seed_dir.name / "best.pt"
                    if canon.is_file():
                        digest = sha256(bp)
                        if digest == sha256(canon):
                            bp.unlink()
                            deletes.append(
                                {
                                    "path": rel(bp),
                                    "reason": "removed duplicate best.pt after sidecar move",
                                    "canonical_path": rel(canon),
                                    "shared_sha256": digest,
                                }
                            )
        else:
            move_tree(src_final, v4 / "final_reward", kind="models")

    # Remove emptied checkpoints_v4 if remains
    ck4 = ROOT / "checkpoints_v4"
    if ck4.exists():
        # move any leftovers
        leftovers = [p for p in ck4.rglob("*") if p.is_file()]
        if leftovers:
            move_tree(ck4, v4 / "_leftover_checkpoints_v4", kind="models")
        else:
            shutil.rmtree(ck4)
            deletes.append({"path": "checkpoints_v4", "reason": "empty after model migration"})


def migrate_results() -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    # V1
    if (results / "final").exists() and not (results / "v1").exists():
        (results / "v1").mkdir(parents=True, exist_ok=True)
        move_tree(results / "final", results / "v1" / "final", kind="results")
    for name in ("summaries", "pilot"):
        src = results / name
        if src.exists():
            move_tree(src, results / "v1" / name, kind="results")

    # V3
    v3 = results / "v3"
    if (results / "v3_hppo").exists():
        move_tree(results / "v3_hppo", v3, kind="results")
    if (ROOT / "results_paper").exists():
        # if v3 already has paper? unlikely
        move_tree(ROOT / "results_paper", v3 / "paper", kind="results")

    # V4
    v4 = results / "v4"
    v4.mkdir(exist_ok=True)
    if (results / "v4_reward").exists():
        move_tree(results / "v4_reward", v4 / "reward_development", kind="results")
    if (results / "v4_test").exists():
        move_tree(results / "v4_test", v4 / "test", kind="results")
    if (ROOT / "results_v4_paper").exists():
        move_tree(ROOT / "results_v4_paper", v4 / "paper", kind="results")
    if (ROOT / "results_v4").exists():
        move_tree(ROOT / "results_v4", v4 / "training_figures", kind="results")

    # Keep development/runs/smoke/raw under results/ (already there)


def migrate_scripts() -> None:
    scripts = ROOT / "scripts"
    common = scripts / "common"
    common.mkdir(exist_ok=True)
    v1 = scripts / "v1"
    v1.mkdir(exist_ok=True)
    v4 = scripts / "v4"
    v4.mkdir(exist_ok=True)

    # Version folders
    if (scripts / "v4_reward").exists():
        move_tree(scripts / "v4_reward", v4 / "reward", kind="scripts")
    if (scripts / "v4_test").exists():
        move_tree(scripts / "v4_test", v4 / "test", kind="scripts")
    if (scripts / "v3_hppo").exists():
        move_tree(scripts / "v3_hppo", scripts / "v3", kind="scripts")

    common_scripts = [
        "train_ppo.py",
        "train_rl.py",
        "evaluate.py",
        "evaluate_policy.py",
        "analyze_results.py",
        "preflight_experiments.py",
        "make_figures.py",
        "make_tables.py",
        "make_splits.py",
        "generate_routes.py",
        "inspect_dataset.py",
        "simulate_route.py",
        "validate_physics.py",
        "audit_corpus.py",
    ]
    v1_scripts = [
        "run_ablations.py",
        "run_baselines.py",
        "run_exact_small.py",
        "run_frvcpy_benchmark.py",
        "run_nonlinear_sensitivity.py",
        "run_restricted_search.py",
        "run_size_generalization.py",
        "run_soc_reserve.py",
        "vendor_frvcpy_benchmark.py",
        "finalize_paper_results.py",
        "cleanup_paper_analysis.py",
    ]
    for name in common_scripts:
        src = scripts / name
        if src.exists():
            move_tree(src, common / name, kind="scripts")
    for name in v1_scripts:
        src = scripts / name
        if src.exists():
            move_tree(src, v1 / name, kind="scripts")

    # Keep this migration script under common (already there once moved)
    # paper/ and v2/ and development/ stay


def migrate_configs() -> None:
    configs = ROOT / "configs"
    common = configs / "common"
    common.mkdir(exist_ok=True)
    for name in ("experiments", "physics", "rl", "routing"):
        src = configs / name
        if src.exists() and not (common / name).exists():
            move_tree(src, common / name, kind="configs")
    # v2 already exists
    (configs / "v3").mkdir(exist_ok=True)
    (configs / "v4").mkdir(exist_ok=True)
    # Copy-reference note files only if present under results
    # Leave empty version folders with README pointers
    (configs / "v3" / "README.md").write_text(
        "V3 experiment configs live with frozen evidence under `results/v3/configs/` when present.\n",
        encoding="utf-8",
    )
    (configs / "v4" / "README.md").write_text(
        "V4 protocol/freeze files are under `results/v4/test/` and `results/v4/reward_development/`.\n"
        "RL training TOMLs used historically remain under `configs/common/rl/`.\n",
        encoding="utf-8",
    )
    (configs / "v1" / "README.md").parent.mkdir(exist_ok=True)
    (configs / "v1" / "README.md").write_text(
        "V1 used `configs/common/` TOMLs (formerly top-level configs/rl, physics, …).\n",
        encoding="utf-8",
    )


def migrate_docs() -> None:
    docs = ROOT / "docs"
    method = docs / "methodology"
    repro = docs / "reproducibility"
    archive = docs / "archive"
    for d in (method, repro, archive):
        d.mkdir(parents=True, exist_ok=True)

    methodology = [
        "physics.md",
        "simulator.md",
        "rl.md",
        "routes.md",
        "splits.md",
        "feasibility.md",
        "experiments.md",
        "frvcpy_env.md",
        "FROZEN_METHOD_NOTES.md",
        "CONTINUOUS_HEAD_LIMITATION.md",
        "OPTIMIZATION_COMPARATOR_DESIGN.md",
        "STRONG_COMPARATOR_FEASIBILITY_STUDY.md",
        "STATION_FEATURE_AUDIT.md",
        "V4_BENCHMARK_WORDING.md",
        "V4_REWARD_DESIGN.md",
        "V4_PROPOSAL.md",
        "V4_METHOD_CANDIDATES.md",
        "ENVIRONMENT_REPRODUCIBILITY.md",
    ]
    for name in methodology:
        src = docs / name
        if src.exists():
            move_tree(src, method / name, kind="docs")

    archive_names = [
        "REPO_CLEANUP_REPORT.md",
        "SCIENTIFIC_FIXES_REPORT.md",
        "V4_REWARD_IMPLEMENTATION_REPORT.md",
        "ARCHIVE_LAYOUT.md",
        "FINAL_CLEANUP_AUDIT.md",
        "CLEANUP_MANIFEST.md",
    ]
    for name in archive_names:
        src = docs / name
        if src.exists():
            move_tree(src, archive / name, kind="archive")

    if (docs / "history").exists():
        move_tree(docs / "history", archive / "history", kind="archive")

    # Root clutter
    if (ROOT / "context.md").exists():
        move_tree(ROOT / "context.md", archive / "context.md", kind="archive")


def cleanup_junk() -> None:
    for name in (".ipynb_checkpoints", ".pytest_cache"):
        p = ROOT / name
        if p.exists():
            delete_path(p, "generated cache/noise")
    for p in ROOT.rglob("__pycache__"):
        if p.is_dir() and ".git" not in p.parts:
            delete_path(p, "bytecode cache")
    for p in list(ROOT.rglob("Thumbs.db")) + list(ROOT.rglob(".DS_Store")):
        if ".git" not in p.parts:
            delete_path(p, "OS junk")
    for empty in (ROOT / "paper" / "figures", ROOT / "paper" / "tables"):
        if empty.exists() and empty.is_dir() and not any(empty.iterdir()):
            delete_path(empty, "empty placeholder")


def write_path_migration_md() -> None:
    lines = [
        "# Path migration map",
        "",
        "Frozen protocol/manifest JSON files that embed historical paths are **byte-preserved**.",
        "Use this map (and `src/repo_paths.py`) to resolve old paths to canonical locations.",
        "",
        "| Old path | New canonical path |",
        "|---|---|",
        "| `checkpoints_v4/final_authoritative/V4_BASE_NO_L_FAIL/` | `models/v4/fa_hppo/` |",
        "| `checkpoints_v4/reward_ablation/` | `models/v4/reward_ablation/` |",
        "| `checkpoints_v3/` | `models/v3/` |",
        "| `checkpoints_v2/` | `models/v2/` |",
        "| `checkpoints/` | `models/v1/` |",
        "| `checkpoints_development/` | `models/development/` |",
        "| `results_v4_paper/` | `results/v4/paper/` |",
        "| `results_v4/` | `results/v4/training_figures/` |",
        "| `results/v4_reward/` | `results/v4/reward_development/` |",
        "| `results/v4_test/` | `results/v4/test/` |",
        "| `results_paper/` | `results/v3/paper/` |",
        "| `results/v3_hppo/` | `results/v3/` |",
        "| `results/final/` | `results/v1/final/` |",
        "| `scripts/v4_reward/` | `scripts/v4/reward/` |",
        "| `scripts/v4_test/` | `scripts/v4/test/` |",
        "| `scripts/v3_hppo/` | `scripts/v3/` |",
        "| `scripts/run_frvcpy_benchmark.py` | `scripts/v1/run_frvcpy_benchmark.py` |",
        "",
    ]
    PATH_MIGRATION_MD.write_text("\n".join(lines), encoding="utf-8")


def write_model_inventory() -> None:
    rows = []
    models = ROOT / "models"
    if models.exists():
        for pt in sorted(models.rglob("*.pt")):
            digest = sha256(pt)
            rel_p = rel(pt)
            version = "unknown"
            for v in ("v1", "v2", "v3", "v4", "development"):
                if f"models/{v}/" in rel_p.replace("\\", "/"):
                    version = v
                    break
            decision = "KEEP"
            used = "historical"
            if "models/v4/fa_hppo/" in rel_p.replace("\\", "/") and pt.name == "best.pt":
                used = "V4 TEST / paper Fig.4 / authoritative"
            elif "reward_ablation" in rel_p:
                used = "V4 reward ablation (VAL)"
            elif version == "development":
                decision = "OPTIONAL_ARCHIVE"
                used = "envelope ablation development"
            elif pt.name == "last.pt":
                decision = "OPTIONAL_ARCHIVE"
                used = "training last snapshot"
            rows.append(
                {
                    "version": version,
                    "path": rel_p,
                    "sha256": digest,
                    "used_by": used,
                    "decision": decision,
                }
            )
    lines = [
        "# Model inventory",
        "",
        "| Version | Path | SHA256 | Used by | Decision |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['version']} | `{r['path']}` | `{r['sha256']}` | {r['used_by']} | {r['decision']} |")
    lines.append("")
    MODEL_INVENTORY.write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "MODEL_INVENTORY.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    write_cleanup_plan()
    migrate_models()
    migrate_results()
    migrate_scripts()
    migrate_configs()
    migrate_docs()
    cleanup_junk()
    write_path_migration_md()
    write_model_inventory()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "moves": moves,
        "deletes": deletes,
        "archives": archives,
        "n_moves": len(moves),
        "n_deletes": len(deletes),
        "hash_mismatches": sum(1 for m in moves if m.get("status") == "HASH_MISMATCH"),
    }
    MIGRATION_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["hash_mismatches"]:
        raise SystemExit("migration completed with hash mismatches")
    print(json.dumps({"n_moves": payload["n_moves"], "n_deletes": payload["n_deletes"], "out": rel(MIGRATION_JSON)}, indent=2))


if __name__ == "__main__":
    # Allow running before it is moved into scripts/common
    main()
