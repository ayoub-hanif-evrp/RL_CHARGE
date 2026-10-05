"""Write FINAL_REWARD_FREEZE.json from authoritative final manifests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANT = "V4_BASE_NO_L_FAIL"
FINAL = ROOT / "results" / "v4_reward" / "final_authoritative" / VARIANT
OUT = ROOT / "results" / "v4_reward" / "FINAL_REWARD_FREEZE.json"
SEEDS = (42, 43, 44, 45, 46)

sys_path_note = "approved outputs ignored for code dirtiness"


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(["git", *cmd], cwd=str(ROOT), text=True).strip()


def main() -> None:
    sys_path = str(ROOT / "src")
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    from experiments.provenance import code_dirty_paths  # noqa: WPS433

    rows = []
    for seed in SEEDS:
        path = FINAL / f"seed_{seed}" / "validation.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    dirty_paths = code_dirty_paths()
    sha = _git(["rev-parse", "HEAD"])
    if dirty_paths:
        raise SystemExit("working tree has uncommitted source changes: " + ", ".join(dirty_paths[:12]))
    if any(r.get("code_git_dirty") for r in rows):
        raise SystemExit("one or more final manifests have code_git_dirty=true")
    bad = [r["seed"] for r in rows if r.get("run_kind") not in (None, "clean_sha")]
    if bad:
        raise SystemExit(f"non-clean run_kind in seeds {bad}")
    train_sha = rows[0].get("training_git_sha") or rows[0]["git_sha"]
    if any((r.get("training_git_sha") or r.get("git_sha")) != train_sha for r in rows):
        raise SystemExit("final manifests disagree on training_git_sha")
    if any(r.get("confirmatory_test_consumed") for r in rows):
        raise SystemExit("TEST consumption flag set")
    for r in rows:
        ckpt = str(r.get("checkpoint", ""))
        if "C:/" in ckpt or "Users/" in ckpt or ":\\" in ckpt:
            raise SystemExit(f"absolute checkpoint path in seed {r['seed']}: {ckpt}")

    feas = [float(r["parent_balanced_val_feasibility"]) for r in rows]
    comp = [float(r["parent_balanced_completion_all"]) for r in rows]
    import statistics

    payload = {
        "reward_kind": VARIANT,
        "human_name": "normalized_time_horizon_reward",
        "equations": {
            "feasible_step": "r_t = -Delta_t / C_train",
            "terminal_failure": "r_fail = -(H - t) / C_train",
            "episode_return_success": "G = -T_completion / C_train",
            "episode_return_failure": "G = -H / C_train",
        },
        "c_train": float(rows[0]["c_train"]),
        "c_train_rule": rows[0]["c_train_rule"],
        "training_git_sha": train_sha,
        "git_sha": train_sha,
        "code_git_dirty": False,
        "git_dirty": False,
        "run_kind": "clean_sha",
        "namespace": "results/v4_reward/final_authoritative/",
        "checkpoint_namespace": "checkpoints_v4/final_authoritative/",
        "train_corpus": rows[0]["train_corpus"],
        "val_corpus": rows[0]["val_corpus"],
        "train_corpus_sha256": rows[0]["train_corpus_sha256"],
        "val_corpus_sha256": rows[0]["val_corpus_sha256"],
        "ppo_config": rows[0]["ppo_config"],
        "ppo_config_sha256": rows[0]["ppo_config_sha256"],
        "val_feasibility_mean": float(statistics.mean(feas)),
        "val_feasibility_sd": float(statistics.stdev(feas)) if len(feas) > 1 else 0.0,
        "val_completion_mean": float(statistics.mean(comp)),
        "val_completion_sd": float(statistics.stdev(comp)) if len(comp) > 1 else 0.0,
        "seeds": {
            str(r["seed"]): {
                "checkpoint": r["checkpoint"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "best_update": r["best_update"],
                "runtime_s": r.get("runtime_s"),
                "parent_balanced_val_feasibility": r["parent_balanced_val_feasibility"],
                "parent_balanced_completion_all": r["parent_balanced_completion_all"],
            }
            for r in rows
        },
        "selection_rationale": (
            "Development selection rule: V4_BASE_NO_L_FAIL matched the best mean development "
            "feasibility and obtained the lowest mean failure-retaining completion among the tied "
            "variants. Effects across seeds were variable, so the reward is selected primarily for "
            "its simple objective-consistent formulation rather than claimed as a large performance "
            "improvement. PBRS remains a negative/neutral ablation."
        ),
        "methodology_statement": (
            "The normalized time-horizon reward provides a simpler objective-aligned formulation "
            "without arbitrary weighted penalties. Development evidence showed no meaningful "
            "feasibility advantage from PBRS, so the simpler reward was selected and evaluated "
            "prospectively on a fresh held-out TEST."
        ),
        "confirmatory_test_consumed": False,
        "v1_v2_v3_untouched": True,
        "prior_final_clean_not_authoritative": (
            "results/v4_reward/final_clean/ is retained as historical; "
            "final_authoritative/ is the frozen V4 reward checkpoint namespace."
        ),
        "freeze_script_head_at_write": sha,
        "provenance_note": sys_path_note,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OUT.relative_to(ROOT)).replace("\\", "/"), "training_git_sha": train_sha}, indent=2))


if __name__ == "__main__":
    main()
