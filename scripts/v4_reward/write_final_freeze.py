"""Write FINAL_REWARD_FREEZE.json from clean final_clean manifests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANT = "V4_BASE_NO_L_FAIL"
FINAL = ROOT / "results" / "v4_reward" / "final_clean" / VARIANT
OUT = ROOT / "results" / "v4_reward" / "FINAL_REWARD_FREEZE.json"
SEEDS = (42, 43, 44, 45, 46)


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(["git", *cmd], cwd=str(ROOT), text=True).strip()


_OUTPUT_PREFIXES = (
    "results/v4_reward/final_clean/",
    "checkpoints_v4/final_reward/",
    "results/v4_reward/FINAL_REWARD_FREEZE.json",
    "results_v4/figures/",
)


def _code_dirty_paths() -> list[str]:
    rows = []
    for line in _git(["status", "--porcelain"]).splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if any(path.startswith(prefix) for prefix in _OUTPUT_PREFIXES):
            continue
        rows.append(path)
    return rows


def main() -> None:
    rows = []
    for seed in SEEDS:
        path = FINAL / f"seed_{seed}" / "validation.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    dirty_paths = _code_dirty_paths()
    sha = _git(["rev-parse", "HEAD"])
    if dirty_paths:
        raise SystemExit("working tree has uncommitted non-output changes: " + ", ".join(dirty_paths[:12]))
    if any(r.get("git_dirty") for r in rows):
        raise SystemExit("one or more final manifests have git_dirty=true")
    # Manifests must match the clean training SHA (may be behind the freeze commit).
    train_sha = rows[0]["git_sha"]
    if any(r.get("git_sha") != train_sha for r in rows):
        raise SystemExit("final manifests disagree on git_sha")
    if any(r.get("confirmatory_test_consumed") for r in rows):
        raise SystemExit("TEST consumption flag set")

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
        "git_sha_training": train_sha,
        "git_sha_freeze_script_head": sha,
        "git_dirty": False,
        "train_corpus": rows[0]["train_corpus"],
        "val_corpus": rows[0]["val_corpus"],
        "train_corpus_sha256": rows[0]["train_corpus_sha256"],
        "val_corpus_sha256": rows[0]["val_corpus_sha256"],
        "ppo_config": rows[0]["ppo_config"],
        "ppo_config_sha256": rows[0]["ppo_config_sha256"],
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
            "Lexicographic TRAIN/VAL development rule: (1) maximize mean parent-balanced "
            "VAL feasibility; (2) among ties, minimize mean parent-balanced failure-retaining "
            "completion. V4_BASE_NO_L_FAIL tied V3_TIME/V4_BASE at 98.2% feasibility and "
            "achieved the best mean completion (3.878). PBRS is retained as a negative/neutral "
            "ablation (97.8% feasibility) and is not the selected reward. Secondary energy/"
            "station/SOC metrics are reported on common-feasible matched pairs only and do not "
            "redefine the objective. No new reward weights were added after seeing VAL results."
        ),
        "methodology_statement": (
            "FA-HPPO uses an objective-aligned time reward rather than a weighted mixture of "
            "arbitrary penalties. Feasible transitions incur normalized elapsed-time cost, "
            "while failure maps the episode to the route horizon. Hard EV and time-window "
            "requirements are enforced structurally. Distance contributes through travel "
            "time, while energy charged, charging stops, and terminal SOC are reported "
            "separately as operational metrics."
        ),
        "confirmatory_test_consumed": False,
        "v1_v2_v3_untouched": True,
        "development_ablation_not_final": (
            "results/v4_reward/ablation/ checkpoints were dirty-tree development runs and "
            "are not the frozen final V4 reward checkpoints."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OUT.relative_to(ROOT)).replace("\\", "/"), "git_sha": sha}, indent=2))


if __name__ == "__main__":
    main()
