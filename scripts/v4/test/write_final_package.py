"""Assemble results/v4/test/FINAL_V4_PACKAGE.json after TEST analysis."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V4T = ROOT / "results" / "v4" / "test"
V4R = ROOT / "results" / "v4" / "reward_development"
OUT = V4T / "FINAL_V4_PACKAGE.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    reward = _load(V4R / "FINAL_REWARD_FREEZE.json")
    protocol = _load(V4T / "PAPER_PROTOCOL.json")
    lock = _load(V4T / "TEST_LOCK.json")
    freeze = _load(V4T / "CHECKPOINT_FREEZE.json")
    consumed = _load(V4T / "EVALUATION_CONSUMED.json")
    stats = _load(V4T / "statistics" / "main_summary.json")
    fig_manifest = _load(ROOT / "results" / "v4" / "paper" / "manifests" / "FIGURE_MANIFEST.json")
    if fig_manifest is None:
        fig_manifest = _load(ROOT / "results" / "v4" / "reward_development" / "figures" / "MANIFEST.json")
    ablation = _load(V4R / "SUMMARY.json")
    payload = {
        "exact_reward_equations": reward.get("equations") if reward else None,
        "c_train": reward.get("c_train") if reward else None,
        "c_train_rule": reward.get("c_train_rule") if reward else None,
        "clean_training_git_sha": reward.get("training_git_sha") or (reward or {}).get("git_sha"),
        "five_checkpoint_hashes": {
            s: v["checkpoint_sha256"] for s, v in (reward or {}).get("seeds", {}).items()
        },
        "final_five_seed_val": {
            "feasibility_mean": (reward or {}).get("val_feasibility_mean"),
            "feasibility_sd": (reward or {}).get("val_feasibility_sd"),
            "completion_mean": (reward or {}).get("val_completion_mean"),
            "completion_sd": (reward or {}).get("val_completion_sd"),
            "per_seed": (reward or {}).get("seeds"),
        },
        "fresh_test_protocol": "results/v4/test/PAPER_PROTOCOL.json",
        "fresh_test_lock": "results/v4/test/TEST_LOCK.json",
        "fresh_test_raw": (consumed or {}).get("raw_file"),
        "paired_v4_vs_v3": (stats or {}).get("paired_v4_vs_v3"),
        "baselines": (stats or {}).get("method_overall"),
        "charging_required": ((stats or {}).get("by_charge_class") or {}).get("charging_required"),
        "layout_difficulty": {
            "by_layout": (stats or {}).get("by_layout"),
            "by_length_bin": (stats or {}).get("by_length_bin"),
            "by_n_customers": (stats or {}).get("by_n_customers"),
        },
        "operational_metrics_tables": "results/v4/reward_development/tables/",
        "reward_ablation": ablation,
        "figures": fig_manifest,
        "protocol_seed_start": (protocol or {}).get("confirmatory_test", {}).get("generator_seed_start"),
        "test_lock_n_routes": (lock or {}).get("n_routes"),
        "checkpoint_freeze": {
            "n_v4": (freeze or {}).get("n_v4_checkpoints"),
            "n_v3": (freeze or {}).get("n_v3_checkpoints"),
        },
        "evaluation_consumed": consumed,
        "ci_tests": "python -m pytest tests -q (must be green before TEST)",
        "v1_v2_v3_untouched": True,
        "conservative_message": (protocol or {}).get("conservative_message"),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
