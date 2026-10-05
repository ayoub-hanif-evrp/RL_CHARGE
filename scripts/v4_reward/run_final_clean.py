"""Authoritative clean final rerun of V4_BASE_NO_L_FAIL on seeds 42–46.

Requires a code-clean git tree (approved experiment outputs ignored).
Does not touch V3 or consume TEST. Writes a new namespace; does not overwrite
development ablation or prior final_clean runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.provenance import code_dirty_paths, code_git_dirty  # noqa: E402
from experiments.stats import logical_checkpoint_path  # noqa: E402
from rl.ablation import AblationConfig  # noqa: E402
from rl.checkpoint import load_hybrid_actor, sha256_file  # noqa: E402
from rl.ppo import PPOConfig  # noqa: E402
from rl.rewards import RewardConfig, RewardKind, compute_c_train  # noqa: E402
from rl.train_loop import assert_learning_split, evaluate_routes, train_hybrid_ppo  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
VARIANT = "V4_BASE_NO_L_FAIL"
OUT_ROOT = ROOT / "results" / "v4_reward" / "final_authoritative" / VARIANT
CKPT_ROOT = ROOT / "checkpoints_v4" / "final_authoritative" / VARIANT
PPO_CFG = ROOT / "configs" / "rl" / "hybrid_ppo_paper.toml"


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(["git", *cmd], cwd=str(ROOT), text=True).strip()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _routes(split: str):
    assert_learning_split(split)
    corpus = ROOT / "data" / "routes_v2" / "synthcharge_final" / split / "corpus.jsonl"
    routes = read_jsonl(corpus)
    test_ids = set()
    for name in ("synthcharge_v3_test.json", "synthcharge_test.json"):
        p = ROOT / "data" / "splits_v2" / name
        if p.is_file():
            test_ids |= set(json.loads(p.read_text(encoding="utf-8"))["route_ids"])
    if any(route.route_id in test_ids for route in routes):
        raise SystemExit("TEST route entered final V4 learning data")
    return routes, corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    dirty_paths = code_dirty_paths()
    dirty = bool(dirty_paths)
    sha = _git(["rev-parse", "HEAD"])
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "working tree has uncommitted source changes; commit fixes before authoritative final run: "
            + ", ".join(dirty_paths[:12])
        )
    if code_git_dirty() != dirty:
        raise SystemExit("internal provenance inconsistency")

    train_routes, train_corpus = _routes("train")
    val_routes, val_corpus = _routes("validation")
    c_train = compute_c_train(train_routes)
    reward = RewardConfig.v4(RewardKind.V4_BASE_NO_L_FAIL, c_train)
    ablation = AblationConfig(name="FULL", time_aware=True, soc_interval="continuation_to_max")
    seeds = list(SEEDS) if args.seed is None else [args.seed]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        out_ckpt = CKPT_ROOT / f"seed_{seed}"
        dest = OUT_ROOT / f"seed_{seed}"
        if (dest / "validation.json").is_file() and (out_ckpt / "best.pt").is_file():
            print(f"skip authoritative {VARIANT} seed={seed}", flush=True)
            continue
        config = PPOConfig.from_toml(PPO_CFG)
        config.seed = int(seed)
        reward.assert_compatible_with_ppo_gamma(float(config.gamma))
        manifest = train_hybrid_ppo(
            train_routes=train_routes,
            val_routes=val_routes,
            config=config,
            ablation=ablation,
            method=f"V4_AUTH_{VARIANT}",
            out_dir=out_ckpt,
            return_scale=1.0,
            reward_config=reward,
        )
        best = out_ckpt / "best.pt" if (out_ckpt / "best.pt").is_file() else out_ckpt / "last.pt"
        actor = load_hybrid_actor(best)
        actor.ablation = ablation
        val = evaluate_routes(val_routes, actor)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("curves.jsonl", "reward_config.json", "manifest.json", "normalizer_provenance.json"):
            src = out_ckpt / name
            if src.is_file():
                (dest / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        ckpt_rel = logical_checkpoint_path(str(best))
        summary = {
            "stage": "V4_FINAL_AUTHORITATIVE",
            "variant": VARIANT,
            "reward_kind": reward.kind.value,
            "human_name": "normalized_time_horizon_reward",
            "c_train": float(c_train),
            "c_train_rule": reward.c_train_rule,
            "seed": int(seed),
            "training_git_sha": sha,
            "code_git_dirty": False,
            "git_sha": sha,
            "git_dirty": False,
            "run_kind": "clean_sha",
            "train_corpus": str(train_corpus.relative_to(ROOT)).replace("\\", "/"),
            "val_corpus": str(val_corpus.relative_to(ROOT)).replace("\\", "/"),
            "train_corpus_sha256": _sha_file(train_corpus),
            "val_corpus_sha256": _sha_file(val_corpus),
            "ppo_config": str(PPO_CFG.relative_to(ROOT)).replace("\\", "/"),
            "ppo_config_sha256": _sha_file(PPO_CFG),
            "checkpoint": ckpt_rel,
            "checkpoint_sha256": sha256_file(best),
            "best_update": manifest.get("best_update"),
            "runtime_s": manifest.get("runtime_s"),
            "parent_balanced_val_feasibility": val["parent_balanced_feasibility"],
            "parent_balanced_completion_all": val["parent_balanced_completion_all"],
            "training_manifest_code_git_dirty": manifest.get("code_git_dirty"),
            "training_manifest_run_kind": manifest.get("run_kind"),
            "confirmatory_test_consumed": False,
            "not_v3_test": True,
        }
        if manifest.get("code_git_dirty") or manifest.get("run_kind") != "clean_sha":
            raise SystemExit(
                f"training manifest not clean for seed={seed}: "
                f"code_git_dirty={manifest.get('code_git_dirty')} run_kind={manifest.get('run_kind')}"
            )
        if "C:/" in str(manifest.get("checkpoint", "")) or "Users/" in str(manifest.get("checkpoint", "")):
            raise SystemExit(f"absolute checkpoint path leaked into training manifest: {manifest.get('checkpoint')}")
        (dest / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
