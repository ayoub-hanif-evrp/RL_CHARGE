"""Full Hybrid PPO training loop: train-only sampling, val early stopping, checkpoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from data.paths import CHECKPOINTS_DIR, RESULTS_DIR
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.ablation import AblationConfig
from rl.checkpoint import save_checkpoint
from rl.env import ShieldedRouteEnv
from rl.features import extract_features
from rl.normalization import Normalizer
from rl.ppo import HybridPPO, PPOConfig
from rl.sampler import HierarchicalSampler
from routing.fixed_route import FrozenRoute
from routing.serialize import canonical_dumps

from experiments.actors import HybridPolicyActor
from experiments.dataset import parse_route_instance
from experiments.evaluate import evaluate_policy
from experiments.provenance import run_manifest, sha256_file


def _concat_feature_arrays(vals):
    import numpy as np

    arrays = [np.asarray(value) for value in vals]
    if not arrays:
        return np.zeros((0, 1), dtype=np.float64)
    if arrays[0].ndim == 1:
        return np.stack(arrays, axis=0)
    return np.concatenate(arrays, axis=0)


def fit_normalizer(routes: List[FrozenRoute], ablation: AblationConfig) -> Normalizer:
    bundles = {"global": [], "next": [], "remaining": [], "stations": []}
    for route in routes:
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance)
        sim = ShieldedRouteEnv(
            instance, route, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP, ablation=ablation
        ).simulator
        feat = extract_features(
            sim,
            use_remaining_route=ablation.use_remaining_route,
            use_terrain_load_features=ablation.use_terrain_load_features,
        )
        arrays = feat.as_arrays()
        for key, value in arrays.items():
            bundles[key].append(value)
    stacked = {key: _concat_feature_arrays(vals) for key, vals in bundles.items() if vals}
    return Normalizer.empty().fit(stacked)


def evaluate_routes(routes: List[FrozenRoute], actor, max_routes: Optional[int] = None) -> dict:
    selected = routes if max_routes is None else routes[:max_routes]
    records = []
    for route in selected:
        instance = parse_route_instance(route)
        result = evaluate_policy(instance=instance, route=route, policy=actor, eval_mode=True)
        records.append(result)
    n = max(len(records), 1)
    feas = sum(1 for r in records if r.feasible) / n
    times = [r.completion_time_all_routes() for r in records]
    return {
        "n": len(records),
        "feasibility": feas,
        "mean_completion_all": float(sum(times) / n),
        "records": records,
    }


def train_hybrid_ppo(
    *,
    train_routes: List[FrozenRoute],
    val_routes: List[FrozenRoute],
    config: PPOConfig,
    ablation: Optional[AblationConfig] = None,
    method: str = "HybridPPO",
    device: str = "cpu",
    out_dir: Optional[Path] = None,
    wall_clock_s: Optional[float] = None,
    val_max_routes: int = 16,
) -> dict:
    ablation = ablation or AblationConfig()
    out_dir = Path(out_dir) if out_dir is not None else CHECKPOINTS_DIR / method / f"seed_{config.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = HierarchicalSampler(train_routes, seed=config.seed)
    normalizer = fit_normalizer(train_routes[: min(32, len(train_routes))], ablation)
    trainer = HybridPPO(config, device=device, ablation=ablation)
    started = time.perf_counter()
    best_val = float("inf")
    best_path = out_dir / "best.pt"
    patience = 0
    curves = []
    status = "completed"

    def env_factory():
        sample = sampler.sample()
        instance = parse_route_instance(sample.route)
        profile = PhysicsProfile.from_instance(instance)
        return ShieldedRouteEnv(
            instance,
            sample.route,
            profile,
            LoadConvention.OFFICIAL_REFERENCE_PICKUP,
            normalizer=normalizer,
            ablation=ablation,
        )

    actor = HybridPolicyActor(
        trainer.policy,
        ablation=ablation,
        snap_discrete_u=ablation.discrete_u,
        normalizer=normalizer,
    )
    for update in range(int(config.budget_updates)):
        if wall_clock_s is not None and (time.perf_counter() - started) > wall_clock_s:
            status = "budget_exhausted"
            break
        buffer = trainer.collect_from_factory(env_factory)
        stats = trainer.update(buffer)
        row = {"update": update, "loss": stats["loss"]}
        if update % max(int(config.eval_interval), 1) == 0 or update == config.budget_updates - 1:
            val = evaluate_routes(val_routes, actor, max_routes=val_max_routes)
            row["val_feasibility"] = val["feasibility"]
            row["val_completion_all"] = val["mean_completion_all"]
            score = val["mean_completion_all"]
            if score + 1e-9 < best_val:
                best_val = score
                patience = 0
                ckpt_hash = save_checkpoint(
                    best_path,
                    policy=trainer.policy,
                    optimizer=trainer.optimizer,
                    config=config,
                    normalizer=normalizer,
                    ablation=ablation,
                    extra={"update": update, "val": val["mean_completion_all"]},
                )
                row["checkpoint_sha256"] = ckpt_hash
            else:
                patience += 1
                if patience >= int(config.early_stopping_patience):
                    status = "early_stop"
                    curves.append(row)
                    break
        curves.append(row)

    last_path = out_dir / "last.pt"
    save_checkpoint(
        last_path,
        policy=trainer.policy,
        optimizer=trainer.optimizer,
        config=config,
        normalizer=normalizer,
        ablation=ablation,
        extra={"status": status},
    )
    runtime = time.perf_counter() - started
    manifest = run_manifest(
        method=method,
        seed=config.seed,
        ablation=ablation.name,
        status=status,
        runtime_s=runtime,
        best_val_completion_all=best_val,
        checkpoint=str(best_path if best_path.is_file() else last_path),
        sampling=HierarchicalSampler.name,
        n_train_routes=len(train_routes),
        n_val_routes=len(val_routes),
    )
    if best_path.is_file():
        manifest["checkpoint_sha256"] = sha256_file(best_path)
    (out_dir / "manifest.json").write_text(canonical_dumps(manifest) + "\n", encoding="utf-8")
    (out_dir / "curves.jsonl").write_text(
        "\n".join(canonical_dumps(row) for row in curves) + ("\n" if curves else ""),
        encoding="utf-8",
    )
    results_dir = RESULTS_DIR / "runs" / f"{method}_seed{config.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "manifest.json").write_text(canonical_dumps(manifest) + "\n", encoding="utf-8")
    return manifest
