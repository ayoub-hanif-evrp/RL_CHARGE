"""Legacy DDQN training on the new simulator. One optimizer; frozen target.

Learning uses TRAIN routes only. Checkpoint selection and early stopping use
VALIDATION. TEST is never loaded here.
"""

from __future__ import annotations

import random
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn

from baselines.legacy_ddqn import LegacyTwoStageDDQN, SOC_LEVELS, double_dqn_next_value
from data.paths import CHECKPOINTS_DIR
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.env import ShieldedRouteEnv
from rl.features import extract_features
from rl.sampler import HierarchicalSampler
from routing.fixed_route import FrozenRoute
from routing.serialize import canonical_dumps
from simulation.shield import evaluate_shield

from experiments.dataset import parse_route_instance
from experiments.provenance import run_manifest, sha256_file


def train_ddqn(
    *,
    train_routes: List[FrozenRoute],
    val_routes: Optional[List[FrozenRoute]] = None,
    config_seed: int = 42,
    gradient_steps: int = 500,
    batch_size: int = 16,
    target_sync: int = 50,
    replay_size: int = 2000,
    device: str = "cpu",
    d_model: int = 32,
    wall_clock_s: Optional[float] = None,
    out_dir: Optional[Path] = None,
    val_interval: int = 50,
    val_max_routes: Optional[int] = None,
    early_stopping_patience: int = 20,
) -> dict:
    if val_routes is None:
        val_routes = []
    out_dir = Path(out_dir) if out_dir is not None else CHECKPOINTS_DIR / "LegacyTwoStageDDQN" / f"seed_{config_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    agent = LegacyTwoStageDDQN(d_model=d_model, lr=1e-3, gamma=1.0)
    agent.online.to(device)
    agent.target.to(device)
    sampler = HierarchicalSampler(train_routes, seed=config_seed)
    replay = deque(maxlen=int(replay_size))
    rng = random.Random(config_seed)
    started = time.perf_counter()
    status = "completed"
    env = _make_env(sampler)
    features = extract_features(env.simulator)
    best_feas = -1.0
    best_completion = float("inf")
    best_path = out_dir / "best.pt"
    patience = 0

    for step in range(int(gradient_steps)):
        if wall_clock_s is not None and (time.perf_counter() - started) > wall_clock_s:
            status = "budget_exhausted"
            break
        shield = evaluate_shield(env.simulator)
        with torch.no_grad():
            q_disc, q_soc = agent.online.q_values(features)
        if rng.random() < max(0.05, 0.3 * (1.0 - step / max(gradient_steps, 1))):
            legal = [i for i, ok in enumerate(shield.mask) if ok]
            discrete = rng.choice(legal) if legal else 0
            level = rng.randrange(len(SOC_LEVELS))
        else:
            discrete = int(torch.argmax(q_disc, dim=-1).item())
            if discrete > 0:
                level = int(torch.argmax(q_soc[0, discrete - 1], dim=-1).item())
            else:
                level = 0
        if discrete == 0:
            u = 0.0
        else:
            target = SOC_LEVELS[level]
            from simulation.shield import soc_interval_for_station, station_ids_of

            station_id = station_ids_of(env.simulator)[discrete - 1]
            interval = soc_interval_for_station(env.simulator, station_id)
            span = max(interval.soc_upper - interval.soc_lower, 1e-12)
            u = (min(interval.soc_upper, max(interval.soc_lower, target)) - interval.soc_lower) / span
        info = env.step(discrete, u)
        next_features = info.features
        replay.append((features, discrete, level, info.reward, next_features, info.done))
        if info.done:
            env = _make_env(sampler)
            features = extract_features(env.simulator)
        else:
            features = next_features
        if len(replay) >= batch_size:
            batch = [replay[rng.randrange(len(replay))] for _ in range(batch_size)]
            _update(agent, batch, device)
        if step > 0 and step % int(target_sync) == 0:
            agent.sync_target()
        if val_routes and (step % max(int(val_interval), 1) == 0 or step == gradient_steps - 1):
            from rl.train_loop import _lexicographic_better, evaluate_routes

            val = evaluate_routes(val_routes, agent, max_routes=val_max_routes)
            if _lexicographic_better(val["feasibility"], val["mean_completion_all"], best_feas, best_completion):
                best_feas = val["feasibility"]
                best_completion = val["mean_completion_all"]
                patience = 0
                torch.save(
                    {"online": agent.online.state_dict(), "seed": config_seed, "d_model": d_model},
                    best_path,
                )
            else:
                patience += 1
                if patience >= int(early_stopping_patience):
                    status = "early_stop"
                    break

    last_path = out_dir / "last.pt"
    torch.save(
        {"online": agent.online.state_dict(), "seed": config_seed, "d_model": d_model},
        last_path,
    )
    if not best_path.is_file():
        torch.save(
            {"online": agent.online.state_dict(), "seed": config_seed, "d_model": d_model},
            best_path,
        )
    ckpt = best_path if best_path.is_file() else last_path
    manifest = run_manifest(
        method="LegacyTwoStageDDQN",
        seed=config_seed,
        status=status,
        runtime_s=time.perf_counter() - started,
        checkpoint=str(ckpt),
        checkpoint_sha256=sha256_file(ckpt),
        n_train_routes=len(train_routes),
        n_val_routes=len(val_routes),
        best_val_feasibility=best_feas if best_feas >= 0.0 else None,
        best_val_completion_all=best_completion if best_completion < float("inf") else None,
        split_used_for_learning="train",
        split_used_for_selection="validation",
    )
    (out_dir / "manifest.json").write_text(canonical_dumps(manifest) + "\n", encoding="utf-8")
    return manifest


def _make_env(sampler: HierarchicalSampler) -> ShieldedRouteEnv:
    sample = sampler.sample()
    instance = parse_route_instance(sample.route)
    profile = PhysicsProfile.from_instance(instance)
    return ShieldedRouteEnv(instance, sample.route, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP)


def _update(agent: LegacyTwoStageDDQN, batch, device: str) -> None:
    loss = 0.0
    agent.online.train()
    for features, discrete, level, reward, next_features, done in batch:
        q_disc, q_soc = agent.online.q_values(features)
        q = q_disc[0, discrete]
        if discrete > 0:
            q = q + q_soc[0, discrete - 1, level]
        with torch.no_grad():
            o_disc, o_soc = agent.online.q_values(next_features)
            t_disc, t_soc = agent.target.q_values(next_features)
            n_q = double_dqn_next_value(o_disc, o_soc, t_disc, t_soc)[0]
            target = torch.tensor(reward, dtype=q.dtype, device=q.device) + (
                0.0 if done else agent.gamma * n_q
            )
        loss = loss + 0.5 * (q - target).pow(2)
    loss = loss / max(len(batch), 1)
    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.online.parameters(), 1.0)
    agent.optimizer.step()
