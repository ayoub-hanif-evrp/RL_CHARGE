"""Legacy DDQN training on the new simulator. One optimizer; frozen target."""

from __future__ import annotations

import random
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from baselines.legacy_ddqn import LegacyTwoStageDDQN, SOC_LEVELS
from data.paths import CHECKPOINTS_DIR
from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from rl.env import ShieldedRouteEnv
from rl.features import extract_features
from rl.sampler import HierarchicalSampler
from routing.fixed_route import FrozenRoute
from routing.serialize import canonical_dumps
from simulation.shield import CONTINUE_INDEX, action_from_discrete, evaluate_shield

from experiments.dataset import parse_route_instance
from experiments.provenance import run_manifest, sha256_file


def train_ddqn(
    *,
    train_routes: List[FrozenRoute],
    config_seed: int = 42,
    gradient_steps: int = 500,
    batch_size: int = 16,
    target_sync: int = 50,
    replay_size: int = 2000,
    device: str = "cpu",
    d_model: int = 32,
    wall_clock_s: Optional[float] = None,
    out_dir: Optional[Path] = None,
) -> dict:
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
            level = int(torch.argmax(q_soc, dim=-1).item())
        if discrete == 0:
            u = 0.0
        else:
            target = SOC_LEVELS[level]
            from simulation.shield import soc_interval_for_station, station_ids_of

            station_id = station_ids_of(env.simulator)[discrete - 1]
            interval = soc_interval_for_station(env.simulator, station_id)
            span = max(interval.soc_upper - interval.soc_lower, 1e-12)
            u = (min(interval.soc_upper, max(interval.soc_lower, target)) - interval.soc_lower) / span
        t0 = env.simulator.state.time.value
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

    ckpt = out_dir / "best.pt"
    torch.save(
        {"online": agent.online.state_dict(), "seed": config_seed, "d_model": d_model},
        ckpt,
    )
    manifest = run_manifest(
        method="LegacyTwoStageDDQN",
        seed=config_seed,
        status=status,
        runtime_s=time.perf_counter() - started,
        checkpoint=str(ckpt),
        checkpoint_sha256=sha256_file(ckpt),
        n_train_routes=len(train_routes),
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
            q = q + q_soc[0, level]
        with torch.no_grad():
            n_disc, n_soc = agent.target.q_values(next_features)
            n_q = n_disc.max(dim=-1).values[0] + n_soc.max(dim=-1).values[0]
            target = torch.tensor(reward, dtype=q.dtype) + (0.0 if done else agent.gamma * n_q)
        loss = loss + 0.5 * (q - target).pow(2)
    loss = loss / max(len(batch), 1)
    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.online.parameters(), 1.0)
    agent.optimizer.step()
