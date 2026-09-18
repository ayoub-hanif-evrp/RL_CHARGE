"""Inspect Hybrid PPO best.pt on VALIDATION only. Not a training or TEST script."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain.load_convention import LoadConvention
from experiments.dataset import load_split_routes, parse_route_instance
from physics.parameters import PhysicsProfile
from rl.checkpoint import load_hybrid_actor
from rl.features import extract_features
from simulation.actions import ChargeAction, ContinueAction
from simulation.shield import CONTINUE_INDEX, action_from_discrete, evaluate_shield
from simulation.simulator import FixedRouteSimulator


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def inspect_checkpoint(ckpt: Path, split: str = "validation") -> dict:
    if str(split).lower() in {"test", "testing"}:
        raise SystemExit("TEST is forbidden for this pilot inspection")
    actor = load_hybrid_actor(ckpt)
    routes = load_split_routes(split)
    n_continue = 0
    n_charge = 0
    us: list[float] = []
    targets: list[float] = []
    visits: list[int] = []
    entropies: list[float] = []
    nan_inf = False
    loop_episodes = 0
    repeated_same_station = 0
    reasons: Counter[str] = Counter()
    n_feasible = 0
    n_completed = 0
    completions_all: list[float] = []

    for route in routes:
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance)
        sim = FixedRouteSimulator(
            instance,
            route.customer_ids,
            profile,
            LoadConvention.OFFICIAL_REFERENCE_PICKUP,
        )
        station_seq: list[str] = []
        n_visit = 0
        had_repeat = False
        steps = 0
        while not sim.state.completed and steps < 10_000:
            features = extract_features(
                sim,
                actor.normalizer,
                use_remaining_route=actor.ablation.use_remaining_route,
                use_terrain_load_features=actor.ablation.use_terrain_load_features,
                soc_interval=actor.ablation.soc_interval,
            )
            with torch.no_grad():
                out = actor.policy.act(features, eval_mode=True)
            ent = float(out.entropy.item())
            entropies.append(ent)
            if not math.isfinite(ent) or not math.isfinite(float(out.u.item())):
                nan_inf = True
            shield = evaluate_shield(sim)
            if not shield.any_legal:
                reasons["NO_FEASIBLE_ACTION"] += 1
                break
            discrete = int(out.discrete_index.item())
            u = float(out.u.item())
            if discrete == CONTINUE_INDEX:
                u = 0.0
            if discrete >= len(shield.mask) or not shield.mask[discrete]:
                reasons["MASK_VIOLATION"] += 1
                break
            action = action_from_discrete(
                sim, discrete, u, soc_mode=actor.ablation.soc_interval
            )
            if isinstance(action, ContinueAction):
                n_continue += 1
            else:
                n_charge += 1
                us.append(u)
                targets.append(float(action.target_soc))
                n_visit += 1
                sid = action.station_id
                if station_seq and station_seq[-1] == sid:
                    repeated_same_station += 1
                    had_repeat = True
                if len(station_seq) >= 2 and sid == station_seq[-2]:
                    had_repeat = True
                station_seq.append(sid)
            result = sim.step(action)
            if not result.feasible:
                reasons[str(getattr(result.reason, "value", result.reason))] += 1
                break
            steps += 1
        if had_repeat:
            loop_episodes += 1
        visits.append(n_visit)
        completed = bool(sim.state.completed)
        n_completed += int(completed)
        n_feasible += int(completed)
        horizon = float(instance.depot.due_date)
        metrics = sim.metrics(feasible=completed)
        completions_all.append(
            metrics.route_completion_time.value if completed else horizon
        )

    n_actions = n_continue + n_charge
    return {
        "checkpoint": str(ckpt),
        "split": split,
        "n_routes": len(routes),
        "n_feasible": n_feasible,
        "feasibility": n_feasible / max(len(routes), 1),
        "n_completed": n_completed,
        "mean_completion_all": float(np.mean(completions_all) if completions_all else 0.0),
        "n_actions": n_actions,
        "frac_continue": n_continue / max(n_actions, 1),
        "frac_charge": n_charge / max(n_actions, 1),
        "mean_u": float(np.mean(us) if us else 0.0),
        "sd_u": float(np.std(us) if us else 0.0),
        "mean_target_soc": float(np.mean(targets) if targets else 0.0),
        "sd_target_soc": float(np.std(targets) if targets else 0.0),
        "mean_station_visits": float(np.mean(visits) if visits else 0.0),
        "sd_station_visits": float(np.std(visits) if visits else 0.0),
        "episodes_with_station_repeat_or_ababa": loop_episodes,
        "consecutive_same_station_charges": repeated_same_station,
        "mean_eval_entropy": float(np.mean(entropies) if entropies else 0.0),
        "sd_eval_entropy": float(np.std(entropies) if entropies else 0.0),
        "min_eval_entropy": float(np.min(entropies) if entropies else 0.0),
        "nan_or_inf": nan_inf or not _finite(entropies + us + targets),
        "fail_reasons": dict(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = inspect_checkpoint(args.checkpoint, split=args.split)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
