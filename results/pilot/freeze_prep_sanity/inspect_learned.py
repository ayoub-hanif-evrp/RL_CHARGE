"""Inspect a learned-method checkpoint on TRAIN or VALIDATION. TEST is forbidden."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.legacy_ddqn import load_ddqn_agent
from domain.load_convention import LoadConvention
from experiments.dataset import load_split_routes, parse_route_instance
from experiments.evaluate import EpisodeResult
from physics.parameters import PhysicsProfile
from rl.checkpoint import load_hybrid_actor
from rl.train_loop import parent_balanced_metrics
from simulation.actions import ChargeAction
from simulation.feasibility import InfeasibilityReason
from simulation.progress import failure_step_reward
from simulation.shield import action_from_discrete, evaluate_shield
from simulation.simulator import FixedRouteSimulator


def _load_actor(ckpt: Path):
    payload = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
    if "online" in payload:
        agent = load_ddqn_agent(ckpt)
        return agent, "LegacyTwoStageDDQN", payload.get("normalizer_provenance")
    actor = load_hybrid_actor(ckpt)
    extra = payload.get("extra") or {}
    return actor, extra.get("method") or "HybridPPO", extra.get("normalizer_provenance")


def inspect_checkpoint(ckpt: Path, split: str = "validation") -> dict:
    if str(split).lower() in {"test", "testing"}:
        raise SystemExit("TEST is forbidden for freeze-prep sanity inspection")
    actor, method, provenance = _load_actor(ckpt)
    routes = load_split_routes(split)
    n_continue = 0
    n_charge = 0
    us: list[float] = []
    visits: list[int] = []
    n_revisits = 0
    consecutive_same = 0
    loop_guard = 0
    reasons: Counter[str] = Counter()
    n_feasible = 0
    completions_all: list[float] = []
    episode_records = []
    parents = []
    nan_inf = False
    ablation_soc = getattr(getattr(actor, "ablation", None), "soc_interval", "continuation_to_max")

    for route in routes:
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance)
        sim = FixedRouteSimulator(
            instance,
            route.customer_ids,
            profile,
            LoadConvention.OFFICIAL_REFERENCE_PICKUP,
        )
        last_station = None
        n_visit = 0
        steps = 0
        g = 0.0
        episode_loop = False
        reason = None
        while not sim.state.completed and steps < 10_000:
            shield = evaluate_shield(sim)
            if not shield.any_legal:
                g += failure_step_reward(sim, decision_time=sim.state.time.value)
                reason = "NO_FEASIBLE_ACTION"
                break
            discrete, u = actor.choose(sim, eval_mode=True)
            if not math.isfinite(float(u)):
                nan_inf = True
            if discrete >= len(shield.mask) or not shield.mask[discrete]:
                g += failure_step_reward(sim, decision_time=sim.state.time.value)
                reason = "MASK_VIOLATION"
                break
            action = action_from_discrete(sim, discrete, u, soc_mode=ablation_soc)
            visited = set(sim.state.stations_visited_since_progress)
            t0 = sim.state.time.value
            if isinstance(action, ChargeAction) and action.station_id in visited:
                n_revisits += 1
            if (
                isinstance(action, ChargeAction)
                and last_station is not None
                and action.station_id == last_station
            ):
                consecutive_same += 1
            result = sim.step(action)
            if not result.feasible:
                g += failure_step_reward(sim, decision_time=t0)
                reason = getattr(result.reason, "value", str(result.reason))
                if result.reason is InfeasibilityReason.LOOP_GUARD:
                    episode_loop = True
                break
            g += -(sim.state.time.value - t0)
            steps += 1
            if discrete == 0:
                n_continue += 1
                last_station = None
            else:
                n_charge += 1
                last_station = getattr(action, "station_id", last_station)
                us.append(float(u))
            n_visit = sim.state.metrics.number_of_station_visits
            if sim.state.n_station_visits_since_last_customer >= sim.profile.loop_guard_station_visits:
                episode_loop = True
        completed = bool(sim.state.completed)
        h = float(instance.depot.due_date)
        if completed:
            reason = None
            n_feasible += 1
        else:
            reasons[reason or "unknown"] += 1
        if episode_loop or reason == "LOOP_GUARD":
            loop_guard += 1
        visits.append(n_visit)
        completions_all.append(sim.state.time.value if completed else h)
        episode_records.append(
            EpisodeResult(
                route_id=route.route_id,
                feasible=completed,
                completed=completed,
                route_completion_time=sim.state.time.value,
                total_net_energy=0.0,
                total_distance=0.0,
                n_station_visits=n_visit,
                return_value=g,
                horizon=h,
                reason=reason,
            )
        )
        parents.append(str(route.base_instance or route.raw_instance_id))

    n_actions = n_continue + n_charge
    balanced = parent_balanced_metrics(parents, episode_records)
    return {
        "checkpoint": str(ckpt),
        "method": method,
        "split": split,
        "n_routes": len(routes),
        "n_feasible": n_feasible,
        "feasibility": n_feasible / max(len(routes), 1),
        "parent_balanced_feasibility": balanced["feasibility"],
        "parent_balanced_completion_all": balanced["completion_all"],
        "mean_completion_all": float(np.mean(completions_all)) if completions_all else None,
        "frac_continue": n_continue / max(n_actions, 1),
        "frac_charge": n_charge / max(n_actions, 1),
        "mean_u": float(np.mean(us)) if us else None,
        "mean_station_visits": float(np.mean(visits)) if visits else 0.0,
        "repeated_station_actions": n_revisits,
        "consecutive_same_station_charges": consecutive_same,
        "loop_guard_hits": loop_guard,
        "fail_reasons": dict(reasons),
        "n_zero_charge_noop": int(reasons.get("ZERO_CHARGE_NOOP", 0)),
        "n_no_feasible_action": int(reasons.get("NO_FEASIBLE_ACTION", 0)),
        "n_station_revisit": int(reasons.get("STATION_REVISIT", 0)),
        "nan_or_inf": nan_inf,
        "normalizer_provenance": provenance,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = inspect_checkpoint(args.ckpt, args.split)
    text = json.dumps(payload, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
