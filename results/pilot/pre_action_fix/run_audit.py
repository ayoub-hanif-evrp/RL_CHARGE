"""Diagnostic-only TRAIN/VAL audit of the failed Hybrid PPO pilot.

Does not use TEST. Does not retrain. Does not change the method.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.greedy import GreedyFullCharge, GreedyMinimumSufficientCharge
from baselines.lookahead import OneStepLookahead
from domain.load_convention import LoadConvention
from experiments.dataset import load_split_routes, parse_route_instance
from physics.parameters import PhysicsProfile
from rl.ablation import AblationConfig
from rl.checkpoint import load_hybrid_actor
from rl.env import ShieldedRouteEnv
from rl.features import extract_features
from rl.policy import HybridPolicy
from rl.seed import seed_everything
from simulation.actions import ChargeAction, ContinueAction
from simulation.feasibility import InfeasibilityReason
from simulation.shield import CONTINUE_INDEX, action_from_discrete, evaluate_shield
from simulation.simulator import FixedRouteSimulator

OUT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints" / "HybridPPO"
FORBIDDEN = {"test", "testing"}


def _reject_test(split: str) -> None:
    if str(split).lower() in FORBIDDEN:
        raise SystemExit("TEST is forbidden")


def _make_sim(route) -> FixedRouteSimulator:
    instance = parse_route_instance(route)
    profile = PhysicsProfile.from_instance(instance)
    return FixedRouteSimulator(
        instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )


def _reason_name(value) -> str:
    if value is None:
        return "none"
    return getattr(value, "value", str(value))


def instrumented_rollout(simulator: FixedRouteSimulator, choose) -> dict:
    """Roll out a (discrete, u) chooser and record same-station / loop / return."""
    t0_episode = simulator.state.time.value
    h = simulator.horizon
    g = 0.0
    steps = 0
    n_continue = 0
    n_charge = 0
    consecutive_same = 0
    repeat_energy = 0.0
    repeat_time = 0.0
    last_station = None
    loop_guard = False
    reason = None
    max_steps = 10_000
    while not simulator.state.completed and steps < max_steps:
        shield = evaluate_shield(simulator)
        if not shield.any_legal:
            dt_fail = h - simulator.state.time.value
            g += -dt_fail
            reason = "NO_FEASIBLE_ACTION"
            break
        t_before = simulator.state.time.value
        e_before = simulator.state.metrics.total_energy_charged
        discrete, u = choose(simulator)
        if discrete >= len(shield.mask) or not shield.mask[discrete]:
            g += -(h - t_before)
            reason = "MASK_VIOLATION"
            break
        action = action_from_discrete(simulator, discrete, u)
        is_repeat = (
            isinstance(action, ChargeAction)
            and last_station is not None
            and action.station_id == last_station
        )
        result = simulator.step(action)
        t_after = simulator.state.time.value
        e_after = simulator.state.metrics.total_energy_charged
        if not result.feasible:
            g += -(h - t_before)
            reason = _reason_name(result.reason)
            if result.reason is InfeasibilityReason.LOOP_GUARD:
                loop_guard = True
            break
        g += -(t_after - t_before)
        steps += 1
        if isinstance(action, ContinueAction):
            n_continue += 1
            last_station = None
        else:
            n_charge += 1
            last_station = action.station_id
            if is_repeat:
                consecutive_same += 1
                repeat_energy += e_after - e_before
                repeat_time += t_after - t_before
        if simulator.state.n_station_visits_since_last_customer >= (
            simulator.profile.loop_guard_station_visits
        ):
            loop_guard = True
    completed = bool(simulator.state.completed)
    if completed:
        reason = None
    metrics = simulator.metrics(feasible=completed)
    return {
        "feasible": completed,
        "completed": completed,
        "return_value": g,
        "horizon": h,
        "t0": t0_episode,
        "g_plus_H": g + h,
        "reason": reason,
        "n_steps": steps,
        "n_continue": n_continue,
        "n_charge": n_charge,
        "n_station_visits": metrics.number_of_station_visits,
        "consecutive_same_station": consecutive_same,
        "repeat_energy_added": repeat_energy,
        "repeat_time_added": repeat_time,
        "loop_guard_hit": loop_guard or reason == "LOOP_GUARD",
        "completion_time": metrics.route_completion_time.value if completed else h,
        "total_energy_charged": metrics.total_energy_charged.value,
        "total_charging_time": metrics.total_charging_time.value,
    }


def _policy_choose(policy_or_baseline):
    if hasattr(policy_or_baseline, "choose"):
        return lambda sim: policy_or_baseline.choose(sim, eval_mode=True)
    name = getattr(policy_or_baseline, "name", "")
    if name == "GreedyMinimumSufficientCharge":
        from baselines.greedy import _greedy_choose

        return lambda sim: _greedy_choose(sim, 0.0)
    if name == "GreedyFullCharge":
        from baselines.greedy import _greedy_choose

        return lambda sim: _greedy_choose(sim, 1.0)
    if name == "OneStepLookahead":
        from baselines.lookahead import _lookahead_choose

        return _lookahead_choose
    raise TypeError(type(policy_or_baseline))


def run_baselines(splits: tuple[str, ...]) -> list[dict]:
    methods = [
        GreedyMinimumSufficientCharge(),
        GreedyFullCharge(),
        OneStepLookahead(),
    ]
    rows = []
    for split in splits:
        _reject_test(split)
        routes = load_split_routes(split)
        for method in methods:
            chooser = _policy_choose(method)
            for route in routes:
                sim = _make_sim(route)
                stats = instrumented_rollout(sim, chooser)
                rows.append(
                    {
                        "split": split,
                        "method": method.name,
                        "route_id": route.route_id,
                        "base_instance": route.base_instance,
                        "n_customers": route.n_customers,
                        "network_group": route.network_group,
                        **stats,
                    }
                )
            print(f"baselines {split} {method.name}: {len(routes)} routes", flush=True)
    return rows


def summarize_method_split(rows: list[dict], split: str, method: str) -> dict:
    group = [r for r in rows if r["split"] == split and r["method"] == method]
    n = len(group)
    feas = [r for r in group if r["feasible"]]
    fails = [r for r in group if not r["feasible"]]
    visits = [r["n_station_visits"] for r in group]
    reasons = Counter(r["reason"] or "success" for r in group)
    return {
        "split": split,
        "method": method,
        "n_routes": n,
        "n_feasible": len(feas),
        "feasibility_rate": len(feas) / max(n, 1),
        "mean_completion_feasible": (
            statistics.mean(r["completion_time"] for r in feas) if feas else None
        ),
        "median_completion_feasible": (
            statistics.median(r["completion_time"] for r in feas) if feas else None
        ),
        "failure_reasons": dict(Counter(r["reason"] for r in fails)),
        "reason_histogram_including_success": dict(reasons),
        "mean_station_visits": statistics.mean(visits) if visits else 0.0,
        "median_station_visits": statistics.median(visits) if visits else 0.0,
        "sum_consecutive_same_station": sum(r["consecutive_same_station"] for r in group),
        "n_routes_with_consecutive_same_station": sum(
            1 for r in group if r["consecutive_same_station"] > 0
        ),
        "n_loop_guard": sum(1 for r in group if r["loop_guard_hit"]),
    }


def union_feasibility(rows: list[dict], split: str) -> dict:
    by_route = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_route[row["route_id"]].append(row["feasible"])
    n = len(by_route)
    n_union = sum(1 for flags in by_route.values() if any(flags))
    return {
        "split": split,
        "n_routes": n,
        "n_union_feasible": n_union,
        "union_rate": n_union / max(n, 1),
        "note": (
            "Diagnostic lower bound only. Routes outside the union are not "
            "claimed infeasible. This is not an exact feasibility oracle."
        ),
    }


def softmax_continue_charge(logits: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, int, bool]:
    masked = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(masked, dim=-1).squeeze(0)
    p_continue = float(probs[0].item()) if probs.numel() else 0.0
    p_charge = float(probs[1:].sum().item()) if probs.numel() > 1 else 0.0
    n_legal_stations = int(mask.squeeze(0)[1:].sum().item()) if mask.numel() > 1 else 0
    continue_legal = bool(mask.squeeze(0)[0].item())
    return p_continue, p_charge, n_legal_stations, continue_legal


def policy_mass(policy: HybridPolicy, features, ablation: AblationConfig | None = None) -> dict:
    ablation = ablation or AblationConfig()
    with torch.no_grad():
        net = policy.forward(features)
        p_c, p_s, n_legal, cont = softmax_continue_charge(net["logits"], net["mask"])
    m = int(len(features.station_ids))
    n_legal_total = n_legal + int(cont)
    return {
        "m_instance_stations": m,
        "n_legal_stations": n_legal,
        "continue_legal": cont,
        "p_continue": p_c,
        "p_charge": p_s,
        "flat_unmasked_p_continue": 1.0 / (m + 1) if m >= 0 else None,
        "flat_unmasked_p_charge": m / (m + 1) if m >= 0 else None,
        "flat_legal_p_continue": (1.0 / n_legal_total) if (cont and n_legal_total) else 0.0,
        "flat_legal_p_charge": (n_legal / n_legal_total) if n_legal_total else 0.0,
    }


def collect_states(split: str, max_dynamic_per_route: int = 8) -> list[dict]:
    _reject_test(split)
    from baselines.greedy import _greedy_choose

    routes = load_split_routes(split)
    states = []
    for route in routes:
        sim = _make_sim(route)
        states.append({"split": split, "route_id": route.route_id, "kind": "reset", "sim": sim.clone()})
        steps = 0
        while not sim.state.completed and steps < max_dynamic_per_route:
            shield = evaluate_shield(sim)
            if not shield.any_legal:
                break
            discrete, u = _greedy_choose(sim, 0.0)
            if discrete >= len(shield.mask) or not shield.mask[discrete]:
                break
            result = sim.step(action_from_discrete(sim, discrete, u))
            if not result.feasible:
                break
            steps += 1
            states.append(
                {
                    "split": split,
                    "route_id": route.route_id,
                    "kind": "greedy_min_step",
                    "sim": sim.clone(),
                }
            )
    print(f"collected {len(states)} {split} states", flush=True)
    return states


def cardinality_from_states(states: list[dict]) -> dict:
    m_vals = []
    legal_stations = []
    continue_legal_flags = []
    structural_p_continue = []
    for item in states:
        shield = evaluate_shield(item["sim"])
        m = len(shield.station_ids)
        n_legal = int(sum(shield.mask[1:]))
        m_vals.append(m)
        legal_stations.append(n_legal)
        continue_legal_flags.append(int(shield.continue_legal))
        structural_p_continue.append(1.0 / (m + 1))
    return {
        "n_states": len(states),
        "m_stations": {
            "min": min(m_vals) if m_vals else None,
            "max": max(m_vals) if m_vals else None,
            "mean": float(np.mean(m_vals)) if m_vals else None,
            "histogram": dict(Counter(m_vals)),
        },
        "n_legal_stations": {
            "min": min(legal_stations) if legal_stations else None,
            "max": max(legal_stations) if legal_stations else None,
            "mean": float(np.mean(legal_stations)) if legal_stations else None,
            "histogram": {str(k): v for k, v in sorted(Counter(legal_stations).items())},
        },
        "frac_continue_legal": float(np.mean(continue_legal_flags)) if continue_legal_flags else None,
        "mean_structural_unmasked_p_continue": float(np.mean(structural_p_continue))
        if structural_p_continue
        else None,
        "mean_structural_unmasked_p_charge": float(np.mean([1.0 - p for p in structural_p_continue]))
        if structural_p_continue
        else None,
    }


def mass_over_states(label: str, policy: HybridPolicy, states: list[dict], ablation=None, normalizer=None) -> dict:
    ablation = ablation or AblationConfig()
    p_cs = []
    p_ss = []
    for item in states:
        feats = extract_features(
            item["sim"],
            normalizer,
            use_remaining_route=ablation.use_remaining_route,
            use_terrain_load_features=ablation.use_terrain_load_features,
            soc_interval=ablation.soc_interval,
        )
        mass = policy_mass(policy, feats, ablation)
        p_cs.append(mass["p_continue"])
        p_ss.append(mass["p_charge"])
    return {
        "label": label,
        "n_states": len(states),
        "mean_p_continue": float(np.mean(p_cs)) if p_cs else None,
        "mean_p_charge": float(np.mean(p_ss)) if p_ss else None,
        "median_p_continue": float(np.median(p_cs)) if p_cs else None,
        "frac_p_charge_gt_0_8": float(np.mean([p > 0.8 for p in p_ss])) if p_ss else None,
        "frac_p_continue_lt_0_1": float(np.mean([p < 0.1 for p in p_cs])) if p_cs else None,
    }


def checkpoint_same_station(ckpt: Path, splits: tuple[str, ...]) -> dict:
    actor = load_hybrid_actor(ckpt)
    chooser = _policy_choose(actor)
    out = {"checkpoint": str(ckpt), "splits": {}}
    for split in splits:
        _reject_test(split)
        routes = load_split_routes(split)
        rows = []
        for route in routes:
            sim = _make_sim(route)
            stats = instrumented_rollout(sim, chooser)
            rows.append({"route_id": route.route_id, **stats})
        fails = [r for r in rows if not r["feasible"]]
        out["splits"][split] = {
            "n_routes": len(rows),
            "n_feasible": sum(1 for r in rows if r["feasible"]),
            "sum_consecutive_same_station": sum(r["consecutive_same_station"] for r in rows),
            "mean_consecutive_same_station": float(
                np.mean([r["consecutive_same_station"] for r in rows])
            ),
            "sum_repeat_energy": float(sum(r["repeat_energy_added"] for r in rows)),
            "sum_repeat_time": float(sum(r["repeat_time_added"] for r in rows)),
            "mean_repeat_energy_on_repeat_routes": (
                float(
                    np.mean(
                        [r["repeat_energy_added"] for r in rows if r["consecutive_same_station"] > 0]
                    )
                )
                if any(r["consecutive_same_station"] > 0 for r in rows)
                else 0.0
            ),
            "failure_reasons": dict(Counter(r["reason"] for r in fails)),
            "n_loop_guard": sum(1 for r in rows if r["loop_guard_hit"]),
            "frac_continue_actions": (
                sum(r["n_continue"] for r in rows)
                / max(sum(r["n_continue"] + r["n_charge"] for r in rows), 1)
            ),
        }
        print(f"same-station {ckpt.name} {split}", flush=True)
    return out


def reward_invariance(n_routes: int = 12) -> dict:
    """G = sum_feasible -(Δt) + 1_fail(-(H-t)) == -H on failures if t0=0."""
    _reject_test("train")
    routes = load_split_routes("train")
    rng = random.Random(0)
    sample = routes[:n_routes]
    demonstrations = []

    def random_legal(sim):
        shield = evaluate_shield(sim)
        legal = [i for i, ok in enumerate(shield.mask) if ok]
        if not legal:
            return CONTINUE_INDEX, 0.0
        return rng.choice(legal), float(rng.random())

    def always_charge(sim):
        shield = evaluate_shield(sim)
        stations = [i for i in range(1, len(shield.mask)) if shield.mask[i]]
        if stations:
            return stations[0], 0.5
        if shield.continue_legal:
            return CONTINUE_INDEX, 0.0
        return CONTINUE_INDEX, 0.0

    choosers = {
        "greedy_min": _policy_choose(GreedyMinimumSufficientCharge()),
        "random_legal": random_legal,
        "always_first_station_u05": always_charge,
    }
    n_fail = 0
    n_match = 0
    max_abs = 0.0
    for route in sample:
        for name, choose in choosers.items():
            sim = _make_sim(route)
            h = sim.horizon
            stats = instrumented_rollout(sim, choose)
            if stats["feasible"]:
                continue
            n_fail += 1
            err = abs(stats["return_value"] + h)
            max_abs = max(max_abs, err)
            if err < 1e-8:
                n_match += 1
            if len(demonstrations) < 12:
                demonstrations.append(
                    {
                        "route_id": route.route_id,
                        "policy": name,
                        "H": h,
                        "G": stats["return_value"],
                        "G_plus_H": stats["g_plus_H"],
                        "reason": stats["reason"],
                        "n_steps": stats["n_steps"],
                        "n_customers_served_index": sim.state.next_customer_index,
                        "matches_minus_H": err < 1e-8,
                    }
                )
    return {
        "initial_time_is_zero": True,
        "algebra": "G = -t_fail + -(H - t_fail) = -H when t0=0",
        "n_failed_rollouts_checked": n_fail,
        "n_exactly_minus_H": n_match,
        "max_abs_G_plus_H": max_abs,
        "confirmed": n_fail > 0 and max_abs < 1e-8,
        "demonstrations": demonstrations,
        "implication": (
            "Every failed episode on a given route has the same return -H, "
            "regardless of how many customers were served. When feasibility "
            "is near zero this is a sparse, indistinguishable learning signal."
        ),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    splits = ("train", "validation")
    baseline_rows = run_baselines(splits)
    fields = [
        "split",
        "method",
        "route_id",
        "base_instance",
        "n_customers",
        "network_group",
        "feasible",
        "reason",
        "completion_time",
        "n_station_visits",
        "consecutive_same_station",
        "repeat_energy_added",
        "repeat_time_added",
        "loop_guard_hit",
        "return_value",
        "horizon",
        "n_continue",
        "n_charge",
    ]
    write_csv(OUT / "baseline_route_level.csv", baseline_rows, fields)
    summaries = [
        summarize_method_split(baseline_rows, split, method)
        for split in splits
        for method in (
            "GreedyMinimumSufficientCharge",
            "GreedyFullCharge",
            "OneStepLookahead",
        )
    ]
    unions = [union_feasibility(baseline_rows, split) for split in splits]

    train_states = collect_states("train")
    val_states = collect_states("validation")
    card = {
        "train": cardinality_from_states(train_states),
        "validation": cardinality_from_states(val_states),
    }

    seed_everything(0)
    fresh = HybridPolicy(d_model=64, n_heads=4, n_layers=1, dropout=0.0)
    fresh.eval()
    # Subsample for NN forwards: all resets + up to 200 dynamic states per split.
    def subsample(states):
        resets = [s for s in states if s["kind"] == "reset"]
        dyn = [s for s in states if s["kind"] != "reset"]
        return resets + dyn[:: max(1, len(dyn) // 200)][:200]

    train_sub = subsample(train_states)
    val_sub = subsample(val_states)
    masses = {
        "untrained_train": mass_over_states("untrained", fresh, train_sub),
        "untrained_validation": mass_over_states("untrained", fresh, val_sub),
    }
    ckpts = {
        "seed42_best": CHECKPOINTS / "seed_42" / "best.pt",
        "seed43_best": CHECKPOINTS / "seed_43" / "best.pt",
        "seed43_last": CHECKPOINTS / "seed_43" / "last.pt",
    }
    same_station = {}
    for key, path in ckpts.items():
        if not path.is_file():
            same_station[key] = {"missing": str(path)}
            continue
        actor = load_hybrid_actor(path)
        masses[f"{key}_train"] = mass_over_states(
            key, actor.policy, train_sub, actor.ablation, actor.normalizer
        )
        masses[f"{key}_validation"] = mass_over_states(
            key, actor.policy, val_sub, actor.ablation, actor.normalizer
        )
        same_station[key] = checkpoint_same_station(path, splits)

    reward = reward_invariance()
    payload = {
        "label": "failed_pre_paper_hybrid_ppo_pilot",
        "test_used": False,
        "baseline_summaries": summaries,
        "union_feasibility": unions,
        "action_cardinality": card,
        "probability_mass": masses,
        "same_station": same_station,
        "reward_invariance": reward,
        "semantics": {
            "charge_action": (
                "One ChargeAction travels to the station (no-op if already there) "
                "then charges continuously to target_soc under linear dt = g * ΔE."
            ),
            "same_station_repeat_useful": (
                "Under linear charging, a second same-station charge only adds "
                "energy if the first action left SOC below max. The first action "
                "could have chosen u=1 and reached max in one step. Repeats burn "
                "the loop-guard counter and are not scientifically required."
            ),
        },
    }
    (OUT / "audit_summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"unions": unions, "summaries": summaries}, indent=2, default=str))
    print("reward", json.dumps(reward, indent=2, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
