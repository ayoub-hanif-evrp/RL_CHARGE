"""Five-seed gold TRAIN/VAL ablation B0/B1/B2/B3 for the FA-HPPO paper.

Development evidence only. Does not read V3 TEST routes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3_hppo"))

from experiments.dataset import parse_route_instance  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from rl.ablation import AblationConfig  # noqa: E402
from rl.checkpoint import load_hybrid_actor  # noqa: E402
from rl.ppo import PPOConfig  # noqa: E402
from rl.train_loop import evaluate_routes, train_hybrid_ppo  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from routing.v2_corpus import global_return_scale  # noqa: E402
from simulation.actions import ChargeAction, ContinueAction  # noqa: E402
from simulation.shield import action_from_discrete, dead_end_diagnostic, evaluate_shield  # noqa: E402
from simulation.simulator import FixedRouteSimulator  # noqa: E402
from domain.load_convention import LoadConvention  # noqa: E402

from common import SEEDS, V3, dump_json  # noqa: E402

VARIANTS = {
    "B0": {"time_aware": False, "scale": False},
    "B1": {"time_aware": True, "scale": False},
    "B3": {"time_aware": False, "scale": True},
    "B2": {"time_aware": True, "scale": True},
}
FORBIDDEN = {"c101", "c205", "r110", "r201", "rc102", "rc208"}


def _load_split(split: str):
    payload = json.loads((ROOT / "data" / "splits_v2" / f"gold_{split}.json").read_text(encoding="utf-8"))
    allowed = set(payload["instance_ids"])
    corpus = ROOT / "data" / "routes_v2" / "gold_official" / "corpus.jsonl"
    routes = [route for route in read_jsonl(corpus) if route.raw_instance_id in allowed]
    parents = {route.base_instance for route in routes}
    if parents & FORBIDDEN:
        raise SystemExit(f"TEST parent in gold {split}")
    return routes


def _certificates() -> dict:
    path = ROOT / "data" / "routes_v2" / "gold_official" / "certificates.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["route_id"]] = row
    return out


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return None if not vals else sum(vals) / len(vals)


def _rollout(routes, actor, certificates: dict) -> dict:
    reasons = Counter()
    rows = []
    for route in routes:
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance)
        sim = FixedRouteSimulator(
            instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
        )
        sim.time_aware_envelope = bool(actor.ablation.time_aware)
        failed = False
        reason = None
        steps = 0
        while not sim.state.completed and steps < 10000:
            shield = evaluate_shield(sim)
            if not shield.any_legal:
                failed = True
                reason = "NO_FEASIBLE_ACTION"
                reasons[reason] += 1
                break
            discrete, u = actor.choose(sim, eval_mode=True)
            if discrete == 0:
                result = sim.step(ContinueAction())
            else:
                action = action_from_discrete(sim, discrete, u)
                result = sim.step(action)
            steps += 1
            if not result.feasible:
                failed = True
                reason = result.reason.value if result.reason else "FAILED"
                reasons[reason] += 1
                break
        cert = certificates.get(route.route_id, {})
        visits = 0
        if cert.get("trace"):
            visits = sum(1 for step in cert["trace"] if step.get("kind") == "CHARGE")
        rows.append(
            {
                "route_id": route.route_id,
                "base_instance": route.base_instance,
                "feasible": bool(sim.state.completed and not failed),
                "reason": reason,
                "certificate_station_visits": visits,
            }
        )
    subset = [r for r in rows if int(r.get("certificate_station_visits") or 0) >= 1]
    return {
        "reason_histogram": dict(reasons),
        "no_feasible_action": int(reasons.get("NO_FEASIBLE_ACTION", 0)),
        "charging_required_feasibility": (
            sum(1 for r in subset if r["feasible"]) / len(subset) if subset else None
        ),
        "rows": rows,
    }


def _curve_stats(curves_text: str, best_update) -> dict:
    rows = [json.loads(line) for line in curves_text.splitlines() if line.strip()]
    selected = next((row for row in rows if row.get("update") == best_update), None)
    return {
        "value_loss_mean": _mean(row.get("value_loss") for row in rows),
        "grad_norm_preclip_mean": _mean(row.get("grad_norm_preclip") for row in rows),
        "value_loss_at_best_update": None if selected is None else selected.get("value_loss"),
        "grad_norm_preclip_at_best_update": None if selected is None else selected.get("grad_norm_preclip"),
    }


def run_variant(name: str, seed: int) -> None:
    if seed not in SEEDS:
        raise SystemExit("seeds must be 42..46")
    out_ckpt = ROOT / "checkpoints_v3" / "ablation" / name / f"seed_{seed}"
    dest = V3 / "ablation" / name / f"seed_{seed}"
    if (dest / "validation.json").is_file() and (out_ckpt / "best.pt").is_file():
        print(f"skip completed {name} seed={seed}", flush=True)
        return
    train_routes = _load_split("train")
    val_routes = _load_split("validation")
    config = PPOConfig.from_toml()
    config.seed = int(seed)
    spec = VARIANTS[name]
    ablation = AblationConfig(name="FULL", time_aware=bool(spec["time_aware"]))
    scale = global_return_scale(train_routes) if spec["scale"] else None
    manifest = train_hybrid_ppo(
        train_routes=train_routes,
        val_routes=val_routes,
        config=config,
        ablation=ablation,
        method=f"V3{name}",
        out_dir=out_ckpt,
        return_scale=scale,
    )
    actor = load_hybrid_actor(out_ckpt / "best.pt" if (out_ckpt / "best.pt").is_file() else out_ckpt / "last.pt")
    actor.ablation = ablation
    val = evaluate_routes(val_routes, actor)
    diag = _rollout(val_routes, actor, _certificates())
    curves = (out_ckpt / "curves.jsonl").read_text(encoding="utf-8")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "curves.jsonl").write_text(curves, encoding="utf-8")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "label": "development/validation ablation, not fresh TEST evidence",
        "variant": name,
        "seed": seed,
        "time_aware": spec["time_aware"],
        "return_scale": scale if scale is not None else 1.0,
        "return_scale_enabled": bool(spec["scale"]),
        "n_train": len(train_routes),
        "n_val": len(val_routes),
        "best_update": manifest.get("best_update"),
        "status": manifest.get("status"),
        "parent_balanced_val_feasibility": val["parent_balanced_feasibility"],
        "parent_balanced_completion_all": val["parent_balanced_completion_all"],
        "route_weighted_feasibility": val["feasibility"],
        "mean_completion_all": val["mean_completion_all"],
        "no_feasible_action": diag["no_feasible_action"],
        "charging_required_feasibility": diag["charging_required_feasibility"],
        "optimization": _curve_stats(curves, manifest.get("best_update")),
        "checkpoint": (out_ckpt / "best.pt").relative_to(ROOT).as_posix()
        if (out_ckpt / "best.pt").is_file()
        else None,
    }
    dump_json(dest / "validation.json", summary)
    (dest / "validation_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in diag["rows"]) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS) + ("all",), default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    seeds = list(SEEDS) if args.seed is None else [args.seed]
    for variant in variants:
        for seed in seeds:
            print(f"=== ablation {variant} seed={seed} ===", flush=True)
            run_variant(variant, seed)


if __name__ == "__main__":
    main()
