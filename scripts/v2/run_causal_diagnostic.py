"""Causal TRAIN/VAL diagnostic for V2. Seeds 42 and 43 only. No TEST."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain.load_convention import LoadConvention  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from rl.ablation import AblationConfig  # noqa: E402
from rl.checkpoint import load_hybrid_actor  # noqa: E402
from rl.ppo import PPOConfig  # noqa: E402
from rl.train_loop import evaluate_routes, train_hybrid_ppo  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from routing.v2_corpus import global_return_scale  # noqa: E402
from simulation.actions import ChargeAction, ContinueAction  # noqa: E402
from simulation.shield import dead_end_diagnostic, evaluate_shield  # noqa: E402
from simulation.simulator import FixedRouteSimulator  # noqa: E402

VARIANTS = {
    "B0": {"corpus": "gold", "time_aware": False, "scale": False},
    "B1": {"corpus": "gold", "time_aware": True, "scale": False},
    "B2": {"corpus": "gold", "time_aware": True, "scale": True},
    "B3": {"corpus": "gold", "time_aware": False, "scale": True},
    "B2_pyvrp": {"corpus": "pyvrp", "time_aware": True, "scale": True},
}


def _load_split(corpus: str, split: str):
    payload = json.loads((ROOT / "data" / "splits_v2" / f"{corpus}_{split}.json").read_text(encoding="utf-8"))
    allowed = set(payload["instance_ids"])
    corpus_path = {
        "gold": ROOT / "data" / "routes_v2" / "gold_official" / "corpus.jsonl",
        "pyvrp": ROOT / "data" / "routes_v2" / "certified_pyvrp" / "corpus.jsonl",
    }[corpus]
    routes = [route for route in read_jsonl(corpus_path) if route.raw_instance_id in allowed]
    parents = {route.base_instance for route in routes}
    forbidden = {"c101", "c205", "r110", "r201", "rc102", "rc208"}
    if parents & forbidden:
        raise SystemExit(f"TEST parent in {corpus} {split}")
    return routes


def _certificates(corpus: str) -> dict:
    folder = "gold_official" if corpus == "gold" else "certified_pyvrp"
    path = ROOT / "data" / "routes_v2" / folder / "certificates.jsonl"
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["route_id"]] = row
    return out


def _rollout(routes, actor, certificates: dict) -> dict:
    reasons = Counter()
    dead_reasons = Counter()
    rows = []
    for route in routes:
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance)
        sim = FixedRouteSimulator(
            instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
        )
        sim.time_aware_envelope = bool(actor.ablation.time_aware)
        us = []
        near = 0
        charges = 0
        dead = None
        failed = False
        reason = None
        steps = 0
        while not sim.state.completed and steps < 10000:
            shield = evaluate_shield(sim)
            if not shield.any_legal:
                failed = True
                reason = "NO_FEASIBLE_ACTION"
                dead = dead_end_diagnostic(sim)
                reasons[reason] += 1
                for value in dead["station_reasons"].values():
                    if value:
                        dead_reasons[value] += 1
                if dead["continue_reason"]:
                    dead_reasons[dead["continue_reason"]] += 1
                break
            discrete, u = actor.choose(sim, eval_mode=True)
            if discrete == 0:
                result = sim.step(ContinueAction())
            else:
                if discrete >= len(shield.mask) or not shield.mask[discrete]:
                    failed = True
                    reason = "ILLEGAL_ACTION"
                    reasons[reason] += 1
                    break
                from simulation.shield import action_from_discrete

                action = action_from_discrete(sim, discrete, u)
                if isinstance(action, ChargeAction):
                    charges += 1
                    us.append(float(u))
                    if float(u) >= 0.9:
                        near += 1
                result = sim.step(action)
            steps += 1
            if not result.feasible:
                failed = True
                reason = result.reason.value if result.reason else "FAILED"
                reasons[reason] += 1
                break
        metrics = sim.metrics(feasible=sim.state.completed and not failed)
        cert = certificates.get(route.route_id, {})
        rows.append(
            {
                "route_id": route.route_id,
                "base_instance": route.base_instance,
                "feasible": bool(sim.state.completed and not failed),
                "n_customers": route.n_customers,
                "terrain": route.terrain_variant,
                "family": route.customer_distribution,
                "schedule_type": route.schedule_type,
                "terminal_soc": float(sim.state.soc.value),
                "charging_time": float(metrics.total_charging_time.value),
                "station_visits": int(metrics.number_of_station_visits),
                "completion_time": float(sim.state.time.value) if sim.state.completed and not failed else None,
                "mean_u": (sum(us) / len(us)) if us else None,
                "fraction_u_near_upper": (near / charges) if charges else None,
                "reason": reason,
                "dead_end": dead,
                "certificate_station_visits": cert.get("trace") and sum(
                    1 for step in cert["trace"] if step.get("kind") == "CHARGE"
                ),
                "certificate_completion_time": (cert.get("trace") or [{}])[-1].get("time_after")
                if cert.get("trace")
                else None,
            }
        )
    feasible = [row for row in rows if row["feasible"]]
    return {
        "n": len(rows),
        "n_feasible": len(feasible),
        "n_missed_known_feasible": len(rows) - len(feasible),
        "reason_histogram": dict(reasons),
        "dead_end_reason_histogram": dict(dead_reasons),
        "mean_terminal_soc_feasible": _mean(row["terminal_soc"] for row in feasible),
        "mean_terminal_soc_failed": _mean(row["terminal_soc"] for row in rows if not row["feasible"]),
        "mean_charging_time_feasible": _mean(row["charging_time"] for row in feasible),
        "mean_charging_time_failed": _mean(row["charging_time"] for row in rows if not row["feasible"]),
        "mean_station_visits_feasible": _mean(row["station_visits"] for row in feasible),
        "mean_station_visits_failed": _mean(row["station_visits"] for row in rows if not row["feasible"]),
        "mean_u": _mean(row["mean_u"] for row in rows if row["mean_u"] is not None),
        "mean_fraction_u_near_upper": _mean(
            row["fraction_u_near_upper"] for row in rows if row["fraction_u_near_upper"] is not None
        ),
        "rows": rows,
    }


def _mean(values) -> float | None:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _charging_required(rows: list[dict]) -> dict:
    subset = [row for row in rows if int(row.get("certificate_station_visits") or 0) >= 1]
    by_parent: dict[str, list[dict]] = {}
    for row in subset:
        by_parent.setdefault(str(row["base_instance"]), []).append(row)
    parent_rates = []
    for group in by_parent.values():
        parent_rates.append(sum(1 for row in group if row["feasible"]) / len(group))
    return {
        "n": len(subset),
        "n_feasible": sum(1 for row in subset if row["feasible"]),
        "route_weighted_feasibility": (
            sum(1 for row in subset if row["feasible"]) / len(subset) if subset else None
        ),
        "parent_balanced_feasibility": (sum(parent_rates) / len(parent_rates)) if parent_rates else None,
        "no_feasible_action": sum(1 for row in subset if row.get("reason") == "NO_FEASIBLE_ACTION"),
        "mean_charging_time_feasible": _mean(row["charging_time"] for row in subset if row["feasible"]),
        "mean_station_visits_feasible": _mean(row["station_visits"] for row in subset if row["feasible"]),
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
    spec = VARIANTS[name]
    train_routes = _load_split(spec["corpus"], "train")
    val_routes = _load_split(spec["corpus"], "validation")
    config = PPOConfig.from_toml()
    config.seed = int(seed)
    ablation = AblationConfig(name="FULL", time_aware=bool(spec["time_aware"]))
    scale = global_return_scale(train_routes) if spec["scale"] else None
    out = ROOT / "checkpoints_v2" / name / f"seed_{seed}"
    manifest = train_hybrid_ppo(
        train_routes=train_routes,
        val_routes=val_routes,
        config=config,
        ablation=ablation,
        method=f"V2{name}",
        out_dir=out,
        return_scale=scale,
    )
    actor = load_hybrid_actor(out / "best.pt" if (out / "best.pt").is_file() else out / "last.pt")
    actor.ablation = ablation
    val = evaluate_routes(val_routes, actor)
    diag = _rollout(val_routes, actor, _certificates(spec["corpus"]))
    curves = (out / "curves.jsonl").read_text(encoding="utf-8")
    dest = ROOT / "results" / "v2" / "diagnostics" / name / f"seed_{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "curves.jsonl").write_text(curves, encoding="utf-8")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "variant": name,
        "seed": seed,
        "time_aware": spec["time_aware"],
        "return_scale": scale if scale is not None else 1.0,
        "run_kind": manifest.get("run_kind"),
        "git_sha": manifest.get("git_sha"),
        "git_dirty": manifest.get("git_dirty"),
        "n_train": len(train_routes),
        "n_val": len(val_routes),
        "best_update": manifest.get("best_update"),
        "status": manifest.get("status"),
        "parent_balanced_feasibility": val["parent_balanced_feasibility"],
        "parent_balanced_completion_all": val["parent_balanced_completion_all"],
        "route_weighted_feasibility": val["feasibility"],
        "mean_feasible_routes": val["feasibility"] * val["n"],
        "mean_completion_all": val["mean_completion_all"],
        "no_feasible_action": int(diag["reason_histogram"].get("NO_FEASIBLE_ACTION", 0)),
        "charging_required_subset": _charging_required(diag["rows"]),
        "optimization": _curve_stats(curves, manifest.get("best_update")),
        "diagnostic": {key: value for key, value in diag.items() if key != "rows"},
    }
    (dest / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (dest / "validation_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in diag["rows"]) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if args.seed not in (42, 43):
        raise SystemExit("V2 development diagnostic is limited to seeds 42 and 43")
    run_variant(args.variant, args.seed)


if __name__ == "__main__":
    main()
