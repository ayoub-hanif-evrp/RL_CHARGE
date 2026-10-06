"""Record one real V4 VAL charging trajectory for Fig. 4 (not TEST).

Selection rule (deterministic, disclosed):
  Among charging-required SynthCharge VALIDATION routes, in stable route_id order,
  choose the first route that is feasible under the frozen V4 seed-42 checkpoint
  and contains at least one charging action.

Does not retrain. Does not touch TEST.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from domain.load_convention import LoadConvention  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from rl.checkpoint import load_hybrid_actor, sha256_file  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from simulation.actions import ContinueAction  # noqa: E402
from simulation.shield import (  # noqa: E402
    CONTINUE_INDEX,
    action_from_discrete,
    evaluate_shield,
    soc_interval_for_station,
    station_ids_of,
)
from simulation.simulator import FixedRouteSimulator  # noqa: E402

SEED = 42
CKPT = ROOT / "checkpoints_v4" / "final_authoritative" / "V4_BASE_NO_L_FAIL" / f"seed_{SEED}" / "best.pt"
OUT = ROOT / "results_v4_paper" / "data" / "illustrative_val_trajectory.json"
VAL_CORPUS = ROOT / "data" / "routes_v2" / "synthcharge_final" / "validation" / "corpus.jsonl"

SELECTION = {
    "split": "validation",
    "charge_class": "charging_required",
    "min_charge_actions": 1,
    "checkpoint_seed": SEED,
    "rule": (
        "First charging-required VALIDATION route in stable route_id order that is "
        "feasible under the frozen V4 seed-42 checkpoint and contains at least one "
        "charging action."
    ),
    "not_test_evidence": True,
}


def _run(route, actor) -> dict | None:
    instance = parse_route_instance(route)
    profile = PhysicsProfile.from_instance(instance, name="synthcharge_linear")
    sim = FixedRouteSimulator(
        instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    sim.time_aware_envelope = True
    actor.ablation = getattr(actor, "ablation", None)

    steps = []
    n_charge = 0
    t = 0
    while not sim.state.completed and t < 10000:
        shield = evaluate_shield(sim)
        if not shield.any_legal:
            return None
        discrete, u = actor.choose(sim, eval_mode=True)
        time_before = float(sim.state.time.value)
        soc_before = float(sim.state.soc.value)
        loc_before = sim.state.current_node_id
        n_events = len(sim.state.events)

        row = {
            "step": t,
            "time_before": time_before,
            "soc_before": soc_before,
            "location_before": loc_before,
            "discrete": int(discrete),
            "u": float(u),
        }
        if discrete == CONTINUE_INDEX:
            row["action"] = "CONTINUE"
            result = sim.step(ContinueAction())
        else:
            station_id = shield.station_ids[int(discrete) - 1]
            interval = soc_interval_for_station(sim, station_id, time_aware=True)
            target = float(interval.map_u(u))
            row.update(
                {
                    "action": "CHARGE",
                    "station_id": station_id,
                    "soc_lower": float(interval.soc_lower),
                    "soc_upper": float(interval.soc_upper),
                    "soc_target": target,
                }
            )
            result = sim.step(action_from_discrete(sim, int(discrete), float(u), time_aware=True))
            n_charge += 1
            new_events = [e.to_dict() for e in sim.state.events[n_events:]]
            charge = next((e for e in new_events if e["kind"] == "charge"), None)
            if charge:
                row["soc_arrival"] = float(charge["soc_before"])
                row["soc_after_charge"] = float(charge["soc_after"])
                row["charge_arrival_time"] = float(charge["arrival_time"])
                row["charge_departure_time"] = float(charge["departure_time"])

        row["time_after"] = float(sim.state.time.value)
        row["soc_after"] = float(sim.state.soc.value)
        row["location_after"] = sim.state.current_node_id
        row["feasible"] = bool(result.feasible)
        if not result.feasible:
            return None
        steps.append(row)
        t += 1

    if not sim.state.completed or n_charge < 1:
        return None

    # Compact SOC timeline for plotting
    timeline = [{"time": 0.0, "soc": 1.0, "kind": "start", "node": instance.depot.string_id}]
    for row in steps:
        if row["action"] == "CONTINUE":
            timeline.append(
                {
                    "time": row["time_after"],
                    "soc": row["soc_after"],
                    "kind": "arrive",
                    "node": row["location_after"],
                }
            )
        else:
            timeline.append(
                {
                    "time": row.get("charge_arrival_time", row["time_before"]),
                    "soc": row.get("soc_arrival", row["soc_before"]),
                    "kind": "station_arrive",
                    "node": row["station_id"],
                    "soc_lower": row["soc_lower"],
                    "soc_upper": row["soc_upper"],
                    "soc_target": row["soc_target"],
                }
            )
            timeline.append(
                {
                    "time": row.get("charge_departure_time", row["time_after"]),
                    "soc": row.get("soc_after_charge", row["soc_after"]),
                    "kind": "charge_depart",
                    "node": row["station_id"],
                    "soc_lower": row["soc_lower"],
                    "soc_upper": row["soc_upper"],
                    "soc_target": row["soc_target"],
                    "u": row["u"],
                }
            )

    return {
        "selection": SELECTION,
        "route_id": route.route_id,
        "raw_instance_id": route.raw_instance_id,
        "seed": SEED,
        "checkpoint": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(CKPT),
        "physics_profile": "synthcharge_linear",
        "customer_ids": list(route.customer_ids),
        "station_ids": list(station_ids_of(sim)),
        "depot_id": instance.depot.string_id,
        "n_charge_actions": n_charge,
        "completed": True,
        "feasible": True,
        "terminal_time": float(sim.state.time.value),
        "terminal_soc": float(sim.state.soc.value),
        "steps": steps,
        "soc_timeline": timeline,
        "not_test_evidence": True,
    }


def main() -> None:
    if not CKPT.is_file():
        raise SystemExit(f"missing checkpoint {CKPT}")
    actor = load_hybrid_actor(CKPT)
    from rl.ablation import AblationConfig

    actor.ablation = AblationConfig(name="FULL", time_aware=True, soc_interval="continuation_to_max")

    meta_rows = [json.loads(line) for line in VAL_CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta_rows = [r for r in meta_rows if r.get("charge_class") == "charging_required"]
    meta_rows.sort(key=lambda r: r["route_id"])
    routes = {route.route_id: route for route in read_jsonl(VAL_CORPUS)}

    chosen = None
    for meta in meta_rows:
        route = routes[meta["route_id"]]
        payload = _run(route, actor)
        if payload is not None:
            payload["layout"] = meta.get("layout")
            payload["length_bin"] = meta.get("length_bin")
            payload["charge_class"] = meta.get("charge_class")
            chosen = payload
            break
    if chosen is None:
        raise SystemExit("no qualifying VAL route found under selection rule")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(OUT.relative_to(ROOT)).replace("\\", "/"),
                "route_id": chosen["route_id"],
                "n_charge_actions": chosen["n_charge_actions"],
                "checkpoint_sha256": chosen["checkpoint_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
