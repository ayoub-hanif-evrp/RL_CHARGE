"""Record one illustrative FA-HPPO episode on SynthCharge VAL (not TEST).

Selection rule (metadata-only, predeclared):
  lexicographically first layout=RC, length_bin=medium, charge_class=charging_required
  route in data/routes_v2/synthcharge_final/validation/corpus.jsonl

Uses frozen checkpoint seed 42. Does not touch V3 TEST.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3_hppo"))

from domain.load_convention import LoadConvention  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from rl.checkpoint import load_hybrid_actor  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from simulation.actions import ChargeAction, ContinueAction  # noqa: E402
from simulation.shield import (  # noqa: E402
    CONTINUE_INDEX,
    action_from_discrete,
    evaluate_shield,
    soc_interval_for_station,
    station_ids_of,
)
from simulation.simulator import FixedRouteSimulator  # noqa: E402

from common import SYNTH_CKPT, dump_json, sha256  # noqa: E402

OUT = ROOT / "results_paper" / "case_study" / "illustrative_val_episode.json"
SEED = 42
SELECTION = {
    "split": "validation",
    "layout": "RC",
    "length_bin": "medium",
    "charge_class": "charging_required",
    "rule": "lexicographically first route_id among matching SynthCharge VAL rows",
    "checkpoint_seed": SEED,
    "not_test_evidence": True,
}


def select_route():
    corpus = ROOT / "data/routes_v2/synthcharge_final/validation/corpus.jsonl"
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    cands = [
        r
        for r in rows
        if r.get("layout") == SELECTION["layout"]
        and r.get("length_bin") == SELECTION["length_bin"]
        and r.get("charge_class") == SELECTION["charge_class"]
    ]
    if not cands:
        raise SystemExit("no matching VAL route")
    cands.sort(key=lambda r: r["route_id"])
    chosen = cands[0]
    routes = {route.route_id: route for route in read_jsonl(corpus)}
    return routes[chosen["route_id"]], chosen


def main() -> None:
    if OUT.is_file() and "--force" not in sys.argv:
        print(f"exists: {OUT} (pass --force to overwrite)")
        return
    route, meta = select_route()
    ckpt = SYNTH_CKPT / f"seed_{SEED}" / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    actor = load_hybrid_actor(ckpt)
    instance = parse_route_instance(route)
    profile = PhysicsProfile.from_instance(instance, name="synthcharge_linear")
    sim = FixedRouteSimulator(
        instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    sim.time_aware_envelope = True

    nodes = {
        n.string_id: {"x": float(n.x), "y": float(n.y), "type": n.node_type.name if hasattr(n.node_type, "name") else str(n.node_type)}
        for n in instance.nodes
    }
    depot_id = instance.depot.string_id
    stations = list(station_ids_of(sim))
    customer_ids = list(route.customer_ids)

    decisions = []
    path = [sim.state.current_node_id]
    soc_samples = [
        {
            "t": float(sim.state.time.value),
            "soc": float(sim.state.soc.value),
            "node": sim.state.current_node_id,
            "kind": "start",
        }
    ]
    steps = 0
    while not sim.state.completed and steps < 10000:
        shield = evaluate_shield(sim)
        t0 = float(sim.state.time.value)
        soc0 = float(sim.state.soc.value)
        loc0 = sim.state.current_node_id
        if not shield.any_legal:
            decisions.append(
                {
                    "step": steps,
                    "time": t0,
                    "soc": soc0,
                    "location": loc0,
                    "action": "NO_FEASIBLE_ACTION",
                    "feasible": False,
                }
            )
            break
        discrete, u = actor.choose(sim, eval_mode=True)
        record = {
            "step": steps,
            "time_before": t0,
            "soc_before": soc0,
            "location_before": loc0,
            "next_customer_index": int(sim.state.next_customer_index),
            "discrete": int(discrete),
            "u": float(u),
            "continue_legal": bool(shield.continue_legal),
        }
        if discrete == CONTINUE_INDEX:
            record["action"] = "CONTINUE"
            result = sim.step(ContinueAction())
            record["location_after"] = sim.state.current_node_id
        else:
            station_id = shield.station_ids[discrete - 1]
            interval = soc_interval_for_station(sim, station_id, time_aware=True)
            target = interval.map_u(u)
            record.update(
                {
                    "action": "CHARGE",
                    "station_id": station_id,
                    "soc_lower": float(interval.soc_lower),
                    "soc_upper": float(interval.soc_upper),
                    "soc_target": float(target),
                }
            )
            result = sim.step(action_from_discrete(sim, discrete, u, time_aware=True))
            record["location_after"] = sim.state.current_node_id
        record["time_after"] = float(sim.state.time.value)
        record["soc_after"] = float(sim.state.soc.value)
        record["feasible"] = bool(result.feasible)
        if not result.feasible:
            record["reason"] = str(result.reason)
            decisions.append(record)
            break
        if record["location_after"] != path[-1]:
            path.append(record["location_after"])
        soc_samples.append(
            {
                "t": record["time_after"],
                "soc": record["soc_after"],
                "node": record["location_after"],
                "kind": record["action"],
                "soc_lower": record.get("soc_lower"),
                "soc_upper": record.get("soc_upper"),
                "soc_target": record.get("soc_target"),
                "u": record.get("u") if record["action"] == "CHARGE" else None,
            }
        )
        decisions.append(record)
        steps += 1

    payload = {
        "label": "Illustrative SynthCharge validation episode — not TEST evidence",
        "selection": SELECTION,
        "route_id": route.route_id,
        "raw_instance_id": route.raw_instance_id,
        "relative_path": route.relative_path,
        "layout": meta.get("layout"),
        "length_bin": meta.get("length_bin"),
        "charge_class": meta.get("charge_class"),
        "n_customers": route.n_customers,
        "customer_ids": customer_ids,
        "depot_id": depot_id,
        "station_ids": stations,
        "nodes": nodes,
        "checkpoint": ckpt.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(ckpt),
        "physics_profile": "synthcharge_linear",
        "completed": bool(sim.state.completed),
        "feasible": bool(sim.state.completed),
        "terminal_soc": float(sim.state.soc.value),
        "terminal_time": float(sim.state.time.value),
        "path_nodes": path,
        "decisions": decisions,
        "soc_samples": soc_samples,
        "n_charge_actions": sum(1 for d in decisions if d.get("action") == "CHARGE"),
        "n_continue_actions": sum(1 for d in decisions if d.get("action") == "CONTINUE"),
    }
    dump_json(OUT, payload)
    print(json.dumps({"route_id": route.route_id, "completed": payload["completed"], "n_charge": payload["n_charge_actions"], "out": OUT.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
