"""Record one illustrative FA-HPPO episode on SynthCharge VAL (not TEST).

Selection rule (disclosed; geometry-only, not performance-based):
  Among layout=RC, length_bin=medium, charge_class=charging_required VAL routes
  that complete under frozen FA-HPPO seed 42 with ≥2 charging actions, choose
  the route maximizing the axis-aligned bounding-box area of visited path nodes.

Does not touch V3 TEST.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from domain.load_convention import LoadConvention  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from rl.checkpoint import load_hybrid_actor  # noqa: E402
from rl.features import extract_features  # noqa: E402
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

OUT = ROOT / "results" / "v3" / "paper" / "case_study" / "illustrative_val_episode.json"
SEED = 42
SELECTION = {
    "split": "validation",
    "layout": "RC",
    "length_bin": "medium",
    "charge_class": "charging_required",
    "min_charge_actions": 2,
    "rule": (
        "Among RC/medium/charging-required SynthCharge VAL routes that complete "
        "under FA-HPPO seed 42 with ≥2 charging actions, select the route "
        "maximizing bounding-box area of visited path nodes (geometry-only)."
    ),
    "checkpoint_seed": SEED,
    "not_test_evidence": True,
    "not_performance_cherry_pick": True,
}

STATION_FEATURE_KEYS = [
    "dist_to_station",
    "travel_time_to_station",
    "energy_to_station",
    "dist_station_to_next",
    "travel_time_station_to_next",
    "energy_station_to_next",
    "arrival_soc_proxy",
    "station_altitude",
    "detour_time",
    "soc_lower",
    "soc_upper",
    "slack_to_next_after_travel",
]


def _bbox_area(nodes: dict, path: list[str]) -> float:
    xs = [nodes[i]["x"] for i in path if i in nodes]
    ys = [nodes[i]["y"] for i in path if i in nodes]
    if len(xs) < 2:
        return 0.0
    return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def _candidate_meta():
    corpus = ROOT / "data/routes_v2/synthcharge_final/validation/corpus.jsonl"
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    cands = [
        r
        for r in rows
        if r.get("layout") == SELECTION["layout"]
        and r.get("length_bin") == SELECTION["length_bin"]
        and r.get("charge_class") == SELECTION["charge_class"]
    ]
    cands.sort(key=lambda r: r["route_id"])
    routes = {route.route_id: route for route in read_jsonl(corpus)}
    return [(routes[r["route_id"]], r) for r in cands]


def _policy_snapshot(actor, sim):
    features = extract_features(
        sim,
        actor.normalizer,
        use_remaining_route=actor.ablation.use_remaining_route,
        use_terrain_load_features=actor.ablation.use_terrain_load_features,
        soc_interval=actor.ablation.soc_interval,
    )
    with torch.no_grad():
        net = actor.policy.forward(features)
        output = actor.policy.act(features, eval_mode=True)
    logits = net["logits"][0].cpu().numpy()
    mask = net["mask"][0].cpu().numpy().astype(bool)
    probs = torch.softmax(net["logits"][0], dim=-1).cpu().numpy()
    alphas = net["alpha"][0].cpu().numpy()
    betas = net["beta"][0].cpu().numpy()
    discrete = int(output.discrete_index.item())
    u = float(output.u.item())
    if discrete == CONTINUE_INDEX:
        u = 0.0
        alpha_sel = None
        beta_sel = None
    else:
        alpha_sel = float(alphas[discrete - 1])
        beta_sel = float(betas[discrete - 1])
    action_names = ["CONTINUE"] + list(features.station_ids)
    station_feat = {}
    for i, sid in enumerate(features.station_ids):
        row = features.stations[i]
        station_feat[sid] = {k: float(row[j]) for j, k in enumerate(STATION_FEATURE_KEYS)}
    return {
        "discrete": discrete,
        "u": u,
        "alpha": alpha_sel,
        "beta": beta_sel,
        "action_names": action_names,
        "logits": [float(x) for x in logits],
        "mask": [bool(x) for x in mask],
        "probs": [float(x) if mask[i] else 0.0 for i, x in enumerate(probs)],
        "alpha_all": [float(x) for x in alphas],
        "beta_all": [float(x) for x in betas],
        "station_features": station_feat,
        "station_ids": list(features.station_ids),
    }


def _events_since(sim, start: int):
    return [e.to_dict() for e in sim.state.events[start:]]


def run_episode(route, meta, actor):
    instance = parse_route_instance(route)
    profile = PhysicsProfile.from_instance(instance, name="synthcharge_linear")
    sim = FixedRouteSimulator(
        instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    sim.time_aware_envelope = True

    nodes = {
        n.string_id: {
            "x": float(n.x),
            "y": float(n.y),
            "type": n.node_type.name if hasattr(n.node_type, "name") else str(n.node_type),
        }
        for n in instance.nodes
    }
    depot_id = instance.depot.string_id
    stations = list(station_ids_of(sim))
    customer_ids = list(route.customer_ids)
    min_soc = float(sim.battery_model.state_from_soc(1.0).min_soc_fraction)
    max_soc = float(sim.battery_model.state_from_soc(1.0).max_soc_fraction)

    decisions = []
    path = [sim.state.current_node_id]
    soc_trace = [
        {
            "index": 0,
            "node": depot_id,
            "kind": "start",
            "soc": float(sim.state.soc.value),
            "time": float(sim.state.time.value),
            "phase": "depart",
        }
    ]
    visit_order = [depot_id]
    steps = 0
    while not sim.state.completed and steps < 10000:
        shield = evaluate_shield(sim)
        t0 = float(sim.state.time.value)
        soc0 = float(sim.state.soc.value)
        loc0 = sim.state.current_node_id
        n_events = len(sim.state.events)
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

        snap = _policy_snapshot(actor, sim)
        discrete, u = snap["discrete"], snap["u"]
        record = {
            "step": steps,
            "time_before": t0,
            "soc_before_decision": soc0,
            "location_before": loc0,
            "next_customer_index": int(sim.state.next_customer_index),
            "discrete": discrete,
            "u": u,
            "continue_legal": bool(shield.continue_legal),
            "policy": {
                "action_names": snap["action_names"],
                "logits": snap["logits"],
                "mask": snap["mask"],
                "probs": snap["probs"],
                "alpha": snap["alpha"],
                "beta": snap["beta"],
                "alpha_all": snap["alpha_all"],
                "beta_all": snap["beta_all"],
            },
            "station_features": snap["station_features"],
        }

        if discrete == CONTINUE_INDEX:
            record["action"] = "CONTINUE"
            result = sim.step(ContinueAction())
            record["location_after"] = sim.state.current_node_id
            new_events = _events_since(sim, n_events)
            travel = next((e for e in new_events if e["kind"] == "travel"), None)
            if travel:
                soc_trace.append(
                    {
                        "index": len(soc_trace),
                        "node": travel["to_node"],
                        "kind": "customer_arrive" if str(travel["to_node"]).startswith("C") else "arrive",
                        "soc": float(sim.state.soc.value) if travel["to_node"] == sim.state.current_node_id else None,
                        "soc_depart": float(soc0),
                        "soc_arrive": float(sim.state.soc.value),
                        "time": float(sim.state.time.value),
                        "phase": "arrive",
                        "from_node": travel["from_node"],
                    }
                )
                # fix arrive SOC from battery_after if present
                if "battery_after" in travel and hasattr(sim, "profile"):
                    # battery energy → use state soc after step (already applied)
                    soc_trace[-1]["soc"] = float(sim.state.soc.value)
                    soc_trace[-1]["soc_arrive"] = float(sim.state.soc.value)
            if record["location_after"] != path[-1]:
                path.append(record["location_after"])
                visit_order.append(record["location_after"])
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
                    "alpha": snap["alpha"],
                    "beta": snap["beta"],
                }
            )
            result = sim.step(action_from_discrete(sim, discrete, u, time_aware=True))
            record["location_after"] = sim.state.current_node_id
            new_events = _events_since(sim, n_events)
            travel = next((e for e in new_events if e["kind"] == "travel"), None)
            charge = next((e for e in new_events if e["kind"] == "charge"), None)
            arrival_soc = float(charge["soc_before"]) if charge else float(soc0)
            depart_soc = float(charge["soc_after"]) if charge else float(sim.state.soc.value)
            record["soc_arrival"] = arrival_soc
            record["soc_departure"] = depart_soc
            if travel:
                # travel depletion into station
                soc_trace.append(
                    {
                        "index": len(soc_trace),
                        "node": station_id,
                        "kind": "station_arrive",
                        "soc": arrival_soc,
                        "soc_depart_prev": float(soc0),
                        "soc_arrive": arrival_soc,
                        "time": float(charge["arrival_time"]) if charge else float(sim.state.time.value),
                        "phase": "arrive",
                        "from_node": travel["from_node"],
                        "station_id": station_id,
                    }
                )
            if charge:
                soc_trace.append(
                    {
                        "index": len(soc_trace),
                        "node": station_id,
                        "kind": "charge_depart",
                        "soc": depart_soc,
                        "soc_arrive": arrival_soc,
                        "soc_target": float(charge["soc_target"]),
                        "soc_lower": float(interval.soc_lower),
                        "soc_upper": float(interval.soc_upper),
                        "u": float(u),
                        "alpha": snap["alpha"],
                        "beta": snap["beta"],
                        "time": float(charge["departure_time"]),
                        "phase": "depart_after_charge",
                        "station_id": station_id,
                    }
                )
            if record["location_after"] != path[-1]:
                path.append(record["location_after"])
                visit_order.append(record["location_after"])

        record["time_after"] = float(sim.state.time.value)
        record["soc_after"] = float(sim.state.soc.value)
        record["feasible"] = bool(result.feasible)
        if not result.feasible:
            record["reason"] = str(result.reason)
            decisions.append(record)
            break
        decisions.append(record)
        steps += 1

    # Ensure return-to-depot appears in visit order
    if sim.state.completed and path[-1] != depot_id:
        path.append(depot_id)
        visit_order.append(depot_id)
    elif sim.state.completed and visit_order[-1] != depot_id:
        visit_order.append(depot_id)

    n_charge = sum(1 for d in decisions if d.get("action") == "CHARGE")
    # Representative hybrid decision: widest feasible SOC interval among charges
    charge_decs = [d for d in decisions if d.get("action") == "CHARGE" and d.get("feasible")]
    representative = None
    if charge_decs:
        representative = max(
            charge_decs,
            key=lambda d: float(d.get("soc_upper", 0) - d.get("soc_lower", 0)),
        )

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
        "checkpoint": str((SYNTH_CKPT / f"seed_{SEED}" / "best.pt").relative_to(ROOT).as_posix()),
        "checkpoint_sha256": sha256(SYNTH_CKPT / f"seed_{SEED}" / "best.pt"),
        "physics_profile": "synthcharge_linear",
        "min_soc_fraction": min_soc,
        "max_soc_fraction": max_soc,
        "completed": bool(sim.state.completed),
        "feasible": bool(sim.state.completed),
        "terminal_soc": float(sim.state.soc.value),
        "terminal_time": float(sim.state.time.value),
        "path_nodes": path,
        "visit_order": visit_order,
        "soc_trace": soc_trace,
        "decisions": decisions,
        "representative_charge_step": representative["step"] if representative else None,
        "bbox_area_visited": _bbox_area(nodes, path),
        "n_charge_actions": n_charge,
        "n_continue_actions": sum(1 for d in decisions if d.get("action") == "CONTINUE"),
    }
    return payload


def select_and_record(actor):
    scored = []
    for route, meta in _candidate_meta():
        ep = run_episode(route, meta, actor)
        ok = (
            ep["completed"]
            and ep["n_charge_actions"] >= SELECTION["min_charge_actions"]
        )
        scored.append(
            {
                "route_id": route.route_id,
                "completed": ep["completed"],
                "n_charge": ep["n_charge_actions"],
                "bbox_area": ep["bbox_area_visited"],
                "eligible": ok,
                "episode": ep if ok else None,
                "meta": meta,
            }
        )
    eligible = [s for s in scored if s["eligible"]]
    if not eligible:
        raise SystemExit("no eligible VAL routes with ≥2 charges")
    eligible.sort(key=lambda s: (-s["bbox_area"], s["route_id"]))
    best = eligible[0]
    ep = best["episode"]
    ep["selection_diagnostics"] = {
        "n_candidates_rc_medium_charge_req": len(scored),
        "n_eligible": len(eligible),
        "ranked": [
            {
                "route_id": s["route_id"],
                "bbox_area": s["bbox_area"],
                "n_charge": s["n_charge"],
                "completed": s["completed"],
                "eligible": s["eligible"],
            }
            for s in sorted(scored, key=lambda x: (-x["bbox_area"], x["route_id"]))
        ],
        "chosen_route_id": best["route_id"],
        "chosen_bbox_area": best["bbox_area"],
    }
    return ep


def main() -> None:
    if OUT.is_file() and "--force" not in sys.argv:
        print(f"exists: {OUT} (pass --force to overwrite)")
        return
    ckpt = SYNTH_CKPT / f"seed_{SEED}" / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    actor = load_hybrid_actor(ckpt)
    payload = select_and_record(actor)
    dump_json(OUT, payload)
    print(
        json.dumps(
            {
                "route_id": payload["route_id"],
                "completed": payload["completed"],
                "n_charge": payload["n_charge_actions"],
                "bbox_area": payload["bbox_area_visited"],
                "representative_charge_step": payload["representative_charge_step"],
                "out": OUT.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
