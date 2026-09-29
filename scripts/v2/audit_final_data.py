"""Audit frozen SynthCharge and legacy data. Independent replay of every certificate."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exact.charging_certificate import certificate_sha256, replay_action_trace  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from routing.synthcharge_benchmark import length_bin, load_spec  # noqa: E402
from domain.load_convention import LoadConvention  # noqa: E402
from simulation.simulator import FixedRouteSimulator, run_continue_only  # noqa: E402

CONSUMED = {"c101", "c205", "r110", "r201", "rc102", "rc208"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _actions(trace: list) -> list[tuple]:
    out = []
    for step in trace:
        if step["kind"] == "CONTINUE":
            out.append(("CONTINUE",))
        else:
            out.append(("CHARGE", step["station_id"], float(step["target_soc"])))
    return out


def audit_bundle(directory: Path, *, spec: dict | None, split: str | None) -> dict:
    routes = {route.route_id: route for route in read_jsonl(directory / "corpus.jsonl")}
    raw_rows = {row["route_id"]: row for row in _rows(directory / "corpus.jsonl")}
    certificates = {row["route_id"]: row for row in _rows(directory / "certificates.jsonl")}
    failures = []
    cells = Counter()
    seeds = []
    for route_id, route in routes.items():
        row = raw_rows[route_id]
        cert = certificates.get(route_id)
        if cert is None or cert.get("status") != "certified_feasible":
            failures.append((route_id, "missing_certificate"))
            continue
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance, name=route.physics_profile or "official_evrptwgr")
        trace = cert["trace"]
        if certificate_sha256(trace) != cert["sha256"]:
            failures.append((route_id, "certificate_hash_mismatch"))
        replayed = replay_action_trace(instance, route.customer_ids, _actions(trace), profile=profile)
        if replayed is None:
            failures.append((route_id, "certificate_replay_failed"))
            continue
        if spec is not None:
            simulator = FixedRouteSimulator(
                instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
            )
            continued = run_continue_only(simulator)
            n_charge = sum(1 for step in trace if step["kind"] == "CHARGE")
            if continued.feasible and simulator.state.completed:
                label = "no_charge_required"
            elif n_charge >= 1:
                label = "charging_required"
            else:
                label = "unclassified"
            if label != row["charge_class"]:
                failures.append((route_id, "charge_class_not_reproducible"))
            if length_bin(route.n_customers, spec) != row["length_bin"]:
                failures.append((route_id, "length_bin_mismatch"))
            if row["split"] != split:
                failures.append((route_id, "split_label_mismatch"))
            cells[(row["layout"], row["length_bin"], row["charge_class"])] += 1
            seeds.append(int(row["generator_seed"]))
        elif route.base_instance not in CONSUMED:
            failures.append((route_id, "legacy_parent_outside_consumed_set"))
    return {
        "n_routes": len(routes),
        "failures": failures,
        "cells": {"|".join(key): value for key, value in sorted(cells.items())},
        "seeds": seeds,
    }


def main() -> None:
    spec = load_spec()
    report = {"synthcharge": {}, "failures": []}
    all_seeds = {}
    route_ids = {}
    for split in ("train", "validation", "test"):
        result = audit_bundle(ROOT / "data" / "routes_v2" / "synthcharge_final" / split, spec=spec, split=split)
        quota = spec["quotas"][split]
        for layout in spec["layouts"]:
            for bin_name in spec["length_bins"]:
                for label in ("charging_required", "no_charge_required"):
                    got = result["cells"].get(f"{layout}|{bin_name}|{label}", 0)
                    if got != int(quota[label]):
                        result["failures"].append((f"{split}:{layout}:{bin_name}:{label}", f"quota {got} != {quota[label]}"))
        seeds = result.pop("seeds")
        if len(seeds) != len(set(seeds)):
            result["failures"].append((split, "more_than_one_route_per_instance"))
        start = int(spec["seed_starts"][split])
        if any(not start <= seed < start + int(spec["max_candidate_seeds_per_split"]) for seed in seeds):
            result["failures"].append((split, "seed_outside_split_range"))
        all_seeds[split] = set(seeds)
        route_ids[split] = set(read_route.route_id for read_route in read_jsonl(ROOT / "data" / "routes_v2" / "synthcharge_final" / split / "corpus.jsonl"))
        report["synthcharge"][split] = result
        report["failures"].extend(result["failures"])
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if all_seeds[left] & all_seeds[right]:
            report["failures"].append((f"{left}/{right}", "generator_seed_reused"))
        if route_ids[left] & route_ids[right]:
            report["failures"].append((f"{left}/{right}", "route_reused"))
    legacy = audit_bundle(ROOT / "data" / "routes_v2" / "legacy_evrptwgr_challenge", spec=None, split=None)
    legacy.pop("seeds")
    report["legacy_evrptwgr_challenge"] = legacy
    report["failures"].extend(legacy["failures"])
    for name in ("gold_official", "certified_pyvrp"):
        for route in read_jsonl(ROOT / "data" / "routes_v2" / name / "corpus.jsonl"):
            if route.base_instance in CONSUMED:
                report["failures"].append((name, "consumed_parent_in_development_corpus"))
    audit_rows = _rows(ROOT / "data" / "routes_v2" / "synthcharge_final" / "candidate_audit.jsonl")
    report["candidate_seeds_examined"] = dict(Counter(row["split"] for row in audit_rows))
    reasons = Counter()
    for row in audit_rows:
        for item in row["rejection_reasons"]:
            reasons[item["reason"]] += 1
    report["candidate_route_rejections"] = dict(reasons)
    report["n_failures"] = len(report["failures"])
    out = ROOT / "results" / "v2" / "final" / "data_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=list) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "synthcharge"}, default=list, indent=1))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
