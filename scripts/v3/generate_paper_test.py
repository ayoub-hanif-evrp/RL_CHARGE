"""Generate the fresh V3-HPPO SynthCharge TEST (policy inspection forbidden).

Writes only under data/routes_v2/synthcharge_v3_test/ and the V3 split JSON.
Does not modify V2 frozen TEST corpora.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
THIRD = ROOT / "third_party" / "SynthCharge_v1.0"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(THIRD))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from data_generator import generate_milp_feasible_instance  # noqa: E402

from routing.synthcharge_benchmark import (  # noqa: E402
    classify_route,
    candidate_design,
    empty_quotas,
    length_bin,
    load_spec,
    propose_routes,
    quota_summary,
    quotas_filled,
)
from exact.charging_certificate import STATUS_CERTIFIED, solve_charging_certificate  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from routing.synthcharge_instance import PROFILE_NAME, instance_from_synthcharge, write_synthcharge_txt  # noqa: E402
from routing.v2_corpus import make_frozen_route, write_route_bundle  # noqa: E402

from common import TEST_CORPUS, TEST_SPLIT, dump_json, git_head  # noqa: E402

SPEC_PATH = ROOT / "results" / "v3" / "configs" / "synthcharge_paper_test.json"
OUT_DIR = ROOT / "data" / "routes_v2" / "synthcharge_v3_test"
INST_DIR = OUT_DIR / "instances"


def _generate(n_customers, n_stations, layout, seed, arguments):
    return generate_milp_feasible_instance(
        n_customers=int(n_customers),
        n_stations=int(n_stations),
        instance_type=layout,
        random_seed=int(seed),
        **arguments,
    )


def fill_v3_test(spec: dict) -> tuple[list, list]:
    split = "test"
    start = int(spec["seed_starts"][split])
    limit = int(spec["max_candidate_seeds_per_split"])
    cells = empty_quotas(spec, split)
    accepted = []
    audit = []
    INST_DIR.mkdir(parents=True, exist_ok=True)
    for offset in range(limit):
        if quotas_filled(cells):
            break
        seed = start + offset
        layout, n_customers, n_stations = candidate_design(spec, offset)
        raw = _generate(n_customers, n_stations, layout, seed, spec["generator_arguments"])
        instance_id = f"sc_{layout}_n{n_customers}_s{seed}"
        relative = f"data/routes_v2/synthcharge_v3_test/instances/{instance_id}.txt"
        destination = ROOT / relative
        instance = instance_from_synthcharge(
            raw,
            instance_id=instance_id,
            relative_path=relative,
            layout=layout,
            seed=seed,
            path=destination,
        )
        write_synthcharge_txt(instance, destination)
        profile = PhysicsProfile.from_instance(instance, name=PROFILE_NAME)
        proposed = propose_routes(instance, profile, spec, seed)
        chosen = None
        reasons = []
        for route in proposed:
            bin_name = length_bin(route.n_customers, spec)
            if bin_name is None:
                reasons.append({"route_id": route.route_id, "reason": "outside_length_bins", "n_customers": route.n_customers})
                continue
            certificate = solve_charging_certificate(
                instance,
                route.customer_ids,
                max_seconds=float(spec["certificate_max_seconds"]),
                max_expansions=int(spec["certificate_max_expansions"]),
                profile=profile,
            )
            if certificate.status != STATUS_CERTIFIED:
                reasons.append(
                    {
                        "route_id": route.route_id,
                        "reason": certificate.status,
                        "n_customers": route.n_customers,
                        "runtime_s": certificate.runtime_s,
                        "n_expansions": certificate.n_expansions,
                    }
                )
                continue
            label = classify_route(instance, route.customer_ids, profile, certificate)
            if label not in cells[(layout, bin_name)]:
                reasons.append({"route_id": route.route_id, "reason": label, "bin": bin_name})
                continue
            if cells[(layout, bin_name)][label] <= 0:
                reasons.append({"route_id": route.route_id, "reason": "quota_full", "label": label, "bin": bin_name})
                continue
            built = make_frozen_route(
                instance,
                route.customer_ids,
                route_id=f"v3sc_{instance_id}_{route.vehicle_index:02d}",
                vehicle_index=route.vehicle_index,
                generator="synthcharge_pyvrp_whole_route_v3",
                charging_feasibility_status="certified_feasible",
                seed=seed,
                profile_name=PROFILE_NAME,
            )
            extra = {
                "v2_source": "synthcharge_v1",
                "v3_paper_test": True,
                "split": split,
                "layout": layout,
                "length_bin": bin_name,
                "charge_class": label,
                "generator_seed": seed,
                "certificate_type": "charging_certificate_search",
                "certificate_sha256": certificate.sha256,
                "certificate_station_visits": certificate.n_station_visits,
                "certificate_completion_time": certificate.completion_time,
                "certificate_trace": certificate.trace,
                "source_commit": spec["source_commit"],
            }
            cells[(layout, bin_name)][label] -= 1
            chosen = (built, extra)
            break
        audit.append(
            {
                "split": split,
                "seed": seed,
                "layout": layout,
                "n_customers": n_customers,
                "n_stations": n_stations,
                "accepted_route_id": None if chosen is None else chosen[0].route_id,
                "rejection_reasons": reasons,
            }
        )
        if chosen is not None:
            accepted.append(chosen)
            remaining = sum(count for counts in cells.values() for count in counts.values())
            print(f"v3test seed={seed} accepted {chosen[0].route_id} remaining={remaining}", flush=True)
    if not quotas_filled(cells):
        missing = {
            f"{layout}:{bin_name}:{label}": count
            for (layout, bin_name), counts in cells.items()
            for label, count in counts.items()
            if count
        }
        raise RuntimeError(f"V3 SynthCharge TEST quota shortfall after {limit} seeds: {missing}")
    return accepted, audit


def main() -> None:
    if "--execute" not in sys.argv:
        raise SystemExit("refusing to generate without --execute")
    protocol = ROOT / "results" / "v3" / "PAPER_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit("PAPER_PROTOCOL.json missing; commit protocol before generation")
    if TEST_CORPUS.is_file():
        raise SystemExit("V3 TEST corpus already exists; refusing overwrite")
    # Refuse if V2 seeds accidentally reused.
    used = set()
    for name in ("synthcharge_train", "synthcharge_validation", "synthcharge_test"):
        path = ROOT / "data" / "splits_v2" / f"{name}.json"
        if path.is_file():
            used.update(json.loads(path.read_text(encoding="utf-8"))["generator_seeds"])
    if any(s >= 400000 for s in used):
        raise SystemExit("unexpected: existing SynthCharge splits already use seed>=400000")
    spec = load_spec(SPEC_PATH)
    rows, audit = fill_v3_test(spec)
    write_route_bundle(
        OUT_DIR,
        rows,
        {
            "benchmark": "synthcharge_v3_hppo_paper_test",
            "split": "test",
            "role": "fresh_external_confirmatory_TEST_for_FA_HPPO_paper",
            "source_commit": spec["source_commit"],
            "physics_profile": "synthcharge_linear",
            "n_routes": len(rows),
            "quota_summary": quota_summary(rows),
            "one_route_per_instance": True,
            "generator_seed_start": 400000,
            "policy_inspection_forbidden_during_generation": True,
            "generation_git_head": git_head(),
        },
    )
    split_payload = {
        "split": "test",
        "benchmark": "synthcharge_v3_hppo_paper_test",
        "role": "fresh_external_confirmatory_TEST_for_FA_HPPO_paper",
        "instance_ids": [route.raw_instance_id for route, _ in rows],
        "route_ids": [route.route_id for route, _ in rows],
        "generator_seeds": [int(extra["generator_seed"]) for _, extra in rows],
        "n_routes": len(rows),
        "quota_summary": quota_summary(rows),
        "seed_start": 400000,
    }
    dump_json(TEST_SPLIT, split_payload)
    audit_path = OUT_DIR / "candidate_audit.jsonl"
    audit_path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":"), sort_keys=True) for row in audit) + "\n",
        encoding="utf-8",
    )
    seeds = set(split_payload["generator_seeds"])
    if seeds & used:
        raise SystemExit("V3 TEST reused a V2 generator seed")
    print(json.dumps({"n_routes": len(rows), "n_seeds": len(seeds), "quota": quota_summary(rows)}, indent=2))


if __name__ == "__main__":
    main()
