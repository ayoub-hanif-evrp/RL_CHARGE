"""Fill the predeclared SynthCharge benchmark without looking at a learned policy."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

from data.paths import REPO_ROOT
from domain.load_convention import LoadConvention
from exact.charging_certificate import STATUS_CERTIFIED, solve_charging_certificate
from physics.parameters import PhysicsProfile
from routing.config import PyVRPConfig
from routing.pyvrp_generator import PyVRPGenerator
from routing.synthcharge_instance import PROFILE_NAME, instance_from_synthcharge, write_synthcharge_txt
from routing.v2_corpus import make_frozen_route
from simulation.simulator import FixedRouteSimulator, run_continue_only

SPEC_PATH = REPO_ROOT / "configs" / "v2" / "synthcharge_final_benchmark.json"


def load_spec(path: Path | None = None) -> dict:
    return json.loads((path or SPEC_PATH).read_text(encoding="utf-8"))


def length_bin(n_customers: int, spec: dict) -> Optional[str]:
    for name, bounds in spec["length_bins"].items():
        if int(bounds[0]) <= int(n_customers) <= int(bounds[1]):
            return name
    return None


def empty_quotas(spec: dict, split: str) -> dict:
    quota = spec["quotas"][split]
    cells = {}
    for layout in spec["layouts"]:
        for bin_name in spec["length_bins"]:
            cells[(layout, bin_name)] = {
                "charging_required": int(quota["charging_required"]),
                "no_charge_required": int(quota["no_charge_required"]),
            }
    return cells


def quotas_filled(cells: dict) -> bool:
    return all(count == 0 for counts in cells.values() for count in counts.values())


def candidate_design(spec: dict, offset: int) -> tuple[str, int, int]:
    layout = spec["layouts"][offset % len(spec["layouts"])]
    customers = spec["customer_counts"][(offset // len(spec["layouts"])) % len(spec["customer_counts"])]
    stations = int(spec["stations_by_customers"][str(customers)])
    return layout, int(customers), stations


def classify_route(instance, customer_ids, profile, certificate) -> str:
    simulator = FixedRouteSimulator(
        instance,
        tuple(customer_ids),
        profile,
        LoadConvention.OFFICIAL_REFERENCE_PICKUP,
    )
    continued = run_continue_only(simulator)
    if continued.feasible and simulator.state.completed:
        return "no_charge_required"
    if certificate.status == STATUS_CERTIFIED and int(certificate.n_station_visits) >= 1:
        return "charging_required"
    return "unclassified"


def propose_routes(instance, profile, spec: dict, seed: int):
    config = replace(
        PyVRPConfig.from_toml(),
        seed=int(seed),
        distance_scale=int(spec["pyvrp_distance_scale"]),
        demand_scale=int(spec["pyvrp_demand_scale"]),
        physics_profile=PROFILE_NAME,
    )
    generator = PyVRPGenerator(config, max_iterations_override=int(spec["pyvrp_max_iterations"]))
    routes = generator.generate(instance, profile)
    return sorted(routes, key=lambda route: route.route_id)


def fill_split(
    spec: dict,
    split: str,
    *,
    root: Path,
    generate_instance: Callable,
    max_seeds: Optional[int] = None,
) -> tuple[list, list]:
    """Return accepted route rows and audit records. Raises if the quota is short."""
    start = int(spec["seed_starts"][split])
    limit = int(max_seeds if max_seeds is not None else spec["max_candidate_seeds_per_split"])
    cells = empty_quotas(spec, split)
    accepted = []
    audit = []
    for offset in range(limit):
        if quotas_filled(cells):
            break
        seed = start + offset
        layout, n_customers, n_stations = candidate_design(spec, offset)
        raw = generate_instance(n_customers, n_stations, layout, seed, spec["generator_arguments"])
        instance_id = f"sc_{layout}_n{n_customers}_s{seed}"
        relative = f"data/routes_v2/synthcharge_final/instances/{instance_id}.txt"
        destination = root / relative
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
                route_id=f"v2sc_{instance_id}_{route.vehicle_index:02d}",
                vehicle_index=route.vehicle_index,
                generator="synthcharge_pyvrp_whole_route",
                charging_feasibility_status="certified_feasible",
                seed=seed,
                profile_name=PROFILE_NAME,
            )
            extra = {
                "v2_source": "synthcharge_v1",
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
            print(
                f"{split} seed={seed} accepted {chosen[0].route_id} "
                f"remaining={sum(count for counts in cells.values() for count in counts.values())}",
                flush=True,
            )
    if not quotas_filled(cells):
        missing = {f"{layout}:{bin_name}:{label}": count for (layout, bin_name), counts in cells.items() for label, count in counts.items() if count}
        raise RuntimeError(f"SynthCharge {split} quota shortfall after {limit} seeds: {missing}")
    return accepted, audit


def quota_summary(rows) -> dict:
    counts = Counter((extra["layout"], extra["length_bin"], extra["charge_class"]) for _route, extra in rows)
    return {f"{layout}|{bin_name}|{label}": int(value) for (layout, bin_name, label), value in sorted(counts.items())}
