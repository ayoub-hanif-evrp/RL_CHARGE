"""V2 certified-route construction. Reads V1 routes; never writes them."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from data.parser import parse_instance
from data.paths import RAW_EVRPTW_GR_DIR, REPO_ROOT, ROUTES_DIR
from experiments.v2_scope import (
    CONSUMED_TEST_PARENT_SET,
    DEV_PARENT_SET,
    TRAIN_PARENT_SET,
    VAL_PARENT_SET,
    split_for_parent,
)
from physics.network import DirectedArcNetwork
from physics.parameters import PhysicsProfile
from routing.audit import routing_time_window_feasible
from routing.fixed_route import FrozenRoute
from routing.official_replay import certified_charge_to_max_trace
from routing.serialize import canonical_dumps, read_jsonl

from exact.charging_certificate import (
    STATUS_CERTIFIED,
    STATUS_EXHAUSTED,
    STATUS_TIMEOUT,
    certificate_sha256,
    solve_charging_certificate,
)

# First budget is the screen. Later budgets are used only for spans that still
# block a complete cover. Timeout and exhaustion stay unverified.
SHORT_BUDGETS = (
    (1.0, 4_000),
    (4.0, 12_000),
    (12.0, 40_000),
    (30.0, 80_000),
)
FULL_BUDGETS = (
    (3.0, 8_000),
    (15.0, 30_000),
)
ESCALATION_CAP_S = 180.0

XLSX_SHA256 = "193f74325ae508dcfeebd1a22b508b2e1b684d986965cb31b353e3df48199470"
MAPPING_PATH = REPO_ROOT / "results" / "pilot" / "correctness_audit" / "official_route_plan_mapping.json"


def instance_index() -> dict[str, Path]:
    files = {}
    for path in RAW_EVRPTW_GR_DIR.rglob("*.txt"):
        files.setdefault(path.stem, path)
    return files


def customer_only_distance(instance, customer_ids: Sequence[str]) -> float:
    profile = PhysicsProfile.from_instance(instance)
    network = DirectedArcNetwork(instance, profile.average_velocity)
    nodes = [instance.depot.string_id, *customer_ids, instance.depot.string_id]
    total = 0.0
    for left, right in zip(nodes, nodes[1:]):
        total += float(network.arc(left, right).distance.value)
    return total


def route_demand(instance, customer_ids: Sequence[str]) -> float:
    return float(sum(instance.node_by_id(cid).demand for cid in customer_ids))


def make_frozen_route(
    instance,
    customer_ids: Sequence[str],
    *,
    route_id: str,
    vehicle_index: int,
    generator: str,
    charging_feasibility_status: str,
    seed: int = 0,
    profile_name: str | None = None,
) -> FrozenRoute:
    meta = instance.metadata
    profile = PhysicsProfile.from_instance(instance, name=profile_name or "official_evrptwgr")
    demand = route_demand(instance, customer_ids)
    distance = customer_only_distance(instance, customer_ids)
    digest = hashlib.sha256(Path(meta.path).read_bytes()).hexdigest()
    capacity_ok = demand <= float(profile.payload_capacity_kg) + 1e-9
    if profile.name == "official_evrptwgr":
        capacity_policy = "official_evrptwgr_payload_capacity_3650kg"
    elif profile.name == "synthcharge_linear":
        capacity_policy = "synthcharge_instance_load_capacity"
    else:
        capacity_policy = profile.name
    tw_ok = routing_time_window_feasible(instance, tuple(customer_ids), profile.name)
    if profile.name == "synthcharge_linear":
        source_dataset = "SynthCharge"
        doi = "10.48550/arXiv.2603.03230"
    else:
        source_dataset = "EVRPTW-GR"
        doi = "10.17632/srfdbp2twv.1"
    return FrozenRoute(
        route_id=route_id,
        source_dataset=source_dataset,
        doi=doi,
        raw_instance_id=meta.instance_id,
        relative_path=meta.relative_path,
        network_group=meta.network_group.value,
        terrain_variant=meta.terrain_variant.value,
        customer_distribution=str(meta.customer_distribution),
        schedule_type=int(meta.schedule_type),
        generator=generator,
        generator_version="v2",
        seed=int(seed),
        stop="certificate",
        n_iterations=0,
        config_hash="",
        physics_profile=profile.name,
        capacity_policy=capacity_policy,
        distance_scale=1,
        demand_scale=1,
        rounding_policy="unscaled",
        routing_problem="frozen_customer_sequence",
        python="",
        os_name="",
        arch="",
        instance_sha256=digest,
        vehicle_index=int(vehicle_index),
        depot_id=instance.depot.string_id,
        customer_ids=tuple(customer_ids),
        route_demand=demand,
        route_distance=distance,
        route_duration_lower_bound=distance,
        n_customers=len(tuple(customer_ids)),
        routing_feasible=bool(capacity_ok),
        charging_feasibility_status=charging_feasibility_status,
        generation_runtime_s=0.0,
        routing_objective=distance,
        base_instance=meta.base_instance,
        customer_folder=meta.customer_folder,
        n_vehicles=1,
        total_distance=distance,
        fixed_vehicle_cost=0,
        lexicographic_objective=distance,
        route_source_instance_id=meta.instance_id,
        terrain_reuse=False,
        routing_tw_feasible=bool(tw_ok),
    )


def dumps_v2(route: FrozenRoute, extra: dict) -> str:
    payload = route.to_dict()
    payload.update(extra)
    return canonical_dumps(payload)


def global_return_scale(routes: Iterable[FrozenRoute]) -> float:
    """One positive scale: max depot horizon on the supplied routes.

    Callers must pass TRAIN routes only. Validation and TEST must not be included.
    """
    horizons = []
    for route in routes:
        if getattr(route, "physics_profile", "") == "synthcharge_linear":
            from experiments.dataset import parse_route_instance

            instance = parse_route_instance(route)
        else:
            instance = parse_instance(RAW_EVRPTW_GR_DIR / route.relative_path)
        horizons.append(float(instance.depot.due_date))
    if not horizons:
        raise ValueError("return scale requires at least one TRAIN route")
    scale = max(horizons)
    if scale <= 0.0:
        raise ValueError("depot horizon scale must be positive")
    return float(scale)


def load_mapping() -> dict:
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def iter_gold_tours(mapping: dict):
    for tour in mapping["tours"]:
        parent = str(tour.get("base_instance") or "")
        replay = tour.get("replay") or {}
        if parent in CONSUMED_TEST_PARENT_SET or parent not in DEV_PARENT_SET:
            continue
        if not replay.get("feasible"):
            continue
        yield tour


def iter_legacy_tours(mapping: dict):
    """Successful official tours of the six historically consumed V1 TEST parents."""
    for tour in mapping["tours"]:
        parent = str(tour.get("base_instance") or "")
        replay = tour.get("replay") or {}
        if parent not in CONSUMED_TEST_PARENT_SET:
            continue
        if not replay.get("feasible"):
            continue
        yield tour


def build_gold_routes(mapping: dict | None = None) -> tuple[list[tuple[FrozenRoute, dict]], dict]:
    mapping = mapping or load_mapping()
    files = instance_index()
    seen = set()
    n_duplicate = 0
    n_replay_failed = 0
    rows = []
    per_instance_index: dict[str, int] = defaultdict(int)
    for tour in iter_gold_tours(mapping):
        instance_id = str(tour["instance_id"])
        key = (instance_id, tuple(tour["customer_ids"]))
        if key in seen:
            n_duplicate += 1
            continue
        seen.add(key)
        path = files.get(instance_id)
        if path is None:
            raise FileNotFoundError(instance_id)
        instance = parse_instance(path)
        if instance.metadata.base_instance in CONSUMED_TEST_PARENT_SET:
            raise AssertionError("V1 TEST parent entered the gold corpus")
        replayed = certified_charge_to_max_trace(instance, list(tour["tour"]))
        if replayed is None:
            n_replay_failed += 1
            continue
        customers, _actions, trace = replayed
        vehicle_index = per_instance_index[instance_id]
        per_instance_index[instance_id] += 1
        route = make_frozen_route(
            instance,
            customers,
            route_id=f"v2gold_{instance_id}_{vehicle_index:03d}",
            vehicle_index=vehicle_index,
            generator="official_evrptwgr_route_plan",
            charging_feasibility_status="certified_feasible",
        )
        extra = {
            "v2_source": "official_evrptwgr_route_plan",
            "split": split_for_parent(route.base_instance),
            "certificate_type": "charge_to_max_replay",
            "certificate_sha256": certificate_sha256(trace),
            "certificate_source_tour": list(tour["tour"]),
            "certificate_completion_time": float(trace[-1]["time_after"]),
            "certificate_station_visits": sum(1 for step in trace if step["kind"] == "CHARGE"),
            "source_xlsx_sha256": XLSX_SHA256,
            "certificate_trace": trace,
        }
        rows.append((route, extra))
    summary = {
        "n_routes": len(rows),
        "n_duplicate_exact_instance_sequences": n_duplicate,
        "n_replay_failed": n_replay_failed,
        "n_train": sum(1 for _route, extra in rows if extra["split"] == "train"),
        "n_validation": sum(1 for _route, extra in rows if extra["split"] == "validation"),
        "n_train_parents": len({route.base_instance for route, extra in rows if extra["split"] == "train"}),
        "n_validation_parents": len({route.base_instance for route, extra in rows if extra["split"] == "validation"}),
        "test_parents_excluded": sorted(CONSUMED_TEST_PARENT_SET),
    }
    return rows, summary


def build_legacy_challenge_routes(mapping: dict | None = None) -> tuple[list[tuple[FrozenRoute, dict]], dict]:
    """Official charge-to-max skeletons for the six consumed V1 TEST parents only.

    This set is a legacy same-domain challenge. It is not a fresh TEST and
    must never enter V2 TRAIN or VAL.
    """
    mapping = mapping or load_mapping()
    files = instance_index()
    seen = set()
    n_duplicate = 0
    n_replay_failed = 0
    rows = []
    per_instance_index: dict[str, int] = defaultdict(int)
    for tour in iter_legacy_tours(mapping):
        instance_id = str(tour["instance_id"])
        key = (instance_id, tuple(tour["customer_ids"]))
        if key in seen:
            n_duplicate += 1
            continue
        seen.add(key)
        path = files.get(instance_id)
        if path is None:
            raise FileNotFoundError(instance_id)
        instance = parse_instance(path)
        if instance.metadata.base_instance not in CONSUMED_TEST_PARENT_SET:
            raise AssertionError("legacy challenge admitted a non-consumed parent")
        replayed = certified_charge_to_max_trace(instance, list(tour["tour"]))
        if replayed is None:
            n_replay_failed += 1
            continue
        customers, _actions, trace = replayed
        vehicle_index = per_instance_index[instance_id]
        per_instance_index[instance_id] += 1
        route = make_frozen_route(
            instance,
            customers,
            route_id=f"v2legacy_{instance_id}_{vehicle_index:03d}",
            vehicle_index=vehicle_index,
            generator="official_evrptwgr_route_plan",
            charging_feasibility_status="certified_feasible",
        )
        extra = {
            "v2_source": "official_evrptwgr_route_plan",
            "benchmark_role": "legacy_same_domain_challenge",
            "split": "legacy_same_domain_challenge",
            "certificate_type": "charge_to_max_replay",
            "certificate_sha256": certificate_sha256(trace),
            "certificate_source_tour": list(tour["tour"]),
            "certificate_completion_time": float(trace[-1]["time_after"]),
            "certificate_station_visits": sum(1 for step in trace if step["kind"] == "CHARGE"),
            "source_xlsx_sha256": XLSX_SHA256,
            "certificate_trace": trace,
        }
        rows.append((route, extra))
    summary = {
        "role": "legacy_same_domain_challenge",
        "not_a_fresh_test": True,
        "n_routes": len(rows),
        "n_duplicate_exact_instance_sequences": n_duplicate,
        "n_replay_failed": n_replay_failed,
        "parents": sorted({route.base_instance for route, _extra in rows}),
    }
    return rows, summary


def _cuts_key(cuts: tuple) -> tuple:
    return tuple((int(i), int(j)) for i, j in cuts)


def _dp_on_range(instance, customers, certified, left: int, right: int):
    inf = (10**9, float("inf"), tuple())
    span = right - left
    best = [inf for _ in range(span + 1)]
    best[0] = (0, 0.0, tuple())
    prev = [None for _ in range(span + 1)]
    for i in range(left, right):
        local_i = i - left
        if best[local_i][0] >= 10**9:
            continue
        for j in range(i + 1, right + 1):
            result = certified.get((i, j))
            if result is None:
                continue
            dist = customer_only_distance(instance, customers[i:j])
            cuts = best[local_i][2] + ((i, j),)
            cand = (best[local_i][0] + 1, best[local_i][1] + dist, _cuts_key(cuts))
            local_j = j - left
            if cand < best[local_j]:
                best[local_j] = cand
                prev[local_j] = i
    return certified, prev, best[span]


def _walk_partition(certified, prev, right: int, left: int) -> list:
    certificates = []
    cursor = right
    while cursor > left:
        found = prev[cursor - left]
        if found is None or (found, cursor) not in certified:
            raise AssertionError("certified partition predecessor missing")
        certificates.append(certified[(found, cursor)])
        cursor = found
    certificates.reverse()
    return certificates


def partition_customers(
    instance,
    customer_ids: Sequence[str],
    search: Callable[[Sequence[str]], object],
) -> dict:
    """Minimum certified contiguous cover of one frozen customer order."""
    customers = list(customer_ids)
    n = len(customers)
    certified = {}
    timeouts = 0
    exhausted = 0
    for i in range(n):
        for j in range(i + 1, n + 1):
            result = search(customers[i:j])
            status = getattr(result, "status", "")
            if status == STATUS_CERTIFIED:
                if not getattr(result, "trace", None):
                    raise AssertionError("certified subroute is missing a replayed trace")
                certified[(i, j)] = result
            elif status == "unverified_timeout":
                timeouts += 1
            elif status == "unverified_search_exhausted":
                exhausted += 1
            elif "infeasible" in str(status):
                raise AssertionError(f"search returned an infeasibility claim: {status}")
    _certs, prev, best_end = _dp_on_range(instance, customers, certified, 0, n)
    if best_end[0] >= 10**9:
        # An incomplete cover is not a benchmark route. Callers must quarantine
        # the whole source sequence instead of dropping the unresolved customers.
        return {
            "covered": False,
            "certificates": [],
            "excluded": [
                {
                    "customer_id": customers[i],
                    "index": i,
                    "reason": "unresolved_certificate_search",
                }
                for i in range(n)
                if (i, i + 1) not in certified
            ],
            "timeouts": timeouts,
            "exhausted": exhausted,
            "n_subroutes": 0,
        }
    certificates = _walk_partition(certified, prev, n, 0)
    return {
        "covered": True,
        "certificates": certificates,
        "excluded": [],
        "timeouts": timeouts,
        "exhausted": exhausted,
        "n_subroutes": len(certificates),
    }


def write_route_bundle(directory: Path, rows: list[tuple[FrozenRoute, dict]], metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    corpus_lines = []
    cert_lines = []
    manifest_rows = []
    for route, extra in rows:
        public = {key: value for key, value in extra.items() if key != "certificate_trace"}
        corpus_lines.append(dumps_v2(route, public))
        cert_lines.append(
            canonical_dumps(
                {
                    "route_id": route.route_id,
                    "sha256": extra.get("certificate_sha256"),
                    "certificate_type": extra.get("certificate_type"),
                    "trace": extra.get("certificate_trace"),
                    "status": "certified_feasible",
                }
            )
        )
        manifest_rows.append(
            {
                "route_id": route.route_id,
                "raw_instance_id": route.raw_instance_id,
                "base_instance": route.base_instance,
                "split": extra.get("split"),
                "n_customers": route.n_customers,
                "charging_feasibility_status": route.charging_feasibility_status,
                "certificate_sha256": extra.get("certificate_sha256"),
                "certificate_type": extra.get("certificate_type"),
            }
        )
    (directory / "corpus.jsonl").write_text("\n".join(corpus_lines) + ("\n" if corpus_lines else ""), encoding="utf-8")
    (directory / "certificates.jsonl").write_text(
        "\n".join(cert_lines) + ("\n" if cert_lines else ""), encoding="utf-8"
    )
    with (directory / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]) if manifest_rows else ["route_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    (directory / "corpus_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_v1_dev_routes() -> list[FrozenRoute]:
    routes = []
    for route in read_jsonl(ROUTES_DIR / "corpus.jsonl"):
        if route.base_instance in CONSUMED_TEST_PARENT_SET:
            continue
        if route.base_instance not in DEV_PARENT_SET:
            continue
        routes.append(route)
    return routes


def _reachability(certified: dict, n: int) -> list[bool]:
    reach = [False] * (n + 1)
    reach[0] = True
    for i in range(n):
        if not reach[i]:
            continue
        for j in range(i + 1, n + 1):
            if (i, j) in certified:
                reach[j] = True
    return reach


def _uncovered_indices(certified: dict, reach: list[bool], n: int) -> list[int]:
    covered = [False] * n
    for i in range(n):
        if not reach[i]:
            continue
        for j in range(i + 1, n + 1):
            if (i, j) not in certified:
                continue
            for k in range(i, j):
                covered[k] = True
    return [index for index, flag in enumerate(covered) if not flag]


def _priority_spans(n: int, uncovered: Sequence[int], certified: dict, reach: list[bool], max_length: int) -> list[tuple[int, int]]:
    uncovered_set = set(uncovered)
    spans = []
    for length in range(1, max_length + 1):
        for i in range(0, n - length + 1):
            j = i + length
            if (i, j) in certified:
                continue
            if not any(k in uncovered_set for k in range(i, j)):
                continue
            if length == 1 or reach[i]:
                spans.append((0 if reach[i] else 1, length, i, j))
    spans.sort()
    return [(i, j) for _rank, _length, i, j in spans]


def certify_complete_partition(instance, customer_ids: Sequence[str], search: Callable) -> dict:
    """Find one complete certified contiguous partition.

    ``search(sequence, max_seconds, max_expansions)`` must return a certificate.
    Singletons and 2–5 customer spans that remain unresolved are retried on
    longer budgets. Search failure is never an infeasibility claim. If the
    sequence still has no complete cover, the result contains no pieces.
    """
    customers = list(customer_ids)
    n = len(customers)
    certified: dict = {}
    attempts: list[dict] = []
    timeouts = 0
    exhausted = 0
    escalation_s = 0.0

    def record(i: int, j: int, result, seconds: float, expansions: int, *, escalated: bool) -> None:
        nonlocal timeouts, exhausted, escalation_s
        status = str(getattr(result, "status", ""))
        if "infeasible" in status:
            raise AssertionError(f"search returned an infeasibility claim: {status}")
        if status == STATUS_TIMEOUT:
            timeouts += 1
        elif status == STATUS_EXHAUSTED:
            exhausted += 1
        runtime = float(getattr(result, "runtime_s", 0.0) or 0.0)
        if escalated:
            escalation_s += runtime
        attempts.append(
            {
                "start": i,
                "end": j,
                "customer_ids": list(customers[i:j]),
                "status": status,
                "max_seconds": float(seconds),
                "max_expansions": int(expansions),
                "n_expansions": int(getattr(result, "n_expansions", 0) or 0),
                "runtime_s": runtime,
            }
        )

    def consider(i: int, j: int, seconds: float, expansions: int, *, escalated: bool):
        if (i, j) in certified:
            return certified[(i, j)]
        result = search(customers[i:j], seconds, expansions)
        record(i, j, result, seconds, expansions, escalated=escalated)
        if getattr(result, "status", "") == STATUS_CERTIFIED:
            if not getattr(result, "trace", None):
                raise AssertionError("certified subroute is missing a replayed trace")
            certified[(i, j)] = result
        return result

    def is_covered() -> bool:
        _certs, _prev, best_end = _dp_on_range(instance, customers, certified, 0, n)
        return best_end[0] < 10**9

    def already_tried(i: int, j: int, seconds: float, expansions: int) -> bool:
        return any(
            row["start"] == i
            and row["end"] == j
            and row["max_seconds"] == float(seconds)
            and row["max_expansions"] == int(expansions)
            for row in attempts
        )

    def escalate_span(i: int, j: int) -> None:
        for seconds, expansions in SHORT_BUDGETS:
            if (i, j) in certified or escalation_s >= ESCALATION_CAP_S:
                return
            if already_tried(i, j, seconds, expansions):
                continue
            consider(i, j, seconds, expansions, escalated=seconds != SHORT_BUDGETS[0][0])

    if n > 0:
        seconds, expansions = FULL_BUDGETS[0]
        consider(0, n, seconds, expansions, escalated=False)

    if (0, n) not in certified:
        for i in range(n):
            seconds, expansions = SHORT_BUDGETS[0]
            consider(i, i + 1, seconds, expansions, escalated=False)
        for i in range(n):
            if (i, i + 1) not in certified:
                escalate_span(i, i + 1)

    if not is_covered() and n > 1:
        limit = min(5, n)
        for seconds, expansions in SHORT_BUDGETS:
            if is_covered() or escalation_s >= ESCALATION_CAP_S:
                break
            reach = _reachability(certified, n)
            uncovered = _uncovered_indices(certified, reach, n)
            for i, j in _priority_spans(n, uncovered, certified, reach, limit):
                if is_covered() or escalation_s >= ESCALATION_CAP_S:
                    break
                if (i, j) in certified or already_tried(i, j, seconds, expansions):
                    continue
                consider(
                    i,
                    j,
                    seconds,
                    expansions,
                    escalated=seconds != SHORT_BUDGETS[0][0],
                )

    if not is_covered():
        for seconds, expansions in FULL_BUDGETS[1:]:
            if is_covered() or escalation_s >= ESCALATION_CAP_S:
                break
            consider(0, n, seconds, expansions, escalated=True)

    if is_covered() and (0, n) not in certified and n > 1:
        # Merge adjacent certified pieces so the admitted routes stay long.
        # Only the base budget is used; a failed merge is not an infeasibility.
        segments = [(i, i + 1) for i in range(n) if (i, i + 1) in certified]
        progress = True
        while progress and escalation_s < ESCALATION_CAP_S:
            progress = False
            merged = []
            index = 0
            while index < len(segments):
                if index + 1 < len(segments) and segments[index][1] == segments[index + 1][0]:
                    left, right = segments[index][0], segments[index + 1][1]
                    seconds, expansions = FULL_BUDGETS[0] if right - left > 5 else SHORT_BUDGETS[0]
                    if (left, right) not in certified and not already_tried(left, right, seconds, expansions):
                        consider(left, right, seconds, expansions, escalated=False)
                    if (left, right) in certified:
                        merged.append((left, right))
                        index += 2
                        progress = True
                        continue
                merged.append(segments[index])
                index += 1
            segments = merged

    if is_covered():
        if (0, n) in certified and n > 0:
            # Prefer the untouched source sequence when it itself is certified.
            pieces = [certified[(0, n)]]
        else:
            _certs, prev, _best = _dp_on_range(instance, customers, certified, 0, n)
            pieces = _walk_partition(certified, prev, n, 0)
        covered = True
    else:
        pieces = []
        covered = False
    return {
        "covered": covered,
        "certificates": pieces,
        "attempts": attempts,
        "timeouts": timeouts,
        "exhausted": exhausted,
        "n_subroutes": len(pieces),
        "customer_ids": customers,
        "escalation_runtime_s": escalation_s,
    }


def repair_pyvrp_routes(
    routes: Sequence[FrozenRoute],
) -> tuple[list[tuple[FrozenRoute, dict]], dict]:
    """Admit only complete certified partitions of V1 development routes.

    A source route that still has an unresolved customer after the adaptive
    search is quarantined entirely. Customers are not dropped one by one.
    """
    files = instance_index()
    cache: dict[tuple, object] = {}

    def search(instance_id: str, sequence: Sequence[str], seconds: float, expansions: int):
        key = (instance_id, tuple(sequence), float(seconds), int(expansions))
        if key in cache:
            return cache[key]
        instance = parse_instance(files[instance_id])
        result = solve_charging_certificate(
            instance,
            sequence,
            max_expansions=int(expansions),
            max_seconds=float(seconds),
        )
        cache[key] = result
        return result

    retained = []
    split_factors = Counter()
    n_seen = 0
    n_unchanged = 0
    n_split = 0
    quarantined = []
    timeouts = 0
    exhausted = 0
    replay_failures = 0
    for route in routes:
        if route.base_instance in CONSUMED_TEST_PARENT_SET:
            raise AssertionError(route.base_instance)
        n_seen += 1
        if n_seen == 1 or n_seen % 10 == 0:
            print(
                f"pyvrp certify {n_seen}/{len(routes)} retained={len(retained)} quarantined={len(quarantined)}",
                flush=True,
            )
        instance = parse_instance(files[route.raw_instance_id])

        def _search(sequence, seconds, expansions, _instance_id=route.raw_instance_id):
            return search(_instance_id, sequence, seconds, expansions)

        try:
            plan = certify_complete_partition(instance, route.customer_ids, _search)
        except AssertionError as exc:
            if "replay failed" in str(exc):
                replay_failures += 1
                quarantined.append(
                    {
                        "source_route_id": route.route_id,
                        "customer_ids": list(route.customer_ids),
                        "search_statuses": ["certificate_replay_failed"],
                        "budgets_attempted": [],
                        "reason": "unresolved_certificate_search",
                        "detail": str(exc),
                    }
                )
                continue
            raise
        timeouts += int(plan["timeouts"])
        exhausted += int(plan["exhausted"])
        if not plan["covered"]:
            quarantined.append(
                {
                    "source_route_id": route.route_id,
                    "customer_ids": list(route.customer_ids),
                    "search_statuses": sorted({row["status"] for row in plan["attempts"]}),
                    "budgets_attempted": [
                        {
                            "start": row["start"],
                            "end": row["end"],
                            "customer_ids": row["customer_ids"],
                            "status": row["status"],
                            "max_seconds": row["max_seconds"],
                            "max_expansions": row["max_expansions"],
                        }
                        for row in plan["attempts"]
                    ],
                    "reason": "unresolved_certificate_search",
                }
            )
            continue
        pieces = plan["certificates"]
        unchanged = len(pieces) == 1 and tuple(getattr(pieces[0], "customer_ids", ())) == tuple(route.customer_ids)
        flat = []
        for cert in pieces:
            flat.extend(_customers_from_trace(route.customer_ids, cert, unchanged and len(pieces) == 1))
        if tuple(flat) != tuple(route.customer_ids):
            raise AssertionError(f"partition reordered or dropped customers on {route.route_id}")
        if unchanged:
            n_unchanged += 1
            split_factors[1] += 1
        else:
            n_split += 1
            split_factors[len(pieces)] += 1
        for offset, cert in enumerate(pieces):
            customers = _customers_from_trace(route.customer_ids, cert, unchanged)
            built = make_frozen_route(
                instance,
                customers,
                route_id=f"v2pyvrp_{route.route_id}_{offset:02d}",
                vehicle_index=route.vehicle_index,
                generator="certified_pyvrp_unchanged" if unchanged else "certified_pyvrp_partition",
                charging_feasibility_status="certified_feasible",
                seed=route.seed,
            )
            if not cert.sha256 or not cert.trace:
                raise AssertionError(f"admitted route {built.route_id} is missing a certificate")
            extra = {
                "v2_source": "pyvrp_customer_order",
                "split": split_for_parent(built.base_instance),
                "certificate_type": "charging_certificate_search",
                "certificate_sha256": cert.sha256,
                "certificate_completion_time": cert.completion_time,
                "certificate_station_visits": cert.n_station_visits,
                "source_parent_route_id": route.route_id,
                "source_xlsx_sha256": "",
                "certificate_trace": cert.trace,
                "retained_unchanged": unchanged,
            }
            retained.append((built, extra))
    total_customers = sum(len(route.customer_ids) for route in routes)
    quarantined_customers = sum(len(row["customer_ids"]) for row in quarantined)
    admitted_customers = sum(len(route.customer_ids) for route, _extra in retained)
    dropped = total_customers - admitted_customers - quarantined_customers
    if dropped != 0:
        raise AssertionError(f"customer accounting mismatch: silently dropped {dropped}")
    summary = {
        "original_pyvrp_routes": len(routes),
        "final_certified_routes": len(retained),
        "retained_unchanged": n_unchanged,
        "split_routes": n_split,
        "split_factor_counts": {str(k): int(v) for k, v in sorted(split_factors.items())},
        "search_timeouts": timeouts,
        "search_exhausted": exhausted,
        "certificate_replay_failures": replay_failures,
        "customer_coverage": {
            "total_source_customers": total_customers,
            "customers_in_admitted_partitions": admitted_customers,
            "quarantined_source_routes": len(quarantined),
            "quarantined_customers": quarantined_customers,
            "coverage_fraction": (admitted_customers / total_customers) if total_customers else 0.0,
            "silently_dropped_customers": dropped,
        },
        "quarantined": quarantined,
        "n_train": sum(1 for _r, extra in retained if extra["split"] == "train"),
        "n_validation": sum(1 for _r, extra in retained if extra["split"] == "validation"),
        "test_parents_touched": False,
        "admission_rule": "complete_certified_partition_only",
    }
    return retained, summary


def _customers_from_trace(original: Sequence[str], cert, unchanged: bool) -> tuple[str, ...]:
    if unchanged:
        return tuple(original)
    customers = tuple(getattr(cert, "customer_ids", ()) or ())
    if not customers:
        raise AssertionError("certificate is missing customer_ids")
    return customers
