"""Audit a frozen-route corpus. Never drops routes; records coverage and TW replay."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from data.models import EVRPTWGRInstance
from data.parser import parse_instance
from data.paths import RAW_EVRPTW_GR_DIR, ROUTES_DIR
from domain.load_convention import LoadConvention
from physics.parameters import DEFAULT_PROFILE_NAME, PhysicsProfile
from routing.fixed_route import FrozenRoute
from routing.serialize import read_jsonl
from simulation.simulator import FixedRouteSimulator, run_continue_only


@dataclass
class AuditIssue:
    kind: str
    message: str
    route_id: Optional[str] = None
    instance_id: Optional[str] = None


@dataclass
class AuditReport:
    n_routes: int
    n_instances: int
    n_issues: int
    issues: List[AuditIssue]
    ok: bool


def annotate_routing_tw_feasibility(
    routes: Sequence[FrozenRoute],
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
    parse: Callable[[Path], EVRPTWGRInstance] = parse_instance,
    dataset_root=None,
) -> List[FrozenRoute]:
    root = Path(dataset_root) if dataset_root is not None else RAW_EVRPTW_GR_DIR
    annotated: List[FrozenRoute] = []
    cache: Dict[str, EVRPTWGRInstance] = {}
    for route in routes:
        instance = cache.get(route.raw_instance_id)
        if instance is None:
            path = root / route.relative_path
            instance = parse(path)
            cache[route.raw_instance_id] = instance
        feasible = _tw_replay(instance, route, profile_name)
        annotated.append(
            FrozenRoute.from_dict({**route.to_dict(), "routing_tw_feasible": feasible})
        )
    return annotated


def _tw_replay(instance: EVRPTWGRInstance, route: FrozenRoute, profile_name: str) -> bool:
    if not route.customer_ids:
        return False
    profile = PhysicsProfile.from_instance(instance, name=profile_name)
    simulator = FixedRouteSimulator(
        instance,
        route.customer_ids,
        profile,
        LoadConvention.OFFICIAL_REFERENCE_PICKUP,
        ignore_energy=True,
    )
    result = run_continue_only(simulator)
    return bool(result.feasible and simulator.state.completed)


def audit_corpus(
    routes_dir: Optional[Path] = None,
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
    parse: Callable[[Path], EVRPTWGRInstance] = parse_instance,
    dataset_root=None,
) -> AuditReport:
    routes_dir = Path(routes_dir) if routes_dir is not None else ROUTES_DIR
    routes = read_jsonl(routes_dir / "corpus.jsonl")
    issues: List[AuditIssue] = []
    by_instance: Dict[str, List[FrozenRoute]] = defaultdict(list)
    for route in routes:
        by_instance[route.raw_instance_id].append(route)

    root = Path(dataset_root) if dataset_root is not None else RAW_EVRPTW_GR_DIR
    instances: Dict[str, EVRPTWGRInstance] = {}
    for route in routes:
        if route.raw_instance_id in instances:
            continue
        instances[route.raw_instance_id] = parse(root / route.relative_path)

    for instance_id, instance_routes in by_instance.items():
        instance = instances[instance_id]
        profile = PhysicsProfile.from_instance(instance, name=profile_name)
        assigned: List[str] = []
        extras = []
        legal = {node.string_id for node in instance.customers}
        for route in instance_routes:
            for cid in route.customer_ids:
                if cid not in legal:
                    extras.append(cid)
                assigned.append(cid)
            if route.route_demand > profile.payload_capacity_kg + 1e-9:
                issues.append(
                    AuditIssue(
                        "capacity",
                        f"{route.route_id}: route_demand {route.route_demand} exceeds "
                        f"{profile.payload_capacity_kg}",
                        route_id=route.route_id,
                        instance_id=instance_id,
                    )
                )
        if extras:
            issues.append(
                AuditIssue(
                    "extra_customers",
                    f"{instance_id}: extra customer ids {extras}",
                    instance_id=instance_id,
                )
            )
        required = [node.string_id for node in instance.customers]
        counted = {}
        for cid in assigned:
            counted[cid] = counted.get(cid, 0) + 1
        duplicates = [cid for cid, n in counted.items() if n > 1]
        if duplicates:
            issues.append(
                AuditIssue(
                    "duplicate_customers",
                    f"{instance_id}: customers assigned more than once {duplicates}",
                    instance_id=instance_id,
                )
            )
        missing = [cid for cid in required if cid not in counted]
        listed_unassigned = list(instance_routes[0].unassigned_customer_ids)
        if set(missing) != set(listed_unassigned):
            issues.append(
                AuditIssue(
                    "unassigned",
                    f"{instance_id}: missing {missing} listed_unassigned {listed_unassigned}",
                    instance_id=instance_id,
                )
            )

    sibling_groups: Dict[Tuple[str, str, str], List[FrozenRoute]] = defaultdict(list)
    for route in routes:
        sibling_groups[
            (route.network_group, route.customer_folder, route.base_instance)
        ].append(route)
    for key, group_routes in sibling_groups.items():
        by_terrain: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
        for route in group_routes:
            by_terrain.setdefault(route.raw_instance_id, [])
        sequences: Dict[str, Tuple[Tuple[str, ...], ...]] = defaultdict(tuple)
        grouped: Dict[str, List[FrozenRoute]] = defaultdict(list)
        for route in group_routes:
            grouped[route.raw_instance_id].append(route)
        seqs = {
            iid: tuple(route.customer_ids for route in sorted(rs, key=lambda r: r.vehicle_index))
            for iid, rs in grouped.items()
        }
        unique = set(seqs.values())
        if len(unique) > 1:
            issues.append(
                AuditIssue(
                    "terrain_sequence_mismatch",
                    f"sibling group {key} has disagreeing customer sequences",
                    instance_id=key[2],
                )
            )

    report = AuditReport(
        n_routes=len(routes),
        n_instances=len(by_instance),
        n_issues=len(issues),
        issues=issues,
        ok=len(issues) == 0,
    )
    (routes_dir / "audit.json").write_text(
        __import__("json").dumps(
            {
                "n_routes": report.n_routes,
                "n_instances": report.n_instances,
                "n_issues": report.n_issues,
                "ok": report.ok,
                "issues": [issue.__dict__ for issue in issues],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
