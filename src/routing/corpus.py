"""Method-independent frozen-route corpus. Never filters on charging behaviour."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data.models import EVRPTWGRInstance
from data.parser import parse_instance
from data.paths import ROUTES_DIR, iter_instance_files
from physics.parameters import DEFAULT_PROFILE_NAME, PhysicsProfile

from .config import PyVRPConfig
from .fixed_route import FrozenRoute
from .pyvrp_generator import (
    PyVRPGenerator,
    build_customer_matrices,
    retarget_routes,
)
from .serialize import canonical_dumps, write_jsonl


TERRAIN_PRIORITY = {"L": 0, "NL": 1, "VG": 2}


class TerrainMatrixMismatchError(ValueError):
    """Sibling terrains disagree on the customer-only routing matrices."""


@dataclass
class GenerationFailure:
    relative_path: str
    error: str


@dataclass
class CorpusResult:
    routes: List[FrozenRoute]
    failures: List[GenerationFailure]
    attempted: int


def sibling_key(instance: EVRPTWGRInstance) -> Tuple[str, str, str]:
    meta = instance.metadata
    return (meta.network_group.value, meta.customer_folder, meta.base_instance)


def pick_canonical(instances: Sequence[EVRPTWGRInstance]) -> EVRPTWGRInstance:
    return min(
        instances,
        key=lambda inst: TERRAIN_PRIORITY.get(inst.metadata.terrain_variant.value, 99),
    )


def assert_identical_customer_matrices(
    instances: Sequence[EVRPTWGRInstance],
    profile_name: str,
    config: PyVRPConfig,
) -> None:
    if len(instances) <= 1:
        return
    matrices = []
    for instance in instances:
        profile = PhysicsProfile.from_instance(instance, name=profile_name)
        matrices.append(build_customer_matrices(instance, profile, config))
    base = matrices[0]
    for instance, other in zip(instances[1:], matrices[1:]):
        same = (
            other.location_ids == base.location_ids
            and np.array_equal(other.distance, base.distance)
            and np.array_equal(other.duration, base.duration)
            and other.pickup == base.pickup
            and other.tw_early == base.tw_early
            and other.tw_late == base.tw_late
            and other.service == base.service
            and other.capacity == base.capacity
        )
        if not same:
            raise TerrainMatrixMismatchError(
                f"customer matrices differ for sibling {instance.metadata.instance_id} "
                f"vs {instances[0].metadata.instance_id}"
            )


def generate_corpus(
    *,
    dataset_root=None,
    out_dir: Optional[Path] = None,
    profile_name: str = DEFAULT_PROFILE_NAME,
    config: Optional[PyVRPConfig] = None,
    generator: Optional[PyVRPGenerator] = None,
    instance_files: Optional[Sequence[Path]] = None,
    parse: Callable[[Path], EVRPTWGRInstance] = parse_instance,
    annotate_tw: bool = True,
    write_metadata: bool = False,
) -> CorpusResult:
    """Generate frozen routes once per terrain sibling group. Failures are recorded, not dropped."""
    config = config or PyVRPConfig.from_toml()
    generator = generator or PyVRPGenerator(config)
    out_dir = Path(out_dir) if out_dir is not None else ROUTES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(instance_files) if instance_files is not None else list(iter_instance_files(dataset_root))
    routes: List[FrozenRoute] = []
    failures: List[GenerationFailure] = []
    parsed: List[EVRPTWGRInstance] = []
    for path in files:
        try:
            parsed.append(parse(path))
        except Exception as exc:
            failures.append(
                GenerationFailure(
                    relative_path=Path(path).as_posix(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    groups: Dict[Tuple[str, str, str], List[EVRPTWGRInstance]] = defaultdict(list)
    for instance in parsed:
        groups[sibling_key(instance)].append(instance)

    for _key, siblings in groups.items():
        siblings = list(siblings)
        try:
            assert_identical_customer_matrices(siblings, profile_name, config)
            canonical = pick_canonical(siblings)
            profile = PhysicsProfile.from_instance(canonical, name=profile_name)
            produced = generator.generate(canonical, profile)
        except Exception as exc:
            for instance in siblings:
                failures.append(
                    GenerationFailure(
                        relative_path=instance.metadata.relative_path,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            continue
        source_id = canonical.metadata.instance_id
        for instance in siblings:
            reuse = instance.metadata.instance_id != source_id
            produced_for = retarget_routes(
                produced,
                instance,
                source_instance_id=source_id,
                terrain_reuse=reuse,
            )
            routes.extend(produced_for)
            write_jsonl(
                out_dir / "by_instance" / f"{instance.metadata.instance_id}.jsonl",
                produced_for,
            )

    if annotate_tw:
        from routing.audit import annotate_routing_tw_feasibility

        routes = annotate_routing_tw_feasibility(
            routes,
            profile_name=profile_name,
            parse=parse,
            dataset_root=dataset_root,
        )

    write_jsonl(out_dir / "corpus.jsonl", routes)
    _write_manifest(out_dir / "manifest.csv", routes)
    (out_dir / "failures.json").write_text(
        canonical_dumps(
            {
                "attempted": len(files),
                "n_routes": len(routes),
                "n_failures": len(failures),
                "failures": [failure.__dict__ for failure in failures],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if write_metadata:
        from routing.audit import audit_corpus

        write_corpus_metadata(out_dir, routes, config=config, profile_name=profile_name)
        audit_corpus(out_dir, profile_name=profile_name)
    return CorpusResult(routes=routes, failures=failures, attempted=len(files))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_corpus_metadata(
    out_dir: Path,
    routes: Sequence[FrozenRoute],
    *,
    config: PyVRPConfig,
    profile_name: str,
) -> Path:
    import importlib.metadata

    try:
        pyvrp_version = importlib.metadata.version("pyvrp")
    except importlib.metadata.PackageNotFoundError:
        pyvrp_version = "unknown"
    first = routes[0] if routes else None
    payload = {
        "dataset": "EVRPTW-GR",
        "doi": "10.17632/srfdbp2twv.1",
        "n_routes": len(routes),
        "n_instances": len({route.raw_instance_id for route in routes}),
        "generator": config.generator,
        "pyvrp_version": pyvrp_version,
        "seed": config.seed,
        "stop": config.stop,
        "iterations": dict(config.iterations),
        "fleet_policy": config.fleet_policy,
        "fixed_vehicle_cost_formula": "F = 2 * n_customers * Dmax + 1",
        "load_convention": "official_reference_pickup",
        "physics_profile": profile_name,
        "demand_mapping": config.demand_mapping,
        "config_hash": config.fingerprint(),
        "generation_command": (
            "python scripts/generate_routes.py --profile official_evrptwgr "
            "--seed 42 --out data/routes"
        ),
        "freeze_note": (
            "Part 3A freeze. Later experiments must not rewrite corpus.jsonl, "
            "manifest.csv, or corpus_metadata.json."
        ),
        "sha256": {
            "corpus.jsonl": sha256_file(out_dir / "corpus.jsonl"),
            "manifest.csv": sha256_file(out_dir / "manifest.csv"),
        },
        "example_fixed_vehicle_cost": None if first is None else first.fixed_vehicle_cost,
    }
    path = out_dir / "corpus_metadata.json"
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    return path


def _write_manifest(path: Path, routes: Iterable[FrozenRoute]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "route_id",
        "raw_instance_id",
        "base_instance",
        "relative_path",
        "network_group",
        "terrain_variant",
        "vehicle_index",
        "n_customers",
        "n_vehicles",
        "route_demand",
        "route_distance",
        "total_distance",
        "fixed_vehicle_cost",
        "lexicographic_objective",
        "routing_feasible",
        "routing_tw_feasible",
        "charging_feasibility_status",
        "route_source_instance_id",
        "terrain_reuse",
        "fleet_policy",
        "generator",
        "generator_version",
        "seed",
        "n_iterations",
        "config_hash",
        "physics_profile",
        "unassigned_customer_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for route in routes:
            writer.writerow(
                {
                    "route_id": route.route_id,
                    "raw_instance_id": route.raw_instance_id,
                    "base_instance": route.base_instance,
                    "relative_path": route.relative_path,
                    "network_group": route.network_group,
                    "terrain_variant": route.terrain_variant,
                    "vehicle_index": route.vehicle_index,
                    "n_customers": route.n_customers,
                    "n_vehicles": route.n_vehicles,
                    "route_demand": route.route_demand,
                    "route_distance": route.route_distance,
                    "total_distance": route.total_distance,
                    "fixed_vehicle_cost": route.fixed_vehicle_cost,
                    "lexicographic_objective": route.lexicographic_objective,
                    "routing_feasible": route.routing_feasible,
                    "routing_tw_feasible": route.routing_tw_feasible,
                    "charging_feasibility_status": route.charging_feasibility_status,
                    "route_source_instance_id": route.route_source_instance_id,
                    "terrain_reuse": route.terrain_reuse,
                    "fleet_policy": route.fleet_policy,
                    "generator": route.generator,
                    "generator_version": route.generator_version,
                    "seed": route.seed,
                    "n_iterations": route.n_iterations,
                    "config_hash": route.config_hash,
                    "physics_profile": route.physics_profile,
                    "unassigned_customer_ids": json.dumps(list(route.unassigned_customer_ids)),
                }
            )
