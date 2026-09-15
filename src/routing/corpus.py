"""Method-independent frozen-route corpus. Never filters on charging behaviour."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from data.models import EVRPTWGRInstance
from data.parser import parse_instance
from data.paths import ROUTES_DIR, iter_instance_files
from physics.parameters import DEFAULT_PROFILE_NAME, PhysicsProfile

from .config import PyVRPConfig
from .fixed_route import FrozenRoute
from .pyvrp_generator import PyVRPGenerator
from .serialize import canonical_dumps, write_jsonl


@dataclass
class GenerationFailure:
    relative_path: str
    error: str


@dataclass
class CorpusResult:
    routes: List[FrozenRoute]
    failures: List[GenerationFailure]
    attempted: int


def generate_corpus(
    *,
    dataset_root=None,
    out_dir: Optional[Path] = None,
    profile_name: str = DEFAULT_PROFILE_NAME,
    config: Optional[PyVRPConfig] = None,
    generator: Optional[PyVRPGenerator] = None,
    instance_files: Optional[Sequence[Path]] = None,
    parse: Callable[[Path], EVRPTWGRInstance] = parse_instance,
) -> CorpusResult:
    """Generate frozen routes for every instance file. Failures are recorded, not dropped."""
    config = config or PyVRPConfig.from_toml()
    generator = generator or PyVRPGenerator(config)
    out_dir = Path(out_dir) if out_dir is not None else ROUTES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(instance_files) if instance_files is not None else list(iter_instance_files(dataset_root))
    routes: List[FrozenRoute] = []
    failures: List[GenerationFailure] = []
    for path in files:
        try:
            instance = parse(path)
        except Exception as exc:
            failures.append(
                GenerationFailure(
                    relative_path=Path(path).as_posix(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        try:
            profile = PhysicsProfile.from_instance(instance, name=profile_name)
            produced = generator.generate(instance, profile)
            routes.extend(produced)
            write_jsonl(
                out_dir / "by_instance" / f"{instance.metadata.instance_id}.jsonl",
                produced,
            )
        except Exception as exc:
            failures.append(
                GenerationFailure(
                    relative_path=instance.metadata.relative_path,
                    error=f"{type(exc).__name__}: {exc}",
                )
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
    return CorpusResult(routes=routes, failures=failures, attempted=len(files))


def _write_manifest(path: Path, routes: Iterable[FrozenRoute]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "route_id",
        "raw_instance_id",
        "relative_path",
        "network_group",
        "terrain_variant",
        "vehicle_index",
        "n_customers",
        "route_demand",
        "route_distance",
        "routing_feasible",
        "charging_feasibility_status",
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
                    "relative_path": route.relative_path,
                    "network_group": route.network_group,
                    "terrain_variant": route.terrain_variant,
                    "vehicle_index": route.vehicle_index,
                    "n_customers": route.n_customers,
                    "route_demand": route.route_demand,
                    "route_distance": route.route_distance,
                    "routing_feasible": route.routing_feasible,
                    "charging_feasibility_status": route.charging_feasibility_status,
                    "generator": route.generator,
                    "generator_version": route.generator_version,
                    "seed": route.seed,
                    "n_iterations": route.n_iterations,
                    "config_hash": route.config_hash,
                    "physics_profile": route.physics_profile,
                    "unassigned_customer_ids": json.dumps(list(route.unassigned_customer_ids)),
                }
            )
