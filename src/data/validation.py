"""Validation checks for EVRPTW-GR instances.

Two severities are reported, and nothing is ever repaired:

``ERROR``
    The instance cannot be trusted -- a broken file, a contradictory field, or
    a missing required value.

``WARNING``
    The instance parses cleanly but contradicts the dataset documentation or
    its own vehicle parameters. Two such contradictions are known to exist in
    the published data and are surfaced rather than silently corrected; see
    ``data/README.md``.

Only constraints the raw dataset actually supports are checked here. In
particular no bound is placed on altitude, because the published values run
from -0.928571 to 1.3 and the documentation states no range.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .errors import EVRPTWGRError
from .models import NODE_ID_PREFIX, NodeType, TerrainVariant
from .parser import parse_instance
from .paths import iter_instance_files, resolve_dataset_root

#: Terrain variants the documentation promises for every network subset.
EXPECTED_TERRAIN_VARIANTS = (
    TerrainVariant.LEVEL,
    TerrainVariant.NEARLY_LEVEL,
    TerrainVariant.VERY_GENTLE,
)

#: The documented relationship between the Very Gentle and Nearly Level
#: variants: "Altitude values are 1.5 times higher than those in the _NL file."
VG_TO_NL_RATIO = 1.5
VG_RATIO_TOLERANCE = 1e-6


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    instance_id: Optional[str] = None
    relative_path: Optional[str] = None

    def as_dict(self):
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "instance_id": self.instance_id,
            "relative_path": self.relative_path,
        }


@dataclass
class ScanResult:
    """Outcome of scanning a dataset tree."""

    instances: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    #: Files present on disk that could not be parsed at all.
    unparsed: list = field(default_factory=list)

    @property
    def errors(self):
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def issues_for(self, code):
        return [i for i in self.issues if i.code == code]


def _issue(severity, code, message, instance):
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        instance_id=instance.metadata.instance_id,
        relative_path=instance.metadata.relative_path,
    )


def validate_instance(instance):
    """Return every issue found in a single parsed instance."""
    issues = []
    meta = instance.metadata

    def error(code, message):
        issues.append(_issue(Severity.ERROR, code, message, instance))

    def warn(code, message):
        issues.append(_issue(Severity.WARNING, code, message, instance))

    duplicates = [sid for sid, n in Counter(n.string_id for n in instance.nodes).items() if n > 1]
    if duplicates:
        error("duplicate_node_id", f"duplicate StringID values: {sorted(duplicates)}")

    depot_count = len(instance.depots)
    if depot_count != 1:
        error("depot_count", f"expected exactly 1 depot, found {depot_count}")

    if not instance.stations:
        error("no_stations", "instance has no charging stations")

    if not instance.customers:
        error("no_customers", "instance has no customers")

    for node in instance.nodes:
        where = f"node {node.string_id!r} (line {node.line_number})"

        expected_prefix = NODE_ID_PREFIX[node.node_type]
        if not node.string_id.startswith(expected_prefix):
            error(
                "id_type_mismatch",
                f"{where}: type {node.node_type.value!r} expects a {expected_prefix!r} prefix",
            )

        if node.demand < 0:
            error("negative_demand", f"{where}: demand is negative ({node.demand})")

        if node.service_time < 0:
            error("negative_service_time", f"{where}: service time is negative ({node.service_time})")

        if node.ready_time > node.due_date:
            error(
                "inverted_time_window",
                f"{where}: ReadyTime {node.ready_time} exceeds DueDate {node.due_date}",
            )

        if node.node_type is not NodeType.CUSTOMER and node.demand != 0:
            warn(
                "non_customer_demand",
                f"{where}: {node.node_type.name.lower()} carries a non-zero demand ({node.demand})",
            )

    # The documentation states Level files have every altitude set to 0.0.
    if meta.terrain_variant is TerrainVariant.LEVEL:
        non_zero = [n.string_id for n in instance.nodes if n.altitude != 0.0]
        if non_zero:
            warn(
                "level_altitude_non_zero",
                f"Level variant has {len(non_zero)} non-zero altitudes: {sorted(non_zero)[:5]}",
            )

    parsed_customers = len(instance.customers)
    if parsed_customers != meta.declared_customer_count:
        error(
            "customer_count_mismatch",
            f"filename declares {meta.declared_customer_count} customers "
            f"but the file contains {parsed_customers}",
        )

    if meta.declared_station_count is not None:
        parsed_stations = len(instance.stations)
        if parsed_stations != meta.declared_station_count:
            error(
                "station_count_mismatch",
                f"filename declares {meta.declared_station_count} stations "
                f"but the file contains {parsed_stations}",
            )

    # Known published inconsistency: Small_Network kept the original Schneider
    # load capacities while carrying the rescaled demands.
    capacity = instance.vehicle.load_capacity
    demands = [n.demand for n in instance.customers]
    if demands and max(demands) > capacity:
        warn(
            "demand_exceeds_capacity",
            f"largest single demand {max(demands)} exceeds vehicle load capacity {capacity}",
        )

    if not instance.vehicle.has_curb_weight:
        warn(
            "missing_curb_weight",
            "footer has no 'M' entry, so vehicle curb weight is unavailable "
            "(expected for Small_Network)",
        )

    return issues


def validate_terrain_consistency(instances):
    """Cross-file checks that need more than one variant of an instance."""
    issues = []
    by_base = defaultdict(dict)
    for instance in instances:
        meta = instance.metadata
        key = (meta.network_group, meta.customer_folder, meta.base_instance)
        by_base[key][meta.terrain_variant] = instance

    for variants in by_base.values():
        nearly = variants.get(TerrainVariant.NEARLY_LEVEL)
        gentle = variants.get(TerrainVariant.VERY_GENTLE)
        if nearly is None or gentle is None:
            continue

        nl_altitudes = {n.string_id: n.altitude for n in nearly.nodes}
        mismatched = []
        for node in gentle.nodes:
            expected = nl_altitudes.get(node.string_id)
            if expected is None:
                continue
            if abs(expected * VG_TO_NL_RATIO - node.altitude) > VG_RATIO_TOLERANCE:
                mismatched.append(node.string_id)

        if mismatched:
            issues.append(
                _issue(
                    Severity.WARNING,
                    "vg_ratio_mismatch",
                    f"{len(mismatched)} node altitudes are not {VG_TO_NL_RATIO}x the "
                    f"Nearly Level values from {nearly.metadata.instance_id!r}: "
                    f"{sorted(mismatched)[:5]}",
                    gentle,
                )
            )

    return issues


def find_missing_variants(instances):
    """Report instances whose documented terrain variants are absent on disk.

    The documentation promises Level, Nearly Level, and Very Gentle for every
    subset, so a missing folder shows up here as a coverage gap.
    """
    present = defaultdict(set)
    folders = {}
    for instance in instances:
        meta = instance.metadata
        key = (meta.network_group, meta.customer_folder, meta.base_instance)
        present[key].add(meta.terrain_variant)
        folders[key] = meta

    missing = []
    for key, variants in sorted(present.items(), key=lambda kv: str(kv[0])):
        meta = folders[key]
        for expected in EXPECTED_TERRAIN_VARIANTS:
            if expected not in variants:
                missing.append(
                    {
                        "network_group": meta.network_group.value,
                        "customer_folder": meta.customer_folder,
                        "base_instance": meta.base_instance,
                        "missing_variant": expected.name,
                    }
                )
    return missing


def scan_dataset(root=None):
    """Parse and validate every instance under ``root``.

    Never raises on a bad instance: a parse failure is recorded as an ERROR
    issue so a single damaged file cannot hide the state of the rest.
    """
    resolved = resolve_dataset_root(root)
    result = ScanResult()

    for path in iter_instance_files(resolved):
        relative = path.relative_to(resolved).as_posix()
        try:
            instance = parse_instance(path, root=resolved)
        except EVRPTWGRError as exc:
            result.unparsed.append(relative)
            result.issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="parse_failure",
                    message=str(exc),
                    instance_id=path.stem,
                    relative_path=relative,
                )
            )
            continue

        result.instances.append(instance)
        result.issues.extend(validate_instance(instance))

    result.issues.extend(validate_terrain_consistency(result.instances))
    return result
