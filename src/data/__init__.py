"""Canonical data layer for the EVRPTW-GR dataset.

Every part of the project reads the raw dataset through this package. Parsing
logic must not be duplicated elsewhere.

This package is ingestion, validation, and inspection only. Energy, charging,
simulation, and route generation live in ``physics``, ``simulation``, and
``routing``.
"""

from .errors import (
    DatasetNotFoundError,
    EVRPTWGRError,
    MalformedInstanceError,
    MissingFieldError,
)
from .models import (
    COLUMN_NAMES,
    EVRPTWGRInstance,
    InstanceMetadata,
    NetworkGroup,
    Node,
    NodeType,
    TerrainVariant,
    VehicleParameter,
    VehicleParameters,
)
from .parser import load_dataset, parse_instance, parse_instance_text
from .paths import (
    PROCESSED_DIR,
    RAW_EVRPTW_GR_DIR,
    REPO_ROOT,
    ROUTES_DIR,
    iter_instance_files,
    resolve_dataset_root,
)
from .validation import (
    ScanResult,
    Severity,
    ValidationIssue,
    find_missing_variants,
    scan_dataset,
    validate_instance,
    validate_terrain_consistency,
)

__all__ = [
    "COLUMN_NAMES",
    "DatasetNotFoundError",
    "EVRPTWGRError",
    "EVRPTWGRInstance",
    "InstanceMetadata",
    "MalformedInstanceError",
    "MissingFieldError",
    "NetworkGroup",
    "Node",
    "NodeType",
    "PROCESSED_DIR",
    "RAW_EVRPTW_GR_DIR",
    "REPO_ROOT",
    "ROUTES_DIR",
    "ScanResult",
    "Severity",
    "TerrainVariant",
    "ValidationIssue",
    "VehicleParameter",
    "VehicleParameters",
    "find_missing_variants",
    "iter_instance_files",
    "load_dataset",
    "parse_instance",
    "parse_instance_text",
    "resolve_dataset_root",
    "scan_dataset",
    "validate_instance",
    "validate_terrain_consistency",
]
