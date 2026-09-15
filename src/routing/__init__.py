"""External customer-route generation and frozen-route serialization."""

from .audit import AuditReport, audit_corpus
from .config import PyVRPConfig
from .corpus import (
    CorpusResult,
    GenerationFailure,
    TerrainMatrixMismatchError,
    generate_corpus,
)
from .fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute
from .pyvrp_generator import (
    PyVRPGenerator,
    build_customer_matrices,
    dominating_fixed_cost,
    retarget_routes,
)
from .serialize import dumps_route, loads_route, read_jsonl, write_jsonl

__all__ = [
    "AuditReport",
    "CHARGING_FEASIBILITY_UNVERIFIED",
    "CorpusResult",
    "FrozenRoute",
    "GenerationFailure",
    "PyVRPConfig",
    "PyVRPGenerator",
    "TerrainMatrixMismatchError",
    "audit_corpus",
    "build_customer_matrices",
    "dominating_fixed_cost",
    "dumps_route",
    "generate_corpus",
    "loads_route",
    "read_jsonl",
    "retarget_routes",
    "write_jsonl",
]
