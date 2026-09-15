"""External customer-route generation and frozen-route serialization."""

from .config import PyVRPConfig
from .corpus import CorpusResult, GenerationFailure, generate_corpus
from .fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute
from .pyvrp_generator import PyVRPGenerator, build_customer_matrices
from .serialize import dumps_route, loads_route, read_jsonl, write_jsonl

__all__ = [
    "CHARGING_FEASIBILITY_UNVERIFIED",
    "CorpusResult",
    "FrozenRoute",
    "GenerationFailure",
    "PyVRPConfig",
    "PyVRPGenerator",
    "build_customer_matrices",
    "dumps_route",
    "generate_corpus",
    "loads_route",
    "read_jsonl",
    "write_jsonl",
]
