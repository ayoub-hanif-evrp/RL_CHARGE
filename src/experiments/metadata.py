"""Experiment metadata. No mixed-unit totals, no invented prices."""

from __future__ import annotations

from typing import Any, Optional

from domain.load_convention import LoadConvention, require_load_convention
from routing.serialize import canonical_dumps


def experiment_metadata(
    *,
    method: str,
    split: str,
    load_convention,
    physics_profile: str,
    seed: int,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "method": method,
        "split": split,
        "load_convention": require_load_convention(load_convention).value,
        "physics_profile": physics_profile,
        "seed": int(seed),
        "primary_objective": "CompletionTimeObjective",
        "charging_feasibility_status": "unverified",
    }
    if extra:
        payload.update(extra)
    return payload


def dumps_metadata(payload: dict[str, Any]) -> str:
    return canonical_dumps(payload)
