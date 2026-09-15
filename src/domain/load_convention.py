"""Explicit payload conventions. Never switch silently."""

from enum import Enum


class LoadConvention(Enum):
    OFFICIAL_REFERENCE_PICKUP = "official_reference_pickup"
    DELIVERY = "delivery"


def require_load_convention(value) -> LoadConvention:
    if value is None:
        raise ValueError(
            "load_convention is required; pass LoadConvention.OFFICIAL_REFERENCE_PICKUP "
            "or LoadConvention.DELIVERY"
        )
    if isinstance(value, LoadConvention):
        return value
    if isinstance(value, str):
        try:
            return LoadConvention(value)
        except ValueError as exc:
            raise ValueError(f"unknown load_convention {value!r}") from exc
    raise TypeError(f"load_convention must be a LoadConvention, got {type(value)!r}")


def initial_payload_kg(convention: LoadConvention, instance, customer_ids) -> float:
    convention = require_load_convention(convention)
    if convention is LoadConvention.OFFICIAL_REFERENCE_PICKUP:
        return 0.0
    return float(sum(instance.node_by_id(cid).demand for cid in customer_ids))


def payload_after_service_kg(
    convention: LoadConvention, payload_before: float, demand: float
) -> float:
    convention = require_load_convention(convention)
    if convention is LoadConvention.OFFICIAL_REFERENCE_PICKUP:
        return payload_before + demand
    return payload_before - demand
