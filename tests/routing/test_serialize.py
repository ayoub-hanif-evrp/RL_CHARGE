"""Frozen-route serialization is byte-stable and does not store charging visits."""

import pytest

from routing.fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute
from routing.serialize import dumps_route, loads_route, read_jsonl, write_jsonl


def _route(**overrides):
    payload = dict(
        route_id="c101C5_L__v0",
        source_dataset="EVRPTW-GR",
        doi="10.17632/srfdbp2twv.1",
        raw_instance_id="c101C5_L",
        relative_path="Small_Network/5_Customers/Level/c101C5_L.txt",
        network_group="Small_Network",
        terrain_variant="L",
        customer_distribution="c",
        schedule_type=1,
        generator="pyvrp",
        generator_version="0.14.0",
        seed=42,
        stop="MaxIterations",
        n_iterations=50,
        config_hash="abc",
        physics_profile="official_evrptwgr",
        capacity_policy="official_3650kg",
        distance_scale=1000,
        demand_scale=10,
        rounding_policy="numpy_round_to_int64",
        routing_problem="vrptw_pickup_customers_only",
        python="3.11.0",
        os_name="Windows",
        arch="AMD64",
        instance_sha256="0" * 64,
        vehicle_index=0,
        depot_id="D0",
        customer_ids=("C30", "C12"),
        route_demand=547.5,
        route_distance=10.0,
        route_duration_lower_bound=10.0,
        n_customers=2,
        routing_feasible=True,
        charging_feasibility_status=CHARGING_FEASIBILITY_UNVERIFIED,
        generation_runtime_s=0.01,
        routing_objective=10.0,
    )
    payload.update(overrides)
    return FrozenRoute(**payload)


def test_round_trip_is_byte_stable(tmp_path):
    route = _route()
    encoded = dumps_route(route)
    assert dumps_route(loads_route(encoded)) == encoded
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [route])
    loaded = read_jsonl(path)
    assert loaded[0].customer_ids == ("C30", "C12")
    assert loaded[0].charging_feasibility_status == "unverified"


def test_customer_sequence_is_immutable():
    route = _route()
    assert route.customer_ids == ("C30", "C12")
    with pytest.raises(AttributeError):
        route.customer_ids = ("C12", "C30")  # type: ignore[misc]
