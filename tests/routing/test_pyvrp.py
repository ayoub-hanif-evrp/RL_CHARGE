"""PyVRP customer-only pickup-VRPTW generator."""

import numpy as np
import pytest

from conftest import node_row
from data.parser import parse_instance
from data.paths import RAW_EVRPTW_GR_DIR
from physics.parameters import PhysicsProfile
from routing.config import PyVRPConfig
from routing.pyvrp_generator import PyVRPGenerator, build_customer_matrices


def _config():
    return PyVRPConfig.from_toml()


def test_config_uses_pickup_and_max_iterations():
    config = _config()
    assert config.demand_mapping == "pickup"
    assert config.stop == "MaxIterations"
    assert config.routing_problem == "vrptw_pickup_customers_only"


def test_terrain_variants_have_identical_customer_matrices():
    config = _config()
    paths = {
        "L": RAW_EVRPTW_GR_DIR / "Small_Network/5_Customers/Level/c101C5_L.txt",
        "NL": RAW_EVRPTW_GR_DIR / "Small_Network/5_Customers/Nearly_Level/c101C5_NL.txt",
        "VG": RAW_EVRPTW_GR_DIR / "Small_Network/5_Customers/Very_Gentle/c101C5_VG.txt",
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("c101C5 terrain variants missing")
    matrices = []
    for path in paths.values():
        instance = parse_instance(path)
        profile = PhysicsProfile.from_instance(instance)
        matrices.append(build_customer_matrices(instance, profile, config))
    base = matrices[0]
    for other in matrices[1:]:
        assert other.location_ids == base.location_ids
        assert np.array_equal(other.distance, base.distance)
        assert np.array_equal(other.duration, base.duration)
        assert other.pickup == base.pickup
        assert other.tw_early == base.tw_early
        assert other.tw_late == base.tw_late
        assert other.capacity == base.capacity


def test_matrices_use_pickup_scaled_demand(write_instance):
    path, root = write_instance()
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    matrices = build_customer_matrices(instance, profile, _config())
    assert matrices.pickup == (
        int(round(instance.customers[0].demand * 10)),
        int(round(instance.customers[1].demand * 10)),
    )
    assert matrices.capacity == int(round(3650.0 * 10))


def test_pyvrp_model_maps_demand_to_pickup_not_delivery(write_instance):
    path, root = write_instance()
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    generator = PyVRPGenerator(_config(), max_iterations_override=20)
    matrices = build_customer_matrices(instance, profile, generator.config)
    model = generator._build_model(matrices)
    data = model.data()
    clients = list(data.clients())
    assert clients
    for client, pickup in zip(clients, matrices.pickup):
        assert list(client.pickup) == [pickup]
        delivery = list(client.delivery)
        assert delivery == [] or delivery == [0]


def test_pyvrp_generates_customer_only_routes(write_instance):
    nodes = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("S0", "f", 50.0, 50.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("C1", "c", 1.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.0),
        node_row("C2", "c", 2.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.0),
        node_row("C3", "c", 3.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.0),
    ]
    path, root = write_instance(nodes=nodes)
    instance = parse_instance(path, root=root)
    profile = PhysicsProfile.from_instance(instance)
    generator = PyVRPGenerator(_config(), max_iterations_override=50)
    routes = generator.generate(instance, profile)
    customer_ids = {node.string_id for node in instance.customers}
    assigned = []
    for route in routes:
        assert route.generator_version
        assert route.seed == 42
        assert route.stop == "MaxIterations"
        assert route.charging_feasibility_status == "unverified"
        assert route.demand_mapping == "pickup"
        assert "S0" not in route.customer_ids
        for cid in route.customer_ids:
            assert cid in customer_ids
            assigned.append(cid)
        assert len(route.customer_ids) == len(set(route.customer_ids))
        assert route.route_demand <= profile.payload_capacity_kg + 1e-9
    assert set(assigned).issubset(customer_ids)
