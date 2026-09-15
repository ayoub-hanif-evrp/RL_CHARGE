"""Corpus builder attempts every instance and never filters on charging."""

from data.parser import parse_instance
from data.paths import iter_instance_files
from routing.corpus import generate_corpus
from routing.fixed_route import CHARGING_FEASIBILITY_UNVERIFIED, FrozenRoute
from routing.config import PyVRPConfig


class StubGenerator:
    def generate(self, instance, profile):
        customers = tuple(node.string_id for node in instance.customers)
        return [
            FrozenRoute(
                route_id=f"{instance.metadata.instance_id}__v0",
                source_dataset="EVRPTW-GR",
                doi="10.17632/srfdbp2twv.1",
                raw_instance_id=instance.metadata.instance_id,
                relative_path=instance.metadata.relative_path,
                network_group=instance.metadata.network_group.value,
                terrain_variant=instance.metadata.terrain_variant.value,
                customer_distribution=instance.metadata.customer_distribution,
                schedule_type=instance.metadata.schedule_type,
                generator="stub",
                generator_version="0",
                seed=0,
                stop="MaxIterations",
                n_iterations=1,
                config_hash="stub",
                physics_profile=profile.name,
                capacity_policy="official_3650kg",
                distance_scale=1000,
                demand_scale=10,
                rounding_policy="numpy_round_to_int64",
                routing_problem="vrptw_pickup_customers_only",
                python="test",
                os_name="test",
                arch="test",
                instance_sha256="0" * 64,
                vehicle_index=0,
                depot_id=instance.depot.string_id,
                customer_ids=customers,
                route_demand=sum(node.demand for node in instance.customers),
                route_distance=0.0,
                route_duration_lower_bound=0.0,
                n_customers=len(customers),
                routing_feasible=True,
                charging_feasibility_status=CHARGING_FEASIBILITY_UNVERIFIED,
                generation_runtime_s=0.0,
                routing_objective=0.0,
            )
        ]


def test_corpus_attempts_all_instances_without_charging_filter(raw_root, tmp_path):
    files = list(iter_instance_files(raw_root))
    assert len(files) == 124
    result = generate_corpus(
        dataset_root=raw_root,
        out_dir=tmp_path,
        generator=StubGenerator(),
        config=PyVRPConfig.from_toml(),
        instance_files=files,
        annotate_tw=False,
    )
    assert result.attempted == 124
    assert result.failures == []
    assert len(result.routes) == 124
    assert all(route.charging_feasibility_status == "unverified" for route in result.routes)
    assert (tmp_path / "failures.json").is_file()
    assert (tmp_path / "manifest.csv").is_file()


class ExplodingGenerator:
    def generate(self, instance, profile):
        raise RuntimeError("solver boom")


def test_failures_are_recorded_not_dropped(write_instance, tmp_path):
    path, root = write_instance()
    result = generate_corpus(
        out_dir=tmp_path,
        generator=ExplodingGenerator(),
        config=PyVRPConfig.from_toml(),
        instance_files=[path],
        parse=lambda p: parse_instance(p, root=root),
        annotate_tw=False,
    )
    assert result.attempted == 1
    assert len(result.failures) == 1
    assert "solver boom" in result.failures[0].error
    assert result.routes == []


def test_terrain_reuse_copies_canonical_sequences(tmp_path):
    from conftest import build_instance_text, node_row

    nodes_l = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("C1", "c", 1.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.1),
        node_row("C2", "c", 2.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.2),
    ]
    nodes_nl = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("C1", "c", 1.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.5),
        node_row("C2", "c", 2.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.8),
    ]
    root = tmp_path / "EVRPTW_GR"
    l_dir = root / "Small_Network" / "5_Customers" / "Level"
    nl_dir = root / "Small_Network" / "5_Customers" / "Nearly_Level"
    l_dir.mkdir(parents=True)
    nl_dir.mkdir(parents=True)
    (l_dir / "c101C5_L.txt").write_text(build_instance_text(nodes=nodes_l), encoding="utf-8")
    (nl_dir / "c101C5_NL.txt").write_text(build_instance_text(nodes=nodes_nl), encoding="utf-8")

    class CountingGenerator(StubGenerator):
        calls = 0

        def generate(self, instance, profile):
            type(self).calls += 1
            return super().generate(instance, profile)

    CountingGenerator.calls = 0
    files = sorted(root.rglob("*.txt"))
    result = generate_corpus(
        dataset_root=root,
        out_dir=tmp_path / "routes",
        generator=CountingGenerator(),
        config=PyVRPConfig.from_toml(),
        instance_files=files,
        parse=lambda p: parse_instance(p, root=root),
        annotate_tw=False,
    )
    assert CountingGenerator.calls == 1
    assert result.attempted == 2
    assert len(result.routes) == 2
    sequences = {route.raw_instance_id: route.customer_ids for route in result.routes}
    assert sequences["c101C5_L"] == sequences["c101C5_NL"]
    reused = [route for route in result.routes if route.terrain_reuse]
    canonical = [route for route in result.routes if not route.terrain_reuse]
    assert len(canonical) == 1
    assert canonical[0].raw_instance_id == "c101C5_L"
    assert reused[0].route_source_instance_id == "c101C5_L"


def test_terrain_matrix_mismatch_fails_group(tmp_path):
    from conftest import build_instance_text, node_row

    nodes_l = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("C1", "c", 1.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.0),
        node_row("C2", "c", 2.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.0),
    ]
    nodes_nl = [
        node_row("D0", "d", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("S0", "f", 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0),
        node_row("C1", "c", 9.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.0),
        node_row("C2", "c", 2.0, 0.0, 10.0, 0.0, 1000.0, 1.0, 0.0),
    ]
    root = tmp_path / "EVRPTW_GR"
    l_dir = root / "Small_Network" / "5_Customers" / "Level"
    nl_dir = root / "Small_Network" / "5_Customers" / "Nearly_Level"
    l_dir.mkdir(parents=True)
    nl_dir.mkdir(parents=True)
    (l_dir / "c101C5_L.txt").write_text(build_instance_text(nodes=nodes_l), encoding="utf-8")
    (nl_dir / "c101C5_NL.txt").write_text(build_instance_text(nodes=nodes_nl), encoding="utf-8")
    files = sorted(root.rglob("*.txt"))
    result = generate_corpus(
        dataset_root=root,
        out_dir=tmp_path / "routes",
        generator=StubGenerator(),
        config=PyVRPConfig.from_toml(),
        instance_files=files,
        parse=lambda p: parse_instance(p, root=root),
        annotate_tw=False,
    )
    assert result.routes == []
    assert len(result.failures) == 2
