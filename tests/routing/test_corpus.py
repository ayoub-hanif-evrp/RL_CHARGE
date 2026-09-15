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
    )
    assert result.attempted == 1
    assert len(result.failures) == 1
    assert "solver boom" in result.failures[0].error
    assert result.routes == []
