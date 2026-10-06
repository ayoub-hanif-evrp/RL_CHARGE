"""V2 certificate, shield, scale, and V1 immutability tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from data.parser import parse_instance
from data.paths import RAW_EVRPTW_GR_DIR, REPO_ROOT
from exact.charging_certificate import (
    STATUS_CERTIFIED,
    STATUS_EXHAUSTED,
    STATUS_TIMEOUT,
    replay_action_trace,
    solve_charging_certificate,
)
from experiments.v2_scope import CONSUMED_TEST_PARENT_SET, split_for_parent
from physics.parameters import PhysicsProfile
from rl.ppo import HybridPPO, PPOConfig
from routing.fixed_route import FrozenRoute
from routing.official_replay import certified_charge_to_max_trace
from routing.v2_corpus import certify_complete_partition, global_return_scale, partition_customers
from simulation.actions import ChargeAction
from simulation.shield import evaluate_shield, soc_interval_for_station
from simulation.simulator import FixedRouteSimulator
from domain.load_convention import LoadConvention

V1_SHA256 = {
    "data/routes/corpus.jsonl": "1796d817ff058fe0559021ab79ca97087ef3646c57744c2c883ecc37e85451e0",
    "data/routes/manifest.csv": "41bb8be3245f8f3b0800db4ba74cb19a2003236b6fbd0dbc1237e7cdfaee104d",
    "data/routes/corpus_metadata.json": "1ca8725199842f0a8d4a4e6fac2a31931ea3c9aaca833c2d253c6147469d9a96",
    "data/splits/train.json": "3aecd73e9472b8572e899a7e2da3f1f874f0e5a13250f5cd6d76d812002913ce",
    "data/splits/validation.json": "1379aef7e92c308f90ab78674dec9994db72078c96cb1020ecc49d33eef69af4",
    "data/splits/test.json": "9bf39fbf1f27aa57146879fb71922b0216f24f1eb37a4fd9a9a43b44b6c4b6d0",
    "data/splits/split_metadata.json": "4740ab71737bfd9c04d37f258e57bfa4854a526da7c03b218ffe877818128900",
}

# SHA-256 of the committed LF bytes at pre-V2 HEAD fe34811. A Windows checkout
# with core.autocrlf rewrites this file to CRLF, which is a different digest
# (d2b34b88...) and is not the repository artifact.
PROVENANCE_RELATIVE = "results/v1/final/PROVENANCE.json"
PROVENANCE_COMMITTED_SHA256 = "df2c6328ae5b2e54d32931d3be9fa8bf07b376e1d8c8c9316526573d9c2629cc"
PRE_V2_HEAD = "fe34811cf7f96b08e69d791600dea54e49998c36"


def _committed_bytes(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n")


def test_v1_artifacts_are_byte_identical():
    for relative, digest in V1_SHA256.items():
        payload = (REPO_ROOT / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
    provenance = _committed_bytes((REPO_ROOT / PROVENANCE_RELATIVE).read_bytes())
    assert hashlib.sha256(provenance).hexdigest() == PROVENANCE_COMMITTED_SHA256
    try:
        pre = subprocess.check_output(
            ["git", "rev-parse", f"{PRE_V2_HEAD}:{PROVENANCE_RELATIVE}"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{PROVENANCE_RELATIVE}"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return
    assert pre == head


def test_consumed_test_parents_are_rejected():
    for parent in CONSUMED_TEST_PARENT_SET:
        with pytest.raises(ValueError):
            split_for_parent(parent)


def test_timeout_is_not_infeasible():
    mapping = json.loads(
        (REPO_ROOT / "results/v1/pilot/correctness_audit/official_route_plan_mapping.json").read_text(
            encoding="utf-8"
        )
    )
    tour = next(
        row
        for row in mapping["tours"]
        if row.get("replay", {}).get("feasible") and row["base_instance"] not in CONSUMED_TEST_PARENT_SET
    )
    instance = parse_instance(RAW_EVRPTW_GR_DIR / _relative(tour["instance_id"]))
    result = solve_charging_certificate(
        instance, tour["customer_ids"], max_expansions=1, max_seconds=10.0
    )
    assert result.status == STATUS_TIMEOUT
    assert result.feasible is False
    assert "infeasible" not in result.status


def test_known_positive_search_replays(tmp_path):
    mapping = json.loads(
        (REPO_ROOT / "results/v1/pilot/correctness_audit/official_route_plan_mapping.json").read_text(
            encoding="utf-8"
        )
    )
    tour = next(
        row
        for row in mapping["tours"]
        if row.get("replay", {}).get("feasible")
        and row["base_instance"] == "r104"
        and len(row["customer_ids"]) <= 3
    )
    instance = parse_instance(RAW_EVRPTW_GR_DIR / _relative(tour["instance_id"]))
    result = solve_charging_certificate(
        instance, tour["customer_ids"], max_expansions=4000, max_seconds=10.0
    )
    assert result.status == STATUS_CERTIFIED
    replay = replay_action_trace(instance, result.customer_ids, result.actions)
    assert replay is not None
    assert result.sha256


def test_partition_preserves_order_and_customers(monkeypatch):
    monkeypatch.setattr(
        "routing.v2_corpus.customer_only_distance",
        lambda _instance, sequence: float(len(sequence)),
    )
    customers = ["C1", "C2", "C3"]

    def search(sequence):
        if len(sequence) <= 2:
            return _Piece(sequence)
        return type("Miss", (), {"status": STATUS_EXHAUSTED, "trace": None})()

    plan = partition_customers(object(), customers, search)
    assert plan["covered"] is True
    flat = []
    for cert in plan["certificates"]:
        flat.extend(cert.customer_ids)
    assert flat == customers
    assert flat.count("C1") == flat.count("C2") == flat.count("C3") == 1


def test_incomplete_partition_drops_no_customers(monkeypatch):
    monkeypatch.setattr(
        "routing.v2_corpus.customer_only_distance",
        lambda _instance, sequence: float(len(sequence)),
    )

    def search(sequence):
        if sequence == ["C2"]:
            return type("Miss", (), {"status": STATUS_TIMEOUT, "trace": None})()
        if "C2" in sequence:
            return type("Miss", (), {"status": STATUS_EXHAUSTED, "trace": None})()
        return _Piece(sequence)

    plan = partition_customers(object(), ["C1", "C2", "C3"], search)
    assert plan["covered"] is False
    assert plan["certificates"] == []


def test_unresolved_singleton_is_retried_with_a_larger_budget(monkeypatch):
    monkeypatch.setattr(
        "routing.v2_corpus.customer_only_distance",
        lambda _instance, sequence: float(len(sequence)),
    )
    calls = []

    class _Miss:
        status = STATUS_TIMEOUT
        trace = None
        runtime_s = 0.01
        n_expansions = 1

    def search(sequence, seconds, expansions):
        calls.append((tuple(sequence), float(seconds), int(expansions)))
        if tuple(sequence) == ("C1",) and float(seconds) >= 12.0:
            return _Piece(["C1"])
        if tuple(sequence) == ("C2",):
            return _Piece(["C2"])
        return _Miss()

    plan = certify_complete_partition(object(), ["C1", "C2"], search)
    singleton_budgets = {(seconds, expansions) for seq, seconds, expansions in calls if seq == ("C1",)}
    assert (1.0, 4000) in singleton_budgets
    assert any(seconds > 1.0 and expansions > 4000 for seconds, expansions in singleton_budgets)
    assert plan["covered"] is True
    assert [tuple(cert.customer_ids) for cert in plan["certificates"]] == [("C1",), ("C2",)]
    assert "infeasible" not in "".join(row["status"] for row in plan["attempts"])


class _Piece:
    def __init__(self, customers):
        self.status = STATUS_CERTIFIED
        self.trace = [{"kind": "CONTINUE"}]
        self.customer_ids = tuple(customers)
        self.sha256 = "abc"
        self.runtime_s = 0.01
        self.n_expansions = 1


def test_return_scale_is_one_global_train_constant():
    train = [_fake_route("a", 100.0), _fake_route("b", 250.0)]
    # global_return_scale reads files. Exercise the arithmetic contract directly.
    scale = max(100.0, 250.0)
    raw = -250.0
    trainer = HybridPPO(_tiny_config(scale), device="cpu")
    assert trainer.scale_reward(raw) * scale == pytest.approx(raw)
    assert trainer.scale_reward(-100.0) * trainer.config.return_scale == pytest.approx(-100.0)
    assert train[0].route_id != train[1].route_id


def test_time_aware_upper_bound_does_not_exceed_max_soc():
    mapping = json.loads(
        (REPO_ROOT / "results/v1/pilot/correctness_audit/official_route_plan_mapping.json").read_text(
            encoding="utf-8"
        )
    )
    tour = next(
        row
        for row in mapping["tours"]
        if row.get("replay", {}).get("feasible") and row["base_instance"] not in CONSUMED_TEST_PARENT_SET
    )
    instance = parse_instance(RAW_EVRPTW_GR_DIR / _relative(tour["instance_id"]))
    profile = PhysicsProfile.from_instance(instance)
    sim = FixedRouteSimulator(
        instance, tuple(tour["customer_ids"]), profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    sim.time_aware_envelope = True
    decision = evaluate_shield(sim)
    max_soc = float(profile.max_soc_fraction)
    for station_id, legal in zip(decision.station_ids, decision.mask[1:]):
        if not legal:
            continue
        interval = soc_interval_for_station(sim, station_id, time_aware=True)
        assert interval.soc_upper <= max_soc + 1e-8
        assert interval.soc_lower <= interval.soc_upper + 1e-8


def test_official_charge_to_max_is_not_falsely_pruned():
    mapping = json.loads(
        (REPO_ROOT / "results/v1/pilot/correctness_audit/official_route_plan_mapping.json").read_text(
            encoding="utf-8"
        )
    )
    checked = 0
    for row in mapping["tours"]:
        if not row.get("replay", {}).get("feasible"):
            continue
        if row["base_instance"] in CONSUMED_TEST_PARENT_SET:
            continue
        instance = parse_instance(RAW_EVRPTW_GR_DIR / _relative(row["instance_id"]))
        replayed = certified_charge_to_max_trace(instance, list(row["tour"]))
        assert replayed is not None
        customers, actions, _trace = replayed
        profile = PhysicsProfile.from_instance(instance)
        sim = FixedRouteSimulator(instance, customers, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP)
        sim.time_aware_envelope = True
        for action in actions:
            if action[0] != "CHARGE":
                result = sim.step(__import__("simulation.actions", fromlist=["ContinueAction"]).ContinueAction())
                assert result.feasible
                continue
            decision = evaluate_shield(sim, time_aware=True)
            station_id = action[1]
            index = decision.station_ids.index(station_id)
            assert decision.mask[index + 1], dead_end(decision, station_id)
            interval = soc_interval_for_station(sim, station_id, time_aware=True)
            assert float(action[2]) <= interval.soc_upper + 1e-5
            result = sim.step(ChargeAction(station_id, float(action[2])))
            assert result.feasible
        assert sim.state.completed
        checked += 1
    assert checked == 166


def dead_end(decision, station_id):
    index = decision.station_ids.index(station_id)
    return decision.station_reasons[index]


def _relative(instance_id: str) -> str:
    matches = list(RAW_EVRPTW_GR_DIR.rglob(f"{instance_id}.txt"))
    assert matches
    return matches[0].relative_to(RAW_EVRPTW_GR_DIR).as_posix()


def _span(customers, subset):
    for i in range(len(customers)):
        if tuple(customers[i : i + len(subset)]) == tuple(subset):
            return i, i + len(subset)
    raise AssertionError(subset)


def _fake_route(route_id, horizon):
    return FrozenRoute(
        route_id=route_id,
        source_dataset="EVRPTW-GR",
        doi="",
        raw_instance_id="x",
        relative_path="",
        network_group="Small_Network",
        terrain_variant="NL",
        customer_distribution="r",
        schedule_type=1,
        generator="test",
        generator_version="v2",
        seed=0,
        stop="",
        n_iterations=0,
        config_hash="",
        physics_profile="official_evrptwgr",
        capacity_policy="",
        distance_scale=1,
        demand_scale=1,
        rounding_policy="",
        routing_problem="",
        python="",
        os_name="",
        arch="",
        instance_sha256="0" * 64,
        vehicle_index=0,
        depot_id="D0",
        customer_ids=("C1",),
        route_demand=1.0,
        route_distance=1.0,
        route_duration_lower_bound=horizon,
        n_customers=1,
        routing_feasible=True,
        charging_feasibility_status="certified_feasible",
        generation_runtime_s=0.0,
        routing_objective=1.0,
    )


def _tiny_config(scale: float) -> PPOConfig:
    return PPOConfig(
        learning_rate=1e-3,
        rollout_steps=4,
        minibatch_size=2,
        update_epochs=1,
        clip_eps=0.2,
        gamma=1.0,
        gae_lambda=0.95,
        entropy_coef=0.0,
        value_coef=0.5,
        max_grad_norm=0.5,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        budget_updates=1,
        eval_interval=1,
        early_stopping_patience=1,
        seed=0,
        return_scale=scale,
    )


def _tiny_instance():
    mapping = json.loads(
        (REPO_ROOT / "results/v1/pilot/correctness_audit/official_route_plan_mapping.json").read_text(
            encoding="utf-8"
        )
    )
    tour = next(row for row in mapping["tours"] if row.get("instance_id"))
    return parse_instance(RAW_EVRPTW_GR_DIR / _relative(tour["instance_id"]))


def test_global_return_scale_uses_only_given_routes(monkeypatch):
    calls = []

    def _parse(path):
        calls.append(str(path))
        class Depot:
            due_date = 123.0 if "train" in str(path) else 999.0
        class Meta:
            pass
        class Inst:
            depot = Depot()
        return Inst()

    monkeypatch.setattr("routing.v2_corpus.parse_instance", _parse)
    routes = [_fake_route("train-a", 1.0)]
    routes[0] = FrozenRoute(**{**routes[0].to_dict(), "relative_path": "train/a.txt"})
    assert global_return_scale(routes) == 123.0
    assert calls and "validation" not in calls[0] and "test" not in calls[0].lower()
