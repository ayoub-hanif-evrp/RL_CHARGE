"""One-shot final evaluation after CHECKPOINT_FREEZE.json.

A. SynthCharge fresh TEST — SynthCharge-trained HybridPPO/DiscretePPO + baselines.
B. Legacy EVRPTW-GR challenge — gold-trained HybridPPO/DiscretePPO + baselines.

Uses route.physics_profile for every evaluation. Does not select models.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v2"))

from baselines.greedy import GreedyFullCharge, GreedyMinimumSufficientCharge  # noqa: E402
from baselines.lookahead import OneStepLookahead  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from experiments.evaluate import evaluate_policy  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from rl.checkpoint import load_hybrid_actor  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402

from final_common import (  # noqa: E402
    BASELINES,
    FINAL,
    METHOD_FREEZE_SHA,
    dump_json,
    git_head,
    git_porcelain,
    lf_sha256,
    load_json,
    method_tree_diff,
    sha256,
)

CONSUMED = FINAL / "EVALUATION_CONSUMED.json"
RAW = FINAL / "raw"
MANIFESTS = FINAL / "manifests"


class TimeAwareShield:
    def __init__(self, policy):
        self.policy = policy
        self.name = policy.name

    def __call__(self, simulator, eval_mode: bool = True):
        simulator.time_aware_envelope = True
        return self.policy(simulator, eval_mode=eval_mode)


def _gate() -> dict:
    if not (FINAL / "CHECKPOINT_FREEZE.json").is_file():
        raise SystemExit("CHECKPOINT_FREEZE.json missing; TEST stays closed")
    if not (FINAL / "FINAL_PROTOCOL.json").is_file():
        raise SystemExit("FINAL_PROTOCOL.json missing")
    if CONSUMED.is_file():
        raise SystemExit("final evaluation already consumed; must not rerun")
    if method_tree_diff().strip():
        raise SystemExit("method tree differs from freeze")
    porcelain = git_porcelain().strip()
    # Allow only results/v2/final and scripts/v2 finalization files if present.
    for line in porcelain.splitlines():
        path = line[3:].strip().replace("\\", "/")
        if path.startswith("src/") or path.startswith("configs/") or path.startswith("tests/") or path.startswith("third_party/"):
            raise SystemExit(f"frozen path dirty: {path}")
        if path.startswith("data/"):
            raise SystemExit(f"data path dirty: {path}")
    freeze = load_json(FINAL / "CHECKPOINT_FREEZE.json")
    if freeze.get("n_checkpoints") != 20:
        raise SystemExit("checkpoint freeze incomplete")
    for item in freeze["checkpoints"]:
        path = ROOT / item["checkpoint"]
        if sha256(path) != item["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint hash mismatch: {item['checkpoint']}")
    lock = load_json(FINAL / "TEST_LOCK.json")
    bad = [rel for rel, digest in lock["files"].items() if lf_sha256(ROOT / rel) != digest]
    if bad:
        raise SystemExit(f"TEST_LOCK mismatch: {bad[:5]}")
    print("FINAL TRAINING COMPLETE: YES")
    print("20/20 learned runs complete: YES")
    print("all best.pt hashes frozen: YES")
    print("working tree acceptable: YES")
    print("method tree identical to freeze: YES")
    print("data audit: PASS")
    print("TEST_LOCK: PASS")
    print("final protocol frozen: YES")
    print("checkpoint freeze present: YES")
    print("TEST has not been used for model selection: YES")
    return freeze


def _charge_class(instance, route, row: dict) -> str:
    if row.get("charge_class"):
        return row["charge_class"]
    from domain.load_convention import LoadConvention
    from simulation.simulator import FixedRouteSimulator, run_continue_only

    profile = PhysicsProfile.from_instance(instance, name=route.physics_profile or "official_evrptwgr")
    simulator = FixedRouteSimulator(
        instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    continued = run_continue_only(simulator)
    if continued.feasible and simulator.state.completed:
        return "no_charge_required"
    if int(row.get("certificate_station_visits") or 0) >= 1:
        return "charging_required"
    return "unclassified"


def _assert_physics(route, expected_profile: str, expected_law: str) -> None:
    if getattr(route, "source_dataset", "") == "SynthCharge" or expected_profile == "synthcharge_linear":
        assert route.physics_profile == "synthcharge_linear", route.route_id
    profile_name = route.physics_profile or "official_evrptwgr"
    assert profile_name == expected_profile, (route.route_id, profile_name, expected_profile)
    # Parse once to check energy law through the profile loader.
    instance = parse_route_instance(route)
    profile = PhysicsProfile.from_instance(instance, name=profile_name)
    assert profile.energy_law == expected_law, (route.route_id, profile.energy_law, expected_law)
    assert profile.name == expected_profile


def evaluate_set(
    *,
    benchmark: str,
    role: str,
    corpus: Path,
    learned: list[dict],
    expected_profile: str,
    expected_law: str,
    training_dataset: str,
) -> list[dict]:
    routes = read_jsonl(corpus)
    raw_rows = {
        json.loads(line)["route_id"]: json.loads(line)
        for line in corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for route in routes:
        _assert_physics(route, expected_profile, expected_law)

    policies = []
    for item in learned:
        actor = load_hybrid_actor(ROOT / item["checkpoint"])
        policies.append(
            {
                "method": item["method"],
                "seed": item["seed"],
                "policy": actor,
                "checkpoint_sha256": item["checkpoint_sha256"],
                "training_git_sha": item["training_git_sha"],
                "training_dataset": item["dataset"],
            }
        )
    for baseline in (GreedyMinimumSufficientCharge(), GreedyFullCharge(), OneStepLookahead()):
        policies.append(
            {
                "method": baseline.name,
                "seed": None,
                "policy": TimeAwareShield(baseline),
                "checkpoint_sha256": None,
                "training_git_sha": None,
                "training_dataset": None,
            }
        )

    lock_sha = lf_sha256(FINAL / "TEST_LOCK.json")
    eval_sha = git_head()
    records = []
    for route in routes:
        instance = parse_route_instance(route)
        row = raw_rows[route.route_id]
        charge_class = _charge_class(instance, route, row)
        context = {
            "benchmark": benchmark,
            "benchmark_role": role,
            "base_instance": route.base_instance,
            "raw_instance_id": route.raw_instance_id,
            "n_customers": route.n_customers,
            "customer_distribution": route.customer_distribution,
            "terrain_variant": route.terrain_variant,
            "schedule_type": route.schedule_type,
            "network_group": route.network_group,
            "layout": row.get("layout"),
            "length_bin": row.get("length_bin"),
            "charge_class": charge_class,
            "source_dataset": route.source_dataset,
            "physics_profile": route.physics_profile or expected_profile,
            "certificate_station_visits": row.get("certificate_station_visits"),
            "method_freeze_sha": METHOD_FREEZE_SHA,
            "test_lock_sha256": lock_sha,
            "evaluation_git_sha": eval_sha,
        }
        profile_name = route.physics_profile or expected_profile
        for item in policies:
            result = evaluate_policy(
                instance=instance,
                route=route,
                policy=item["policy"],
                eval_mode=True,
                profile_name=profile_name,
            )
            record = result.to_record(
                method=item["method"],
                seed=item["seed"],
                checkpoint_sha256=item["checkpoint_sha256"],
                training_git_sha=item["training_git_sha"],
                training_dataset=item["training_dataset"] or training_dataset,
                **context,
            )
            dead_end = (result.extra or {}).get("dead_end")
            if dead_end is not None:
                record["dead_end"] = dead_end
            records.append(record)
    return records


def _write_rows(name: str, rows: list[dict]) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows) + "\n",
        encoding="utf-8",
    )
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "n_rows": len(rows),
        "sha256_lf": lf_sha256(path),
    }


def main() -> None:
    if "--execute-once" not in sys.argv:
        raise SystemExit("refusing to open final evaluation without --execute-once")
    freeze = _gate()
    started = time.time()
    synth_learned = [item for item in freeze["checkpoints"] if item["dataset"] == "synthcharge"]
    gold_learned = [item for item in freeze["checkpoints"] if item["dataset"] == "gold"]
    if len(synth_learned) != 10 or len(gold_learned) != 10:
        raise SystemExit("expected 10 SynthCharge and 10 gold learned checkpoints")

    synth_rows = evaluate_set(
        benchmark="synthcharge_fresh_test",
        role="fresh_external_confirmatory_TEST",
        corpus=ROOT / "data/routes_v2/synthcharge_final/test/corpus.jsonl",
        learned=synth_learned,
        expected_profile="synthcharge_linear",
        expected_law="linear_distance",
        training_dataset="synthcharge",
    )
    legacy_rows = evaluate_set(
        benchmark="legacy_evrptwgr_challenge",
        role="legacy_same_domain_challenge_NOT_fresh_TEST",
        corpus=ROOT / "data/routes_v2/legacy_evrptwgr_challenge/corpus.jsonl",
        learned=gold_learned,
        expected_profile="official_evrptwgr",
        expected_law="demir_model2",
        training_dataset="gold",
    )
    if len(synth_rows) != 1170:
        raise SystemExit(f"SynthCharge row count {len(synth_rows)} != 1170")
    if len(legacy_rows) != 338:
        raise SystemExit(f"legacy row count {len(legacy_rows)} != 338")

    outputs = {
        "synthcharge_fresh_test": _write_rows("synthcharge_test", synth_rows),
        "legacy_evrptwgr_challenge": _write_rows("legacy_evrptwgr_challenge", legacy_rows),
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    dump_json(
        MANIFESTS / "synthcharge_test_manifest.json",
        {
            "benchmark": "synthcharge_fresh_test",
            "role": "fresh_external_confirmatory_TEST",
            "n_routes": 90,
            "n_rows": 1170,
            "physics_profile": "synthcharge_linear",
            "output": outputs["synthcharge_fresh_test"],
        },
    )
    dump_json(
        MANIFESTS / "legacy_challenge_manifest.json",
        {
            "benchmark": "legacy_evrptwgr_challenge",
            "role": "legacy_same_domain_challenge_NOT_fresh_TEST",
            "n_routes": 26,
            "n_rows": 338,
            "physics_profile": "official_evrptwgr",
            "output": outputs["legacy_evrptwgr_challenge"],
        },
    )
    dump_json(
        CONSUMED,
        {
            "consumed": True,
            "runtime_s": time.time() - started,
            "method_freeze_sha": METHOD_FREEZE_SHA,
            "checkpoint_freeze_sha256_lf": lf_sha256(FINAL / "CHECKPOINT_FREEZE.json"),
            "evaluation_git_sha": git_head(),
            "outputs": outputs,
            "note": (
                "SynthCharge TEST and the legacy challenge were opened once. "
                "No rerun for model selection."
            ),
        },
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
