"""One-shot V4 TEST evaluation: V4 FA-HPPO vs V3 FA-HPPO + three baselines."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v4_test"))

from baselines.greedy import GreedyFullCharge, GreedyMinimumSufficientCharge  # noqa: E402
from baselines.lookahead import OneStepLookahead  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from experiments.evaluate import evaluate_policy  # noqa: E402
from experiments.provenance import code_dirty_paths  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from rl.checkpoint import load_hybrid_actor  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402

from common import TEST_CORPUS, V4, dump_json, git_head, lf_sha256, load_json, sha256  # noqa: E402

CONSUMED = V4 / "EVALUATION_CONSUMED.json"
RAW = V4 / "raw"


class TimeAwareShield:
    def __init__(self, policy):
        self.policy = policy
        self.name = policy.name

    def __call__(self, simulator, eval_mode: bool = True):
        simulator.time_aware_envelope = True
        return self.policy(simulator, eval_mode=eval_mode)


def _gate() -> dict:
    for name in ("PAPER_PROTOCOL.json", "CHECKPOINT_FREEZE.json", "TEST_LOCK.json"):
        if not (V4 / name).is_file():
            raise SystemExit(f"{name} missing; TEST stays closed")
    if CONSUMED.is_file():
        raise SystemExit("V4 evaluation already consumed; must not rerun")
    dirty = code_dirty_paths()
    # Allow only locked V4 TEST data paths.
    bad = []
    for path in dirty:
        if path.startswith("data/routes_v2/synthcharge_v4_test") or path.startswith(
            "data/splits_v2/synthcharge_v4_test"
        ):
            continue
        if path.startswith("results/v4_test/"):
            continue
        bad.append(path)
    if bad:
        raise SystemExit(f"source tree dirty before TEST eval: {bad[:12]}")
    freeze = load_json(V4 / "CHECKPOINT_FREEZE.json")
    if freeze.get("n_v4_checkpoints") != 5 or freeze.get("n_v3_checkpoints") != 5:
        raise SystemExit("checkpoint freeze incomplete")
    for item in freeze["checkpoints"]:
        if sha256(ROOT / item["checkpoint"]) != item["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint hash mismatch: {item['checkpoint']}")
    lock = load_json(V4 / "TEST_LOCK.json")
    mismatched = [rel for rel, digest in lock["files"].items() if lf_sha256(ROOT / rel) != digest]
    if mismatched:
        raise SystemExit(f"TEST_LOCK mismatch: {mismatched[:5]}")
    print("V4 GATE: protocol+freeze+lock OK; TEST closed until --execute-once")
    return freeze


def evaluate() -> list[dict]:
    freeze = _gate()
    routes = read_jsonl(TEST_CORPUS)
    raw_rows = {
        json.loads(line)["route_id"]: json.loads(line)
        for line in TEST_CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(routes) != 180:
        raise SystemExit(f"expected 180 routes, got {len(routes)}")

    policies = []
    for item in freeze["checkpoints"]:
        actor = load_hybrid_actor(ROOT / item["checkpoint"])
        policies.append(
            {
                "method": item["method"],
                "seed": item["seed"],
                "policy": actor,
                "checkpoint_sha256": item["checkpoint_sha256"],
                "training_git_sha": item.get("training_git_sha"),
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
            }
        )

    lock_sha = lf_sha256(V4 / "TEST_LOCK.json")
    eval_sha = git_head()
    records = []
    for route in routes:
        assert route.physics_profile == "synthcharge_linear", route.route_id
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance, name="synthcharge_linear")
        assert profile.energy_law == "linear_distance", route.route_id
        row = raw_rows[route.route_id]
        context = {
            "benchmark": "synthcharge_v4_reward_paper_test",
            "benchmark_role": "fresh_external_confirmatory_TEST_for_V4_reward",
            "base_instance": route.base_instance,
            "raw_instance_id": route.raw_instance_id,
            "n_customers": route.n_customers,
            "n_customers_instance": row.get("n_customers_instance"),
            "layout": row.get("layout"),
            "length_bin": row.get("length_bin"),
            "charge_class": row.get("charge_class"),
            "physics_profile": "synthcharge_linear",
            "certificate_station_visits": row.get("certificate_station_visits"),
            "test_lock_sha256": lock_sha,
            "evaluation_git_sha": eval_sha,
            "generator_seed": row.get("generator_seed"),
        }
        for item in policies:
            result = evaluate_policy(
                instance=instance,
                route=route,
                policy=item["policy"],
                eval_mode=True,
                profile_name="synthcharge_linear",
            )
            records.append(
                result.to_record(
                    method=item["method"],
                    seed=item["seed"],
                    checkpoint_sha256=item["checkpoint_sha256"],
                    training_git_sha=item["training_git_sha"],
                    **context,
                )
            )
    return records


def main() -> None:
    if "--execute-once" not in sys.argv:
        raise SystemExit("refusing to open V4 TEST without --execute-once")
    started = time.time()
    rows = evaluate()
    expected = 180 * (5 + 5 + 3)  # V4 seeds + V3 seeds + 3 baselines
    if len(rows) != expected:
        raise SystemExit(f"row count {len(rows)} != expected {expected}")
    methods = {r["method"] for r in rows}
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "synthcharge_v4_test.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows) + "\n",
        encoding="utf-8",
    )
    dump_json(
        CONSUMED,
        {
            "consumed": True,
            "benchmark": "synthcharge_v4_reward_paper_test",
            "n_routes": 180,
            "n_rows": len(rows),
            "methods": sorted(methods),
            "raw_file": path.relative_to(ROOT).as_posix(),
            "raw_sha256_lf": lf_sha256(path),
            "evaluation_git_sha": git_head(),
            "runtime_s": time.time() - started,
            "no_retrain_after_test": True,
            "no_checkpoint_reselection": True,
            "no_reward_redesign_after_test": True,
            "independent_of_v3_test": True,
        },
    )
    print(json.dumps({"n_rows": len(rows), "methods": sorted(methods)}, indent=2))


if __name__ == "__main__":
    main()
