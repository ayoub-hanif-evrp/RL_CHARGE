"""One-shot V3-HPPO paper TEST evaluation.

Methods: HybridPPO (5 seeds) + three greedy/lookahead baselines + FA-HPPO-Min/Max.
No DiscretePPO. Does not select checkpoints. Requires protocol + freeze + TEST_LOCK.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from baselines.common import run_discrete_policy  # noqa: E402
from baselines.greedy import GreedyFullCharge, GreedyMinimumSufficientCharge  # noqa: E402
from baselines.lookahead import OneStepLookahead  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from experiments.evaluate import evaluate_policy  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from rl.checkpoint import load_hybrid_actor  # noqa: E402
from repo_paths import resolve_repo_path  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402

from common import (  # noqa: E402
    METHOD_FREEZE_SHA,
    TEST_CORPUS,
    V3,
    dump_json,
    git_head,
    git_porcelain,
    lf_sha256,
    load_json,
    method_tree_diff,
    sha256,
)

CONSUMED = V3 / "EVALUATION_CONSUMED.json"
RAW = V3 / "raw"


class TimeAwareShield:
    def __init__(self, policy):
        self.policy = policy
        self.name = policy.name

    def __call__(self, simulator, eval_mode: bool = True):
        simulator.time_aware_envelope = True
        return self.policy(simulator, eval_mode=eval_mode)


class ForcedUActor:
    """Eval-only amount ablation: keep discrete choice, force continuous u."""

    def __init__(self, actor, u_fixed: float, name: str):
        self.actor = actor
        self.u_fixed = float(u_fixed)
        self.name = name
        self.ablation = actor.ablation

    def __call__(self, simulator, eval_mode: bool = True):
        simulator.time_aware_envelope = bool(self.ablation.time_aware)

        def choose(sim):
            discrete, _u = self.actor.choose(sim, eval_mode=eval_mode)
            if discrete == 0:
                return 0, 0.0
            return discrete, self.u_fixed

        return run_discrete_policy(simulator, choose, soc_mode=self.ablation.soc_interval)


def _gate() -> dict:
    for name in ("PAPER_PROTOCOL.json", "CHECKPOINT_FREEZE.json", "TEST_LOCK.json"):
        if not (V3 / name).is_file():
            raise SystemExit(f"{name} missing; TEST stays closed")
    if CONSUMED.is_file():
        raise SystemExit("V3 evaluation already consumed; must not rerun")
    if method_tree_diff().strip():
        raise SystemExit("method tree differs from freeze")
    for line in git_porcelain().splitlines():
        path = line[3:].strip().replace("\\", "/")
        if path.startswith(("src/", "configs/", "tests/", "third_party/", "data/")):
            # Allow only new V3 data paths already locked.
            if path.startswith("data/routes_v2/synthcharge_v3_test") or path.startswith(
                "data/splits_v2/synthcharge_v3_test"
            ):
                continue
            raise SystemExit(f"frozen/scientific path dirty: {path}")
    freeze = load_json(V3 / "CHECKPOINT_FREEZE.json")
    if freeze.get("n_checkpoints") != 5:
        raise SystemExit("checkpoint freeze incomplete")
    for item in freeze["checkpoints"]:
        if item["method"] != "HybridPPO":
            raise SystemExit("DiscretePPO must not enter V3 evaluation")
        if sha256(resolve_repo_path(item["checkpoint"])) != item["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint hash mismatch: {item['checkpoint']}")
    lock = load_json(V3 / "TEST_LOCK.json")
    bad = [rel for rel, digest in lock["files"].items() if lf_sha256(resolve_repo_path(rel)) != digest]
    if bad:
        raise SystemExit(f"TEST_LOCK mismatch: {bad[:5]}")
    print("V3 GATE: protocol+freeze+lock OK; HybridPPO-only; TEST closed until --execute-once")
    return freeze


def _charge_class(instance, route, row: dict) -> str:
    if row.get("charge_class"):
        return row["charge_class"]
    from domain.load_convention import LoadConvention
    from simulation.simulator import FixedRouteSimulator, run_continue_only

    profile = PhysicsProfile.from_instance(instance, name=route.physics_profile or "synthcharge_linear")
    simulator = FixedRouteSimulator(
        instance, route.customer_ids, profile, LoadConvention.OFFICIAL_REFERENCE_PICKUP
    )
    continued = run_continue_only(simulator)
    if continued.feasible and simulator.state.completed:
        return "no_charge_required"
    if int(row.get("certificate_station_visits") or 0) >= 1:
        return "charging_required"
    return "unclassified"


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
        actor = load_hybrid_actor(resolve_repo_path(item["checkpoint"]))
        policies.append(
            {
                "method": "HybridPPO",
                "seed": item["seed"],
                "policy": actor,
                "checkpoint_sha256": item["checkpoint_sha256"],
                "training_git_sha": item["training_git_sha"],
            }
        )
        policies.append(
            {
                "method": "FA-HPPO-Min",
                "seed": item["seed"],
                "policy": ForcedUActor(actor, 0.0, "FA-HPPO-Min"),
                "checkpoint_sha256": item["checkpoint_sha256"],
                "training_git_sha": item["training_git_sha"],
            }
        )
        policies.append(
            {
                "method": "FA-HPPO-Max",
                "seed": item["seed"],
                "policy": ForcedUActor(actor, 1.0, "FA-HPPO-Max"),
                "checkpoint_sha256": item["checkpoint_sha256"],
                "training_git_sha": item["training_git_sha"],
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

    lock_sha = lf_sha256(V3 / "TEST_LOCK.json")
    eval_sha = git_head()
    records = []
    for route in routes:
        assert route.physics_profile == "synthcharge_linear", route.route_id
        instance = parse_route_instance(route)
        profile = PhysicsProfile.from_instance(instance, name="synthcharge_linear")
        assert profile.energy_law == "linear_distance", route.route_id
        row = raw_rows[route.route_id]
        charge_class = _charge_class(instance, route, row)
        context = {
            "benchmark": "synthcharge_v3_hppo_paper_test",
            "benchmark_role": "fresh_external_confirmatory_TEST_for_FA_HPPO_paper",
            "base_instance": route.base_instance,
            "raw_instance_id": route.raw_instance_id,
            "n_customers": route.n_customers,
            "layout": row.get("layout"),
            "length_bin": row.get("length_bin"),
            "charge_class": charge_class,
            "source_dataset": route.source_dataset,
            "physics_profile": "synthcharge_linear",
            "certificate_station_visits": row.get("certificate_station_visits"),
            "method_freeze_sha": METHOD_FREEZE_SHA,
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
            record = result.to_record(
                method=item["method"],
                seed=item["seed"],
                checkpoint_sha256=item["checkpoint_sha256"],
                training_git_sha=item["training_git_sha"],
                training_dataset="synthcharge",
                **context,
            )
            records.append(record)
    return records


def main() -> None:
    if "--execute-once" not in sys.argv:
        raise SystemExit("refusing to open V3 TEST without --execute-once")
    started = time.time()
    rows = evaluate()
    expected = 180 * (5 + 5 + 5 + 3)  # Hybrid + Min + Max + 3 baselines
    if len(rows) != expected:
        raise SystemExit(f"row count {len(rows)} != expected {expected}")
    methods = {r["method"] for r in rows}
    if "DiscretePPO" in methods:
        raise SystemExit("DiscretePPO leaked into V3 TEST")
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "synthcharge_test.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows) + "\n",
        encoding="utf-8",
    )
    dump_json(
        CONSUMED,
        {
            "consumed": True,
            "benchmark": "synthcharge_v3_hppo_paper_test",
            "n_routes": 180,
            "n_rows": len(rows),
            "methods": sorted(methods),
            "raw_file": path.relative_to(ROOT).as_posix(),
            "raw_sha256_lf": lf_sha256(path),
            "evaluation_git_sha": git_head(),
            "runtime_s": time.time() - started,
            "no_retrain_after_test": True,
            "no_checkpoint_reselection": True,
        },
    )
    print(json.dumps({"n_rows": len(rows), "methods": sorted(methods)}, indent=2))


if __name__ == "__main__":
    main()
