"""Build V2 gold and certified-PyVRP corpora. Does not modify V1 files or open TEST."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.parser import parse_instance  # noqa: E402
from data.paths import RAW_EVRPTW_GR_DIR, ROUTES_DIR  # noqa: E402
from experiments.provenance import git_dirty, git_sha, v2_corpus_hashes, v2_split_hashes  # noqa: E402
from experiments.v2_scope import CONSUMED_TEST_PARENT_SET, split_for_parent  # noqa: E402
from rl.ablation import AblationConfig  # noqa: E402
from rl.env import ShieldedRouteEnv  # noqa: E402
from rl.features import extract_features  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402
from routing.v2_corpus import (  # noqa: E402
    build_gold_routes,
    instance_index,
    load_mapping,
    load_v1_dev_routes,
    repair_pyvrp_routes,
    write_route_bundle,
)
from simulation.shield import evaluate_shield, soc_interval_for_station  # noqa: E402
from exact.charging_certificate import solve_charging_certificate  # noqa: E402
from physics.parameters import PhysicsProfile  # noqa: E402
from domain.load_convention import LoadConvention  # noqa: E402


def _write_splits(directory: Path, rows, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        ids = sorted({route.raw_instance_id for route, extra in rows if extra["split"] == split})
        parents = sorted({route.base_instance for route, extra in rows if extra["split"] == split})
        payload = {
            "split": split,
            "instance_ids": ids,
            "base_instances": parents,
            "n_routes": sum(1 for _route, extra in rows if extra["split"] == split),
            "consumed_test_parents_excluded": sorted(CONSUMED_TEST_PARENT_SET),
        }
        (directory / f"{stem}_{split}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _search_gold(rows) -> dict:
    files = instance_index()
    found = 0
    timeout = 0
    exhausted = 0
    records = []
    for index, (route, extra) in enumerate(rows, start=1):
        instance = parse_instance(files[route.raw_instance_id])
        result = solve_charging_certificate(
            instance, route.customer_ids, max_expansions=8_000, max_seconds=3.0
        )
        if result.status == "certified_feasible":
            found += 1
        elif result.status == "unverified_timeout":
            timeout += 1
        else:
            exhausted += 1
        records.append(
            {
                "route_id": route.route_id,
                "official_certificate_replay": True,
                "search_status": result.status,
                "search_runtime_s": result.runtime_s,
                "search_station_visits": result.n_station_visits,
                "search_completion_time": result.completion_time,
                "official_station_visits": extra["certificate_station_visits"],
                "official_completion_time": extra["certificate_completion_time"],
            }
        )
        if index % 10 == 0:
            print(f"gold search {index}/{len(rows)} found={found}", flush=True)
    return {
        "n_gold": len(rows),
        "search_certified": found,
        "unverified_timeout": timeout,
        "unverified_search_exhausted": exhausted,
        "records": records,
    }


def _comparison(mapping, pyvrp_rows, summary: dict) -> None:
    official = Counter()
    parents = {}
    for tour in mapping["tours"]:
        instance_id = tour.get("instance_id")
        if not instance_id:
            continue
        official[instance_id] += 1
        parents[instance_id] = tour.get("base_instance")
    v1 = Counter()
    for route in read_jsonl(ROUTES_DIR / "corpus.jsonl"):
        v1[route.raw_instance_id] += 1
    v2 = Counter(route.raw_instance_id for route, _extra in pyvrp_rows)
    out = ROOT / "results" / "v2" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for instance_id in sorted(official):
        parent = parents[instance_id]
        withheld = parent in CONSUMED_TEST_PARENT_SET
        rows.append(
            {
                "instance_id": instance_id,
                "base_instance": parent,
                "consumed_v1_test_parent": withheld,
                "official_tours": official[instance_id],
                "original_pyvrp_routes": v1[instance_id],
                "v2_certified_routes": "" if withheld else v2[instance_id],
            }
        )
    with (out / "route_construction_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dev = [row for row in rows if not row["consumed_v1_test_parent"]]
    def _sum(key):
        return sum(int(row[key]) for row in dev if row[key] != "")
    lines = [
        "# V2 route-construction diagnostic",
        "",
        "Consumed V1 TEST parents are listed in the CSV with V2 counts withheld.",
        "This is not an optimality claim and does not force V2 counts to match official plans.",
        "",
        f"- official instance variants: {len(rows)}",
        f"- development variants compared: {len(dev)}",
        f"- official tours on development variants: {_sum('official_tours')}",
        f"- original PyVRP routes on development variants: {_sum('original_pyvrp_routes')}",
        f"- V2 certified routes on development variants: {_sum('v2_certified_routes')}",
        "",
        "Admitted benchmark routes are complete certified partitions only.",
        "An unresolved source route is quarantined with all of its customers.",
        "",
        f"- original development routes: {summary.get('original_pyvrp_routes')}",
        f"- admitted certified routes: {summary.get('final_certified_routes')}",
        f"- search timeouts: {summary.get('search_timeouts')}",
        f"- search exhaustions: {summary.get('search_exhausted')}",
        f"- customer coverage: {json.dumps(summary.get('customer_coverage'), sort_keys=True)}",
        "",
    ]
    (out / "route_construction_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _expert_traces(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    files = instance_index()
    ablation = AblationConfig(time_aware=True)
    lines = []
    for route, extra in rows:
        if extra["split"] != "train":
            continue
        instance = parse_instance(files[route.raw_instance_id])
        profile = PhysicsProfile.from_instance(instance)
        env = ShieldedRouteEnv(
            instance,
            route,
            profile,
            LoadConvention.OFFICIAL_REFERENCE_PICKUP,
            ablation=ablation,
        )
        sim = env.simulator
        for step in extra["certificate_trace"]:
            decision = evaluate_shield(sim, time_aware=True)
            features = extract_features(
                sim,
                use_remaining_route=True,
                use_terrain_load_features=True,
                soc_interval=ablation.soc_interval,
            )
            if step["kind"] == "CONTINUE":
                discrete = 0
                station_id = None
                target = None
                u = 0.0
                legal = bool(decision.mask[0])
            else:
                station_id = step["station_id"]
                target = float(step["target_soc"])
                discrete = 1 + decision.station_ids.index(station_id)
                interval = soc_interval_for_station(sim, station_id, time_aware=True)
                width = interval.soc_upper - interval.soc_lower
                u = 0.0 if width <= 1e-8 else (target - interval.soc_lower) / width
                legal = bool(decision.mask[discrete]) if discrete < len(decision.mask) else False
            lines.append(
                json.dumps(
                    {
                        "route_id": route.route_id,
                        "features": {key: value.tolist() for key, value in features.as_arrays().items()},
                        "legal_mask": [bool(item) for item in decision.mask],
                        "expert_discrete": discrete,
                        "expert_station_id": station_id,
                        "target_soc": target,
                        "u": u,
                        "legal_under_time_aware_shield": legal,
                    }
                )
            )
            if step["kind"] == "CONTINUE":
                from simulation.actions import ContinueAction

                result = sim.step(ContinueAction())
            else:
                from simulation.actions import ChargeAction

                result = sim.step(ChargeAction(station_id, float(step["target_soc"])))
            if not result.feasible:
                raise AssertionError(f"expert replay failed for {route.route_id}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "gold", "pyvrp"), default="all")
    args = parser.parse_args()
    mapping = load_mapping()
    if args.stage in {"all", "gold"}:
        gold_rows, gold_summary = build_gold_routes(mapping)
        for route, _extra in gold_rows:
            if route.base_instance in CONSUMED_TEST_PARENT_SET:
                raise SystemExit("TEST parent in gold corpus")
            split_for_parent(route.base_instance)
        gold_dir = ROOT / "data" / "routes_v2" / "gold_official"
        write_route_bundle(gold_dir, gold_rows, gold_summary)
        _write_splits(ROOT / "data" / "splits_v2", gold_rows, "gold")
        print("gold", json.dumps(gold_summary), flush=True)
        search_report = _search_gold(gold_rows)
        report_dir = ROOT / "results" / "v2" / "diagnostics"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "gold_search_report.json").write_text(
            json.dumps({key: value for key, value in search_report.items() if key != "records"}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (report_dir / "gold_search_records.jsonl").write_text(
            "\n".join(json.dumps(row) for row in search_report["records"]) + "\n",
            encoding="utf-8",
        )
        print(
            "gold search",
            {k: search_report[k] for k in ("search_certified", "unverified_timeout", "unverified_search_exhausted")},
            flush=True,
        )
        _expert_traces(gold_rows, ROOT / "results" / "v2" / "expert_traces" / "gold_train.jsonl")
        if args.stage == "gold":
            return
    dev_routes = load_v1_dev_routes()
    py_rows, py_summary = repair_pyvrp_routes(dev_routes)
    py_dir = ROOT / "data" / "routes_v2" / "certified_pyvrp"
    quarantined = py_summary.get("quarantined", [])
    public = {key: value for key, value in py_summary.items() if key != "quarantined"}
    public["quarantine_file"] = "quarantine.jsonl"
    write_route_bundle(py_dir, py_rows, public)
    (py_dir / "quarantine.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in quarantined),
        encoding="utf-8",
    )
    (py_dir / "repair_report.json").write_text(
        json.dumps({**public, "quarantined": quarantined}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_splits(ROOT / "data" / "splits_v2", py_rows, "pyvrp")
    printable = {key: value for key, value in public.items() if key != "quarantined"}
    print("pyvrp", json.dumps(printable), flush=True)
    _comparison(mapping, py_rows, public)
    _expert_traces(py_rows, ROOT / "results" / "v2" / "expert_traces" / "pyvrp_train.jsonl")
    dirty = git_dirty()
    provenance = {
        "stage": "v2_development",
        "git_sha": git_sha(),
        "git_dirty": dirty,
        "run_kind": "dirty_exploratory" if dirty else "clean_sha",
        "final_v2_requires_clean_frozen_sha": True,
        "v1_test_parents_quarantined": sorted(CONSUMED_TEST_PARENT_SET),
        "v2_corpus_hashes": v2_corpus_hashes(),
        "v2_split_hashes": v2_split_hashes(),
        "customer_coverage": public.get("customer_coverage"),
        "note": "Final V2 experiments must run from a clean frozen V2 SHA. This file records the development corpora.",
    }
    (ROOT / "results" / "v2" / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print("v2 corpora written", flush=True)


if __name__ == "__main__":
    main()
