"""Native FRVCP benchmark.

Default: official e-VRO testdata.json routes on frvcpy-instance.json.
--fixtures-only: tiny smoke fixtures (not official published tours).
--montoya: optional XML sequential-ID tours for nonlinear sensitivity;
those are NOT official published FRVCP tours.

Never joined to EVRPTW-GR splits. Calls real ``frvcpy.solver.Solver``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.frvcp_greedy import greedy_frvcp  # noqa: E402
from baselines.frvcpy_adapter import NOT_EQUIVALENT  # noqa: E402
from baselines.frvcpy_reference import official_reference_specs, parity_against_testdata  # noqa: E402
from baselines.frvcpy_solver import frvcpy_available, load_instance_json, optimality_gap_percent, solve_native  # noqa: E402
from data.paths import EXTERNAL_DIR  # noqa: E402
from experiments.isolation import require_scenario  # noqa: E402
from experiments.provenance import detect_device, git_sha  # noqa: E402
from routing.serialize import canonical_dumps  # noqa: E402
from data.paths import RESULTS_DIR  # noqa: E402


def _n_insertions(original, feas_route) -> int | None:
    if feas_route is None:
        return None
    try:
        return max(0, len(list(feas_route)) - len(list(original)))
    except TypeError:
        return None


def _load_instance(root: Path, spec: dict) -> dict | None:
    if spec.get("instance"):
        path = root / spec["instance"]
        if path.is_file():
            return load_instance_json(path)
    xml_rel = spec.get("instance_xml")
    if xml_rel:
        xml_path = root / xml_rel
        json_path = root / "json" / (Path(xml_rel).stem + ".json")
        if json_path.is_file():
            return load_instance_json(json_path)
        if xml_path.is_file():
            try:
                from frvcpy.translator import translate
            except ImportError:
                return None
            json_path.parent.mkdir(parents=True, exist_ok=True)
            translate(str(xml_path), to_filename=str(json_path))
            return load_instance_json(json_path)
    return None


def _records_for_spec(root: Path, spec: dict, *, scenario: str, experiment_id: str) -> list[dict]:
    instance = _load_instance(root, spec)
    if instance is None:
        return [
            {
                "method": "frvcpy_Solver",
                "route_id": spec["route_id"],
                "split": "native_frvcp",
                "seed": 0,
                "scenario": scenario,
                "experiment_id": experiment_id,
                "feasible": False,
                "reason": "instance_unavailable_or_translate_failed",
                "equivalent_to_evrptwgr": "native_frvcp",
                "git_sha": git_sha(),
                "device": detect_device(),
                "frvcpy_available": frvcpy_available(),
            }
        ]
    route = list(spec["nodes"])
    q_init = spec.get("q_init")
    if q_init is None:
        q_init = float(instance.get("max_q") or 0.0)
    q_init = float(q_init)
    solved = solve_native(instance, route, q_init)
    greedy = greedy_frvcp(instance, route, q_init, full_charge=False)
    full = greedy_frvcp(instance, route, q_init, full_charge=True)
    gap_g = optimality_gap_percent(greedy.get("duration"), solved.duration) if greedy.get("feasible") else None
    gap_f = optimality_gap_percent(full.get("duration"), solved.duration) if full.get("feasible") else None
    common = {
        "route_id": spec["route_id"],
        "split": "native_frvcp",
        "seed": 0,
        "scenario": scenario,
        "experiment_id": experiment_id,
        "equivalent_to_evrptwgr": "native_frvcp",
        "git_sha": git_sha(),
        "device": detect_device(),
        "n_insertions": _n_insertions(route, solved.feasible_route),
    }
    return [
        {
            **common,
            "method": "frvcpy_Solver",
            "feasible": solved.status == "optimal" and solved.duration is not None,
            "reason": solved.status,
            "route_completion_time": solved.duration,
            "completion_time_all_routes": solved.duration,
            "runtime_s": solved.runtime_s,
            "frvcpy_available": frvcpy_available(),
        },
        {
            **common,
            "method": "FRVCPGreedyMin",
            "feasible": bool(greedy["feasible"]),
            "reason": greedy.get("reason"),
            "route_completion_time": greedy.get("duration"),
            "completion_time_all_routes": greedy.get("duration"),
            "optimality_gap_percent": gap_g,
        },
        {
            **common,
            "method": "FRVCPGreedyFull",
            "feasible": bool(full["feasible"]),
            "reason": full.get("reason"),
            "route_completion_time": full.get("duration"),
            "completion_time_all_routes": full.get("duration"),
            "optimality_gap_percent": gap_f,
        },
    ]


def _dump(records: list[dict], run_id: str, extra: dict) -> Path:
    raw = RESULTS_DIR / "raw" / f"{run_id}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical_dumps(row) for row in records]
    raw.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    run_dir = RESULTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        canonical_dumps({"run_id": run_id, "n_records": len(records), **extra}) + "\n",
        encoding="utf-8",
    )
    return raw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=EXTERNAL_DIR / "frvcpy")
    parser.add_argument("--run-id", default="frvcpy_native")
    parser.add_argument("--scenario", default="frvcpy_native")
    parser.add_argument("--fixtures-only", action="store_true", help="tiny fixtures, not official tours")
    parser.add_argument("--montoya", action="store_true", help="optional sequential-ID XML tours")
    parser.add_argument("--parity-only", action="store_true", help="run official testdata Solver parity and exit")
    args = parser.parse_args(argv)
    scenario = require_scenario(args.scenario)
    if args.parity_only:
        report = parity_against_testdata(args.data)
        print(canonical_dumps(report))
        if not report["available"]:
            return 1
        return 0 if not report["errors"] else 1
    if args.fixtures_only:
        routes_path = args.data / "routes.json"
        root = args.data
        specs = json.loads(routes_path.read_text(encoding="utf-8"))["routes"]
        official = False
    elif args.montoya:
        bench = args.data / "benchmark"
        routes_path = bench / "routes.json"
        root = bench
        payload = json.loads(routes_path.read_text(encoding="utf-8"))
        specs = payload["routes"]
        official = False
    else:
        root = args.data
        specs = official_reference_specs(root)
        official = True
    records = []
    for spec in specs:
        recs = _records_for_spec(root, spec, scenario=scenario, experiment_id=args.run_id)
        if official and spec.get("known_obj") is not None:
            recs[0]["known_obj"] = spec["known_obj"]
            recs[0]["official_published_tour"] = True
        else:
            recs[0]["official_published_tour"] = False
        records.extend(recs)
    records.append(
        {
            "method": "evrptwgr_surrogate_flag",
            "split": "not_a_benchmark",
            "scenario": scenario,
            "experiment_id": args.run_id,
            "seed": 0,
            "equivalent_to_evrptwgr": NOT_EQUIVALENT,
            "note": "evrptwgr_to_frvcp_surrogate remains not_equivalent; not reported as EVRPTW-GR optimality",
        }
    )
    extra = {
        "scenario": scenario,
        "official_published_tours": official,
        "frvcpy_available": frvcpy_available(),
        "n_specs": len(specs),
    }
    path = _dump(records, args.run_id, extra)
    print(f"wrote {len(records)} records to {path} frvcpy_available={frvcpy_available()} n_specs={len(specs)} official={official}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
