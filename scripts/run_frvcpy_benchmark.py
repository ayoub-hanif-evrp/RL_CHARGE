"""Native FRVCP benchmark: frvcpy Solver plus thin-env greedy. Never joined to EVRPTW-GR splits."""

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
from baselines.frvcpy_solver import frvcpy_available, load_instance_json, optimality_gap_percent, solve_native  # noqa: E402
from data.paths import EXTERNAL_DIR  # noqa: E402
from experiments.batch import dump_run  # noqa: E402
from experiments.provenance import detect_device, git_sha  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=EXTERNAL_DIR / "frvcpy")
    parser.add_argument("--run-id", default="frvcpy_native")
    args = parser.parse_args(argv)
    instance_path = args.data / "frvcpy-instance.json"
    routes_path = args.data / "routes.json"
    routes = json.loads(routes_path.read_text(encoding="utf-8"))
    records = []
    for spec in routes["routes"]:
        inst_file = args.data / spec.get("instance", "frvcpy-instance.json")
        instance = load_instance_json(inst_file)
        route = list(spec["nodes"])
        q_init = float(spec.get("q_init", instance["max_q"]))
        solved = solve_native(instance, route, q_init)
        greedy = greedy_frvcp(instance, route, q_init, full_charge=False)
        full = greedy_frvcp(instance, route, q_init, full_charge=True)
        gap_g = optimality_gap_percent(greedy.get("duration"), solved.duration) if greedy.get("feasible") else None
        gap_f = optimality_gap_percent(full.get("duration"), solved.duration) if full.get("feasible") else None
        records.append(
            {
                "method": "frvcpy_Solver",
                "route_id": spec["route_id"],
                "split": "native_frvcp",
                "seed": 0,
                "feasible": solved.status == "optimal" and solved.duration is not None,
                "reason": solved.status,
                "route_completion_time": solved.duration,
                "completion_time_all_routes": solved.duration,
                "runtime_s": solved.runtime_s,
                "equivalent_to_evrptwgr": "native_frvcp",
                "git_sha": git_sha(),
                "device": detect_device(),
                "frvcpy_available": frvcpy_available(),
            }
        )
        records.append(
            {
                "method": "FRVCPGreedyMin",
                "route_id": spec["route_id"],
                "split": "native_frvcp",
                "seed": 0,
                "feasible": bool(greedy["feasible"]),
                "reason": greedy.get("reason"),
                "route_completion_time": greedy.get("duration"),
                "completion_time_all_routes": greedy.get("duration"),
                "optimality_gap_percent": gap_g,
                "equivalent_to_evrptwgr": "native_frvcp",
                "git_sha": git_sha(),
                "device": detect_device(),
            }
        )
        records.append(
            {
                "method": "FRVCPGreedyFull",
                "route_id": spec["route_id"],
                "split": "native_frvcp",
                "seed": 0,
                "feasible": bool(full["feasible"]),
                "reason": full.get("reason"),
                "route_completion_time": full.get("duration"),
                "completion_time_all_routes": full.get("duration"),
                "optimality_gap_percent": gap_f,
                "equivalent_to_evrptwgr": "native_frvcp",
                "git_sha": git_sha(),
                "device": detect_device(),
            }
        )
    # Keep the EVRPTW-GR surrogate flag documented; do not join IDs into splits.
    records.append(
        {
            "method": "evrptwgr_surrogate_flag",
            "split": "not_a_benchmark",
            "equivalent_to_evrptwgr": NOT_EQUIVALENT,
            "note": "evrptwgr_to_frvcp_surrogate remains not_equivalent; not reported as EVRPTW-GR optimality",
        }
    )
    path = dump_run(records, args.run_id)
    print(f"wrote {len(records)} records to {path} frvcpy_available={frvcpy_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
