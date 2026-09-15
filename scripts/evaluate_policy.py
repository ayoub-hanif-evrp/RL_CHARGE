"""Evaluate a named policy on frozen routes. Reports completion time, energy, distance separately."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.greedy import GreedyFullCharge, GreedyMinimumSufficientCharge  # noqa: E402
from baselines.lookahead import OneStepLookahead  # noqa: E402
from data.parser import parse_instance  # noqa: E402
from data.paths import RAW_EVRPTW_GR_DIR  # noqa: E402
from domain.load_convention import LoadConvention  # noqa: E402
from experiments.evaluate import evaluate_policy  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402


POLICIES = {
    "greedy_min": GreedyMinimumSufficientCharge(),
    "greedy_full": GreedyFullCharge(),
    "lookahead": OneStepLookahead(),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="greedy_min")
    args = parser.parse_args(argv)
    route = read_jsonl(args.route)[0]
    instance = parse_instance(RAW_EVRPTW_GR_DIR / route.relative_path)
    result = evaluate_policy(
        instance=instance,
        route=route,
        policy=POLICIES[args.policy],
        load_convention=LoadConvention.OFFICIAL_REFERENCE_PICKUP,
    )
    print(
        json.dumps(
            {
                "route_id": result.route_id,
                "feasible": result.feasible,
                "completed": result.completed,
                "route_completion_time": result.route_completion_time,
                "total_net_energy": result.total_net_energy,
                "total_distance": result.total_distance,
                "n_station_visits": result.n_station_visits,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
