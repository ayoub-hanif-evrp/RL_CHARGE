"""Replay a frozen route. Default policy is Continue-only (not a corpus filter)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.parser import parse_instance  # noqa: E402
from data.paths import RAW_EVRPTW_GR_DIR  # noqa: E402
from physics.parameters import DEFAULT_PROFILE_NAME, PhysicsProfile  # noqa: E402
from routing.serialize import loads_route  # noqa: E402
from simulation.simulator import FixedRouteSimulator, run_continue_only  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--policy", choices=["continue"], default="continue")
    parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME)
    args = parser.parse_args(argv)

    raw = args.route.read_text(encoding="utf-8").splitlines()[0]
    route = loads_route(raw)
    instance = parse_instance(RAW_EVRPTW_GR_DIR / route.relative_path)
    profile = PhysicsProfile.from_instance(instance, name=args.profile)
    simulator = FixedRouteSimulator(instance, route.customer_ids, profile)
    if args.policy != "continue":
        raise SystemExit(f"unsupported policy {args.policy}")
    result = run_continue_only(simulator)
    payload = {
        "route_id": route.route_id,
        "customer_ids": list(route.customer_ids),
        "served_customers": list(simulator.state.served_customers),
        "transition_feasible": result.feasible,
        "reason": None if result.reason is None else result.reason.value,
        "metrics": result.metrics.to_dict(),
        "events": [event.to_dict() for event in simulator.state.events],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
