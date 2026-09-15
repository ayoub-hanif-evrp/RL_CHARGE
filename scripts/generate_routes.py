"""Generate the method-independent frozen-route corpus with PyVRP.

Does not filter routes by charging behaviour, RL, or energy feasibility.
Charging feasibility is recorded as ``unverified``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.paths import ROUTES_DIR  # noqa: E402
from physics.parameters import DEFAULT_PROFILE_NAME  # noqa: E402
from routing.config import PyVRPConfig  # noqa: E402
from routing.corpus import generate_corpus  # noqa: E402
from routing.pyvrp_generator import PyVRPGenerator  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROUTES_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override MaxIterations for every network group (tests only).",
    )
    args = parser.parse_args(argv)

    config = PyVRPConfig.from_toml(args.config)
    if args.seed is not None:
        object.__setattr__(config, "seed", args.seed)
    generator = PyVRPGenerator(config, max_iterations_override=args.max_iterations)
    result = generate_corpus(
        out_dir=args.out,
        profile_name=args.profile,
        config=config,
        generator=generator,
        write_metadata=True,
    )
    print(
        f"attempted={result.attempted} routes={len(result.routes)} "
        f"failures={len(result.failures)}"
    )
    for failure in result.failures:
        print(f"FAILURE {failure.relative_path}: {failure.error}")
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
