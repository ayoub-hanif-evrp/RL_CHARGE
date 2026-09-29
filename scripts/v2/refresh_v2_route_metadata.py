"""Rewrite V2 route metadata only: capacity label and routing time-window flag."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data.parser import parse_instance  # noqa: E402
from data.paths import RAW_EVRPTW_GR_DIR  # noqa: E402
from routing.audit import routing_time_window_feasible  # noqa: E402


def main() -> None:
    for relative in (
        "data/routes_v2/gold_official/corpus.jsonl",
        "data/routes_v2/certified_pyvrp/corpus.jsonl",
    ):
        path = ROOT / relative
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instance = parse_instance(RAW_EVRPTW_GR_DIR / row["relative_path"])
            row["capacity_policy"] = "official_evrptwgr_payload_capacity_3650kg"
            row["routing_tw_feasible"] = routing_time_window_feasible(
                instance, row["customer_ids"], "official_evrptwgr"
            )
            lines.append(json.dumps(row, separators=(",", ":"), sort_keys=True))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(relative, len(lines))


if __name__ == "__main__":
    main()
