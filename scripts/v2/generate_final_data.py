"""Generate the frozen SynthCharge benchmark and the legacy EVRPTW-GR challenge.

Run only after V2_METHOD_FREEZE_SHA. This script does not train a policy and
does not change generator parameters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
THIRD = ROOT / "third_party" / "SynthCharge_v1.0"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(THIRD))

from data_generator import generate_milp_feasible_instance  # noqa: E402

from routing.synthcharge_benchmark import fill_split, load_spec, quota_summary  # noqa: E402
from routing.v2_corpus import build_legacy_challenge_routes, write_route_bundle  # noqa: E402


def _generate(n_customers, n_stations, layout, seed, arguments):
    return generate_milp_feasible_instance(
        n_customers=int(n_customers),
        n_stations=int(n_stations),
        instance_type=layout,
        random_seed=int(seed),
        **arguments,
    )


def _write_split(name: str, rows: list) -> None:
    payload = {
        "split": name,
        "benchmark": "synthcharge_final",
        "role": "fresh_external_confirmatory_benchmark" if name == "test" else "development_split",
        "instance_ids": [route.raw_instance_id for route, _extra in rows],
        "route_ids": [route.route_id for route, _extra in rows],
        "generator_seeds": [int(extra["generator_seed"]) for _route, extra in rows],
        "n_routes": len(rows),
        "quota_summary": quota_summary(rows),
    }
    path = ROOT / "data" / "splits_v2" / f"synthcharge_{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_synthcharge() -> None:
    spec = load_spec()
    audit = []
    for split in ("train", "validation", "test"):
        print(f"filling {split}", flush=True)
        rows, records = fill_split(spec, split, root=ROOT, generate_instance=_generate)
        directory = ROOT / "data" / "routes_v2" / "synthcharge_final" / split
        write_route_bundle(
            directory,
            rows,
            {
                "benchmark": "synthcharge_final",
                "split": split,
                "role": "fresh_external_confirmatory_benchmark" if split == "test" else "development_split",
                "source_commit": spec["source_commit"],
                "paper_arxiv": spec["paper_arxiv"],
                "structural_screen_is_not_a_charging_certificate": True,
                "physics_profile": "synthcharge_linear",
                "n_routes": len(rows),
                "quota_summary": quota_summary(rows),
                "one_route_per_instance": True,
            },
        )
        _write_split("train" if split == "train" else ("validation" if split == "validation" else "test"), rows)
        audit.extend(records)
        print(f"{split} accepted {len(rows)}", flush=True)
    audit_path = ROOT / "data" / "routes_v2" / "synthcharge_final" / "candidate_audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":"), sort_keys=True) for row in audit) + "\n",
        encoding="utf-8",
    )
    seeds = {}
    for split in ("train", "validation", "test"):
        payload = json.loads((ROOT / "data" / "splits_v2" / f"synthcharge_{split}.json").read_text(encoding="utf-8"))
        seeds[split] = set(payload["generator_seeds"])
    if seeds["train"] & seeds["validation"] or seeds["validation"] & seeds["test"] or seeds["train"] & seeds["test"]:
        raise SystemExit("generator seed reused across SynthCharge splits")
    meta = {
        "benchmark": "synthcharge_final",
        "source_commit": spec["source_commit"],
        "paper_arxiv": spec["paper_arxiv"],
        "physics_profile": "synthcharge_linear",
        "n_train": 180,
        "n_validation": 90,
        "n_test": 90,
        "not_evrptw_gr_gradient_physics": True,
    }
    (ROOT / "data" / "routes_v2" / "synthcharge_final" / "corpus_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def generate_legacy() -> None:
    rows, summary = build_legacy_challenge_routes()
    write_route_bundle(ROOT / "data" / "routes_v2" / "legacy_evrptwgr_challenge", rows, summary)
    print(json.dumps(summary), flush=True)


def main() -> None:
    if "--execute" not in sys.argv:
        raise SystemExit("refusing to generate without --execute")
    which = sys.argv[sys.argv.index("--execute") + 1] if len(sys.argv) > sys.argv.index("--execute") + 1 else "all"
    if which in {"all", "synthcharge"}:
        generate_synthcharge()
    if which in {"all", "legacy"}:
        generate_legacy()


if __name__ == "__main__":
    main()
