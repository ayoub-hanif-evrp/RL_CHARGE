"""Scan the raw EVRPTW-GR dataset and summarise it.

Reads every instance through the canonical parser in ``src/data``, reports
validation problems, and writes machine-readable metadata to
``data/processed``. The raw dataset is only ever read.

    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --fail-on-warning
    python scripts/inspect_dataset.py --root path/to/EVRPTW_GR --out-dir /tmp/out
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import (  # noqa: E402
    EVRPTWGRError,
    Severity,
    find_missing_variants,
    scan_dataset,
)
from data.paths import PROCESSED_DIR, RAW_EVRPTW_GR_DIR  # noqa: E402

MANIFEST_NAME = "dataset_manifest.csv"
SUMMARY_NAME = "dataset_summary.json"

MANIFEST_COLUMNS = (
    "instance_id",
    "relative_path",
    "network_group",
    "customer_folder",
    "terrain_variant",
    "terrain_folder",
    "base_instance",
    "customer_distribution",
    "schedule_type",
    "n_nodes",
    "n_depots",
    "n_stations",
    "n_customers",
    "declared_customer_count",
    "declared_station_count",
    "demand_min",
    "demand_max",
    "demand_total",
    "altitude_min",
    "altitude_max",
    "ready_time_min",
    "due_date_max",
    "service_time_min",
    "service_time_max",
    "tank_capacity",
    "load_capacity",
    "consumption_rate",
    "inverse_refueling_rate",
    "average_velocity",
    "curb_weight",
    "tank_capacity_label",
    "n_errors",
    "n_warnings",
)


def _spread(values):
    """Return min/max/median for a sequence, or nulls when it is empty."""
    if not values:
        return {"min": None, "max": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def _instance_row(instance, issues):
    meta = instance.metadata
    vehicle = instance.vehicle
    customer_demands = [n.demand for n in instance.customers]
    altitudes = [n.altitude for n in instance.nodes]
    service_times = [n.service_time for n in instance.customers]

    return {
        "instance_id": meta.instance_id,
        "relative_path": meta.relative_path,
        "network_group": meta.network_group.value,
        "customer_folder": meta.customer_folder,
        "terrain_variant": meta.terrain_variant.name,
        "terrain_folder": meta.terrain_folder,
        "base_instance": meta.base_instance,
        "customer_distribution": meta.customer_distribution,
        "schedule_type": meta.schedule_type,
        "n_nodes": len(instance.nodes),
        "n_depots": len(instance.depots),
        "n_stations": len(instance.stations),
        "n_customers": len(instance.customers),
        "declared_customer_count": meta.declared_customer_count,
        "declared_station_count": meta.declared_station_count,
        "demand_min": min(customer_demands) if customer_demands else None,
        "demand_max": max(customer_demands) if customer_demands else None,
        "demand_total": sum(customer_demands) if customer_demands else None,
        "altitude_min": min(altitudes),
        "altitude_max": max(altitudes),
        "ready_time_min": min(n.ready_time for n in instance.nodes),
        "due_date_max": max(n.due_date for n in instance.nodes),
        "service_time_min": min(service_times) if service_times else None,
        "service_time_max": max(service_times) if service_times else None,
        "tank_capacity": vehicle.tank_capacity,
        "load_capacity": vehicle.load_capacity,
        "consumption_rate": vehicle.consumption_rate,
        "inverse_refueling_rate": vehicle.inverse_refueling_rate,
        "average_velocity": vehicle.average_velocity,
        "curb_weight": vehicle.curb_weight,
        "tank_capacity_label": vehicle.label_for("Q"),
        "n_errors": sum(1 for i in issues if i.severity is Severity.ERROR),
        "n_warnings": sum(1 for i in issues if i.severity is Severity.WARNING),
    }


def build_manifest(result):
    by_path = {}
    for issue in result.issues:
        by_path.setdefault(issue.relative_path, []).append(issue)
    return [
        _instance_row(instance, by_path.get(instance.metadata.relative_path, []))
        for instance in result.instances
    ]


def build_summary(result, root, rows):
    instances = result.instances
    customer_counts = [r["n_customers"] for r in rows]
    station_counts = [r["n_stations"] for r in rows]
    altitudes = [n.altitude for i in instances for n in i.nodes]
    demands = [n.demand for i in instances for n in i.customers]
    ready_times = [n.ready_time for i in instances for n in i.nodes]
    due_dates = [n.due_date for i in instances for n in i.nodes]
    missing = find_missing_variants(instances)

    return {
        "dataset": {
            "name": "EVRPTW-GR",
            "doi": "10.17632/srfdbp2twv.1",
            "root": str(root),
        },
        "counts": {
            "files_scanned": len(instances) + len(result.unparsed),
            "instances_parsed": len(instances),
            "instances_unparsed": len(result.unparsed),
            "by_network_group": dict(Counter(r["network_group"] for r in rows)),
            "by_customer_folder": dict(Counter(r["customer_folder"] for r in rows)),
            "by_terrain_variant": dict(Counter(r["terrain_variant"] for r in rows)),
            "by_customer_count": {
                str(k): v for k, v in sorted(Counter(customer_counts).items())
            },
            "by_schedule_type": {
                str(k): v for k, v in sorted(Counter(r["schedule_type"] for r in rows).items())
            },
            "by_customer_distribution": dict(
                Counter(r["customer_distribution"] for r in rows)
            ),
        },
        "ranges": {
            "customers_per_instance": _spread(customer_counts),
            "stations_per_instance": _spread(station_counts),
            "altitude": _spread(altitudes),
            "customer_demand": _spread(demands),
            "ready_time": _spread(ready_times),
            "due_date": _spread(due_dates),
        },
        "vehicle_parameters": {
            "tank_capacity": sorted({r["tank_capacity"] for r in rows}),
            "load_capacity": sorted({r["load_capacity"] for r in rows}),
            "consumption_rate": sorted({r["consumption_rate"] for r in rows}),
            "inverse_refueling_rate": sorted({r["inverse_refueling_rate"] for r in rows}),
            "average_velocity": sorted({r["average_velocity"] for r in rows}),
            "curb_weight_present": sum(1 for r in rows if r["curb_weight"] is not None),
            "curb_weight_absent": sum(1 for r in rows if r["curb_weight"] is None),
        },
        "validation": {
            "n_errors": len(result.errors),
            "n_warnings": len(result.warnings),
            "by_code": dict(Counter(i.code for i in result.issues)),
            "unparsed_files": result.unparsed,
            "issues": [i.as_dict() for i in result.issues],
        },
        "coverage": {
            "documented_variants_per_instance": 3,
            "n_missing_variant_files": len(missing),
            "missing_variants": missing,
        },
    }


def write_outputs(rows, summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    summary_path = out_dir / SUMMARY_NAME

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest_path, summary_path


def print_report(summary, manifest_path, summary_path):
    counts = summary["counts"]
    ranges = summary["ranges"]
    validation = summary["validation"]
    coverage = summary["coverage"]

    print("EVRPTW-GR dataset inspection")
    print(f"  root: {summary['dataset']['root']}")
    print()
    print(f"  files scanned     : {counts['files_scanned']}")
    print(f"  parsed            : {counts['instances_parsed']}")
    print(f"  failed to parse   : {counts['instances_unparsed']}")
    print()
    print("  by network group  :", counts["by_network_group"])
    print("  by terrain variant:", counts["by_terrain_variant"])
    print("  by customer count :", counts["by_customer_count"])
    print()
    for label, key in (
        ("customers/instance", "customers_per_instance"),
        ("stations/instance ", "stations_per_instance"),
        ("altitude          ", "altitude"),
        ("customer demand   ", "customer_demand"),
        ("ready time        ", "ready_time"),
        ("due date          ", "due_date"),
    ):
        spread = ranges[key]
        print(f"  {label}: min={spread['min']} max={spread['max']} median={spread['median']}")
    print()
    print(f"  errors   : {validation['n_errors']}")
    print(f"  warnings : {validation['n_warnings']}")
    for code, count in sorted(validation["by_code"].items()):
        print(f"      {code}: {count}")
    print()
    print(
        f"  documented variant files absent from this download: "
        f"{coverage['n_missing_variant_files']}"
    )
    if manifest_path is None:
        print("  (--no-write: no files written)")
    else:
        print()
        print(f"  wrote {manifest_path}")
        print(f"  wrote {summary_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=f"raw EVRPTW-GR directory (default: {RAW_EVRPTW_GR_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"where to write generated metadata (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print the report without writing files"
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="exit non-zero if any warning is reported, not just errors",
    )
    args = parser.parse_args(argv)

    try:
        result = scan_dataset(args.root)
    except EVRPTWGRError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = build_manifest(result)
    root = args.root or RAW_EVRPTW_GR_DIR
    summary = build_summary(result, root, rows)

    if args.no_write:
        manifest_path = summary_path = None
    else:
        manifest_path, summary_path = write_outputs(rows, summary, args.out_dir)

    print_report(summary, manifest_path, summary_path)

    if result.errors:
        print(f"\nFAILED: {len(result.errors)} validation error(s).", file=sys.stderr)
        return 1
    if args.fail_on_warning and result.warnings:
        print(
            f"\nFAILED: {len(result.warnings)} validation warning(s) with --fail-on-warning.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
