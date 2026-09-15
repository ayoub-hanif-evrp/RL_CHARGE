"""Parent-level Solomon-Schneider train/validation/test splits."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from data.models import EVRPTWGRInstance, NetworkGroup
from data.parser import parse_instance
from data.paths import SPLITS_DIR, iter_instance_files
from routing.serialize import canonical_dumps


SPLIT_SEED = 42
TRAIN, VALIDATION, TEST = "train", "validation", "test"
DOCUMENTED_LARGE_SWAP = ("c108", "c101")


@dataclass(frozen=True)
class ParentRecord:
    base_instance: str
    customer_distribution: str
    schedule_type: int
    instance_ids: Tuple[str, ...]
    relative_paths: Tuple[str, ...]
    has_large: bool
    n_instances: int


@dataclass(frozen=True)
class SplitAssignment:
    train: Tuple[str, ...]
    validation: Tuple[str, ...]
    test: Tuple[str, ...]
    parents: Mapping[str, ParentRecord]


def load_parents(dataset_root=None) -> Dict[str, ParentRecord]:
    files = list(iter_instance_files(dataset_root))
    grouped: Dict[str, List[EVRPTWGRInstance]] = defaultdict(list)
    for path in files:
        instance = parse_instance(path)
        grouped[instance.metadata.base_instance].append(instance)
    parents = {}
    for base, instances in grouped.items():
        instances = sorted(instances, key=lambda inst: inst.metadata.instance_id)
        parents[base] = ParentRecord(
            base_instance=base,
            customer_distribution=instances[0].metadata.customer_distribution,
            schedule_type=instances[0].metadata.schedule_type,
            instance_ids=tuple(inst.metadata.instance_id for inst in instances),
            relative_paths=tuple(inst.metadata.relative_path for inst in instances),
            has_large=any(
                inst.metadata.network_group is NetworkGroup.LARGE for inst in instances
            ),
            n_instances=len(instances),
        )
    return parents


def assign_splits(
    parents: Mapping[str, ParentRecord],
    *,
    seed: int = SPLIT_SEED,
) -> SplitAssignment:
    strata: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for base, record in parents.items():
        strata[(record.customer_distribution, record.schedule_type)].append(base)
    train: List[str] = []
    validation: List[str] = []
    test: List[str] = []
    rng = random.Random(seed)
    for key in sorted(strata):
        names = sorted(strata[key])
        rng.shuffle(names)
        test.append(names[-1])
        validation.append(names[-2])
        train.extend(names[:-2])
    train, validation, test = _repair_large(parents, train, validation, test)
    return SplitAssignment(
        train=tuple(sorted(train)),
        validation=tuple(sorted(validation)),
        test=tuple(sorted(test)),
        parents=parents,
    )


def _has_large_files(
    parents: Mapping[str, ParentRecord], names: Sequence[str]
) -> bool:
    return any(parents[name].has_large for name in names)


def _repair_large(
    parents: Mapping[str, ParentRecord],
    train: List[str],
    validation: List[str],
    test: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    if _has_large_files(parents, test):
        return train, validation, test
    val_large = [
        name
        for name in validation
        if parents[name].has_large
    ]
    if not val_large:
        return train, validation, test
    documented_val, documented_test = DOCUMENTED_LARGE_SWAP
    if documented_val in test and documented_test in val_large:
        test_parent, val_parent = documented_val, documented_test
    else:
        test_parent = test[0]
        same = [
            name
            for name in val_large
            if (
                parents[name].customer_distribution
                == parents[test_parent].customer_distribution
                and parents[name].schedule_type == parents[test_parent].schedule_type
            )
        ]
        if not same:
            # Fall back to any same-stratum pair.
            for candidate in test:
                same = [
                    name
                    for name in val_large
                    if (
                        parents[name].customer_distribution
                        == parents[candidate].customer_distribution
                        and parents[name].schedule_type == parents[candidate].schedule_type
                    )
                ]
                if same:
                    test_parent = candidate
                    break
        val_parent = same[0]
    test = [val_parent if name == test_parent else name for name in test]
    validation = [test_parent if name == val_parent else name for name in validation]
    return train, validation, test


def split_of_parent(assignment: SplitAssignment, base_instance: str) -> str:
    if base_instance in assignment.train:
        return TRAIN
    if base_instance in assignment.validation:
        return VALIDATION
    if base_instance in assignment.test:
        return TEST
    raise KeyError(base_instance)


def write_splits(assignment: SplitAssignment, out_dir: Path | None = None) -> Dict[str, Path]:
    out_dir = Path(out_dir) if out_dir is not None else SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    split_map = {
        TRAIN: assignment.train,
        VALIDATION: assignment.validation,
        TEST: assignment.test,
    }
    rows = []
    for split_name, parents in split_map.items():
        instance_ids = []
        relative_paths = []
        for base in parents:
            record = assignment.parents[base]
            instance_ids.extend(record.instance_ids)
            relative_paths.extend(record.relative_paths)
            for instance_id, rel in zip(record.instance_ids, record.relative_paths):
                rows.append(
                    {
                        "split": split_name,
                        "base_instance": base,
                        "instance_id": instance_id,
                        "relative_path": rel,
                        "customer_distribution": record.customer_distribution,
                        "schedule_type": record.schedule_type,
                        "has_large": record.has_large,
                    }
                )
        payload = {
            "split": split_name,
            "seed": SPLIT_SEED,
            "n_parents": len(parents),
            "n_instances": len(instance_ids),
            "parents": list(parents),
            "instance_ids": instance_ids,
            "relative_paths": relative_paths,
        }
        path = out_dir / f"{split_name}.json"
        path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
        written[split_name] = path
    manifest = out_dir / "split_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "base_instance",
                "instance_id",
                "relative_path",
                "customer_distribution",
                "schedule_type",
                "has_large",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["split"], item["base_instance"], item["instance_id"])):
            writer.writerow(row)
    metadata = {
        "seed": SPLIT_SEED,
        "unit": "solomon_schneider_parent_base_instance",
        "ratios": {"train": 20, "validation": 6, "test": 6},
        "algorithm": (
            "stratify by (customer_distribution, schedule_type); sort names; "
            "Random(42).shuffle; last->test, second-last->val, rest->train; "
            "if TEST has zero Large files, swap the TEST parent with a same-stratum "
            "Large parent from VAL (this dataset: c108 <-> c101)"
        ),
        "n_parents": {
            TRAIN: len(assignment.train),
            VALIDATION: len(assignment.validation),
            TEST: len(assignment.test),
        },
        "n_instances": {
            TRAIN: sum(assignment.parents[p].n_instances for p in assignment.train),
            VALIDATION: sum(assignment.parents[p].n_instances for p in assignment.validation),
            TEST: sum(assignment.parents[p].n_instances for p in assignment.test),
        },
        "large_parents": {
            TRAIN: sorted(
                p for p in assignment.train if assignment.parents[p].has_large
            ),
            VALIDATION: sorted(
                p for p in assignment.validation if assignment.parents[p].has_large
            ),
            TEST: sorted(
                p for p in assignment.test if assignment.parents[p].has_large
            ),
        },
        "documented_large_swap": list(DOCUMENTED_LARGE_SWAP),
        "leakage": "no base_instance, instance_id, or route_id may be shared across splits",
    }
    (out_dir / "split_metadata.json").write_text(
        canonical_dumps(metadata) + "\n", encoding="utf-8"
    )
    written["manifest"] = manifest
    written["metadata"] = out_dir / "split_metadata.json"
    return written


def assert_no_leakage(assignment: SplitAssignment, routes=None) -> None:
    sets = {
        TRAIN: set(assignment.train),
        VALIDATION: set(assignment.validation),
        TEST: set(assignment.test),
    }
    assert not (sets[TRAIN] & sets[VALIDATION])
    assert not (sets[TRAIN] & sets[TEST])
    assert not (sets[VALIDATION] & sets[TEST])
    instances = {
        name: {
            iid
            for parent in parents
            for iid in assignment.parents[parent].instance_ids
        }
        for name, parents in (
            (TRAIN, assignment.train),
            (VALIDATION, assignment.validation),
            (TEST, assignment.test),
        )
    }
    assert not (instances[TRAIN] & instances[VALIDATION])
    assert not (instances[TRAIN] & instances[TEST])
    assert not (instances[VALIDATION] & instances[TEST])
    if routes is not None:
        route_ids = {TRAIN: set(), VALIDATION: set(), TEST: set()}
        parent_of = {}
        for parent, record in assignment.parents.items():
            split = split_of_parent(assignment, parent)
            for iid in record.instance_ids:
                parent_of[iid] = split
        for route in routes:
            split = parent_of[route.raw_instance_id]
            if route.route_id in route_ids[TRAIN] | route_ids[VALIDATION] | route_ids[TEST]:
                raise AssertionError(f"duplicate route_id {route.route_id}")
            route_ids[split].add(route.route_id)
        assert not (route_ids[TRAIN] & route_ids[VALIDATION])
        assert not (route_ids[TRAIN] & route_ids[TEST])
        assert not (route_ids[VALIDATION] & route_ids[TEST])
