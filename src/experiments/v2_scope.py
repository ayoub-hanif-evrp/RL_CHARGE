"""V2 development scope. Consumed V1 TEST parents are excluded."""

from __future__ import annotations

TRAIN_PARENTS = (
    "c102",
    "c103",
    "c104",
    "c106",
    "c206",
    "c208",
    "r103",
    "r104",
    "r105",
    "r107",
    "r202",
    "r203",
    "rc101",
    "rc103",
    "rc104",
    "rc105",
    "rc106",
    "rc201",
    "rc202",
    "rc204",
)
VAL_PARENTS = ("c108", "c202", "r102", "r209", "rc108", "rc205")
CONSUMED_TEST_PARENTS = ("c101", "c205", "r110", "r201", "rc102", "rc208")

TRAIN_PARENT_SET = frozenset(TRAIN_PARENTS)
VAL_PARENT_SET = frozenset(VAL_PARENTS)
CONSUMED_TEST_PARENT_SET = frozenset(CONSUMED_TEST_PARENTS)
DEV_PARENT_SET = TRAIN_PARENT_SET | VAL_PARENT_SET


def split_for_parent(base_instance: str) -> str:
    if base_instance in CONSUMED_TEST_PARENT_SET:
        raise ValueError(f"consumed V1 TEST parent is not a V2 development parent: {base_instance}")
    if base_instance in TRAIN_PARENT_SET:
        return "train"
    if base_instance in VAL_PARENT_SET:
        return "validation"
    raise ValueError(f"parent is outside the V2 development assignment: {base_instance}")
