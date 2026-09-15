"""Join frozen corpus routes to parent-level splits. Train-only by default."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from data.parser import parse_instance
from data.paths import RAW_EVRPTW_GR_DIR, ROUTES_DIR, SPLITS_DIR
from routing.fixed_route import FrozenRoute
from routing.serialize import read_jsonl


def load_split_payload(split: str, splits_dir: Path | None = None) -> dict:
    path = (Path(splits_dir) if splits_dir else SPLITS_DIR) / f"{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_split_routes(
    split: str,
    *,
    corpus_path: Path | None = None,
    splits_dir: Path | None = None,
    network_group: str | None = None,
    max_customers: int | None = None,
) -> List[FrozenRoute]:
    payload = load_split_payload(split, splits_dir)
    allowed = set(payload["instance_ids"])
    routes = read_jsonl(corpus_path or (ROUTES_DIR / "corpus.jsonl"))
    selected = [route for route in routes if route.raw_instance_id in allowed]
    if network_group is not None:
        selected = [route for route in selected if route.network_group == network_group]
    if max_customers is not None:
        selected = [route for route in selected if route.n_customers <= int(max_customers)]
    return selected


def parse_route_instance(route: FrozenRoute, dataset_root=None):
    root = Path(dataset_root) if dataset_root is not None else RAW_EVRPTW_GR_DIR
    return parse_instance(root / route.relative_path)
