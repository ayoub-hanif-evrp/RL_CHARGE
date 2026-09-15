"""Canonical JSON serialization for frozen routes. Round-trips must be byte-stable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .fixed_route import FrozenRoute

SEPARATORS = (",", ":")


def canonical_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=SEPARATORS, ensure_ascii=True)


def dumps_route(route: FrozenRoute) -> str:
    return canonical_dumps(route.to_dict())


def loads_route(text: str) -> FrozenRoute:
    return FrozenRoute.from_dict(json.loads(text))


def write_jsonl(path: Path, routes: Iterable[FrozenRoute]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [dumps_route(route) for route in routes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl(path: Path) -> List[FrozenRoute]:
    text = Path(path).read_text(encoding="utf-8")
    routes = []
    for line in text.splitlines():
        if line.strip():
            routes.append(loads_route(line))
    return routes
