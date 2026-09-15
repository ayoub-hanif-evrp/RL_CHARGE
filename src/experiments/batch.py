"""Batch evaluation of every method on a predetermined route population. No filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from data.paths import RESULTS_DIR
from routing.fixed_route import FrozenRoute
from routing.serialize import canonical_dumps

from .dataset import parse_route_instance
from .evaluate import EpisodeResult, evaluate_policy
from .provenance import run_manifest


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical_dumps(row) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl_dicts(path: Path) -> List[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def evaluate_population(
    *,
    routes: List[FrozenRoute],
    method: str,
    policy: Callable,
    split: str,
    seed: int = 0,
    extra_context: Optional[dict] = None,
    charging_model=None,
    min_soc_fraction: Optional[float] = None,
    scenario: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> List[dict]:
    context = dict(extra_context or {})
    scenario = scenario or context.get("scenario")
    if not scenario:
        raise ValueError("scenario is required on every evaluation record")
    from experiments.isolation import require_scenario

    require_scenario(str(scenario))
    context["scenario"] = str(scenario)
    context["experiment_id"] = experiment_id or context.get("experiment_id") or context.get("run_id") or "unnamed"
    records = []
    for route in routes:
        instance = parse_route_instance(route)
        result = evaluate_policy(
            instance=instance,
            route=route,
            policy=policy,
            eval_mode=True,
            charging_model=charging_model,
            min_soc_fraction=min_soc_fraction,
        )
        records.append(
            result.to_record(
                method=method,
                split=split,
                seed=int(seed),
                base_instance=route.base_instance,
                terrain=route.terrain_variant,
                network_group=route.network_group,
                customer_distribution=route.customer_distribution,
                schedule_type=route.schedule_type,
                n_customers=route.n_customers,
                **context,
            )
        )
    return records


def dump_run(records: List[dict], run_id: str, extra_manifest: Optional[dict] = None) -> Path:
    raw = RESULTS_DIR / "raw" / f"{run_id}.jsonl"
    write_jsonl(raw, records)
    manifest = run_manifest(run_id=run_id, n_records=len(records), methods=sorted({r["method"] for r in records}))
    if extra_manifest:
        manifest.update(extra_manifest)
    run_dir = RESULTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(canonical_dumps(manifest) + "\n", encoding="utf-8")
    return raw
