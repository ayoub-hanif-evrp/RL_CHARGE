"""Cluster-aware statistics. Routes that share a parent are not independent."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def cluster_means(rows: Sequence[dict], value_key: str, cluster_key: str = "base_instance") -> Dict[str, float]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get(value_key) is None:
            continue
        cluster = row.get(cluster_key) or row.get("route_id") or row.get("method") or "unknown"
        buckets[str(cluster)].append(float(row[value_key]))
    return {key: float(np.mean(vals)) for key, vals in buckets.items()}


def cluster_bootstrap_ci(
    rows: Sequence[dict],
    value_key: str,
    cluster_key: str = "base_instance",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    clusters = cluster_means(rows, value_key, cluster_key)
    names = list(clusters)
    values = np.array([clusters[name] for name in names], dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n_clusters": 0}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(values, size=values.size, replace=True)
        boots.append(float(draw.mean()))
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "lo": lo,
        "hi": hi,
        "n_clusters": int(values.size),
        "n_rows": len(rows),
    }


def paired_parent_diff(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    value_key: str,
    cluster_key: str = "base_instance",
) -> Dict[str, float]:
    a = cluster_means(rows_a, value_key, cluster_key)
    b = cluster_means(rows_b, value_key, cluster_key)
    keys = sorted(set(a) & set(b))
    return {key: a[key] - b[key] for key in keys}


def permutation_pvalue(diffs: Sequence[float], n_perm: int = 2000, seed: int = 0) -> float:
    arr = np.asarray(list(diffs), dtype=np.float64)
    if arr.size == 0:
        return 1.0
    if arr.size <= 20:
        return exact_sign_flip_pvalue(arr)
    observed = abs(arr.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=arr.size)
        if abs((arr * signs).mean()) >= observed - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


def exact_sign_flip_pvalue(diffs: Sequence[float]) -> float:
    """Two-sided sign-flip p-value by enumerating all 2^n assignments. n <= 20."""
    arr = np.asarray(list(diffs), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return 1.0
    if n > 20:
        raise ValueError("exact sign-flip enumeration is only for n <= 20")
    observed = abs(float(arr.mean()))
    total = 1 << n
    count = 0
    for mask in range(total):
        signed_sum = 0.0
        for i in range(n):
            signed_sum += -arr[i] if (mask >> i) & 1 else arr[i]
        if abs(signed_sum / n) >= observed - 1e-15:
            count += 1
    return count / total


def holm(pvalues: Sequence[Tuple[str, float]]) -> List[Tuple[str, float, float]]:
    ordered = sorted(pvalues, key=lambda item: item[1])
    m = len(ordered)
    adjusted = []
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, p * (m - i))
        running = max(running, adj)
        adjusted.append((name, p, running))
    by_name = {name: (p, adj) for name, p, adj in adjusted}
    return [(name, by_name[name][0], by_name[name][1]) for name, _ in pvalues]


def summarize_method(rows: Sequence[dict], value_key: str) -> dict:
    usable = [row for row in rows if row.get(value_key) is not None]
    values = np.array([float(row[value_key]) for row in usable], dtype=np.float64)
    ci = cluster_bootstrap_ci(usable, value_key)
    return {
        "n": len(usable),
        "mean": float(values.mean()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "cluster_mean": ci["mean"],
        "ci95_lo": ci["lo"],
        "ci95_hi": ci["hi"],
        "n_clusters": ci["n_clusters"],
    }


def per_seed_metrics(rows: Sequence[dict], value_key: str) -> Dict[int, dict]:
    by_seed: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get(value_key) is None:
            continue
        by_seed[int(row.get("seed", 0))].append(row)
    return {seed: summarize_method(group, value_key) for seed, group in sorted(by_seed.items())}


def mean_sd_across_training_seeds(rows: Sequence[dict], value_key: str) -> dict:
    per_seed = per_seed_metrics(rows, value_key)
    means = [stats["mean"] for stats in per_seed.values() if stats.get("mean") is not None]
    arr = np.asarray(means, dtype=np.float64)
    return {
        "n_seeds": len(arr),
        "mean_across_seeds": float(arr.mean()) if arr.size else None,
        "sd_across_seeds": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "per_seed": {str(seed): stats for seed, stats in per_seed.items()},
        "resampling_unit": "training_seed_then_base_instance",
    }


def hierarchical_bootstrap_ci(
    rows: Sequence[dict],
    value_key: str,
    cluster_key: str = "base_instance",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap by drawing training seeds, then parents within each seed."""
    nested: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get(value_key) is None:
            continue
        train_seed = int(row.get("seed", 0))
        parent = str(row.get(cluster_key) or row.get("route_id") or "unknown")
        nested[train_seed][parent].append(float(row[value_key]))
    seed_parent_mean = {
        train_seed: {parent: float(np.mean(vals)) for parent, vals in parents.items()}
        for train_seed, parents in nested.items()
    }
    seeds = list(seed_parent_mean)
    if not seeds:
        return {"mean": None, "lo": None, "hi": None, "n_seeds": 0, "n_clusters": 0}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        drawn_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        parent_vals = []
        for train_seed in drawn_seeds:
            parents = list(seed_parent_mean[int(train_seed)])
            if not parents:
                continue
            drawn_parents = rng.choice(parents, size=len(parents), replace=True)
            parent_vals.extend(seed_parent_mean[int(train_seed)][str(p)] for p in drawn_parents)
        if parent_vals:
            boots.append(float(np.mean(parent_vals)))
    boots.sort()
    point_vals = []
    for parents in seed_parent_mean.values():
        point_vals.extend(parents.values())
    point = float(np.mean(point_vals)) if point_vals else None
    lo = boots[int(alpha / 2 * len(boots))] if boots else None
    hi = boots[int((1 - alpha / 2) * len(boots))] if boots else None
    return {
        "mean": point,
        "lo": lo,
        "hi": hi,
        "n_seeds": len(seeds),
        "n_clusters": int(sum(len(v) for v in seed_parent_mean.values()) / max(len(seeds), 1)),
        "n_rows": len(rows),
        "resampling_unit": "training_seed_then_base_instance",
        "sd_across_seeds": mean_sd_across_training_seeds(rows, value_key)["sd_across_seeds"],
    }


def parent_mean_over_seeds(
    rows: Sequence[dict],
    value_key: str,
    cluster_key: str = "base_instance",
) -> Dict[str, float]:
    """Mean over training seeds of the per-parent metric (learned methods)."""
    nested: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get(value_key) is None:
            continue
        parent = str(row.get(cluster_key) or row.get("route_id") or "unknown")
        nested[parent][int(row.get("seed", 0))].append(float(row[value_key]))
    out = {}
    for parent, by_seed in nested.items():
        seed_means = [float(np.mean(vals)) for vals in by_seed.values()]
        out[parent] = float(np.mean(seed_means))
    return out


def paired_parent_diff_hierarchical(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    value_key: str,
    cluster_key: str = "base_instance",
    *,
    a_mean_over_seeds: bool = True,
    b_mean_over_seeds: bool = False,
) -> Dict[str, float]:
    """Paired Hybrid vs baseline: matched parents. Learned side averages seeds."""
    a = parent_mean_over_seeds(rows_a, value_key, cluster_key) if a_mean_over_seeds else cluster_means(rows_a, value_key, cluster_key)
    b = parent_mean_over_seeds(rows_b, value_key, cluster_key) if b_mean_over_seeds else cluster_means(rows_b, value_key, cluster_key)
    keys = sorted(set(a) & set(b))
    return {key: a[key] - b[key] for key in keys}


PAPER_EXCLUDED_METHODS = {"LegacyTwoStageDDQN"}
ABLATION_DISPLAY = {
    "HybridPPO": "FULL",
    "DiscretePPO": "A1",
    "HybridPPO_FULL": "FULL",
    "HybridPPO_A1": "A1",
    "HybridPPO_A2": "A2",
    "HybridPPO_A3": "A3",
    "HybridPPO_A4": "A4",
    "HybridPPO_A5": "A5",
}
LEARNED_PAPER_METHODS = {"HybridPPO", "DiscretePPO", "AttentionPPO"}
STATELESS_PAPER_METHODS = {
    "GreedyMinimumSufficientCharge",
    "GreedyFullCharge",
    "OneStepLookahead",
}


def exclude_non_paper_methods(rows: Sequence[dict]) -> List[dict]:
    return [row for row in rows if str(row.get("method")) not in PAPER_EXCLUDED_METHODS]


def map_ablation_method(name: str) -> str:
    return ABLATION_DISPLAY.get(str(name), str(name))


def parent_balanced_mean(
    rows: Sequence[dict], value_key: str, cluster_key: str = "base_instance"
) -> float | None:
    means = cluster_means(rows, value_key, cluster_key)
    if not means:
        return None
    return float(np.mean(list(means.values())))


def route_weighted_mean(rows: Sequence[dict], value_key: str) -> float | None:
    values = [float(row[value_key]) for row in rows if row.get(value_key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def attach_feas_rate(rows: Sequence[dict]) -> List[dict]:
    return [{**row, "feas_rate": 1.0 if row.get("feasible") else 0.0} for row in rows]


def dedupe_soc_greedy(rows: Sequence[dict]) -> List[dict]:
    """Keep one GreedyMinimumSufficientCharge row per (route_id, min_soc_fraction)."""
    seen = set()
    out: List[dict] = []
    for row in rows:
        if row.get("method") != "GreedyMinimumSufficientCharge":
            out.append(row)
            continue
        key = (row.get("route_id"), row.get("min_soc_fraction"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


REQUIRED_TERRAINS = ("L", "NL", "VG")


def matched_terrain_sibling_groups(routes: Sequence | None = None) -> List[dict]:
    """Complete L/NL/VG groups that share the same frozen customer sequence."""
    if routes is None:
        from data.paths import ROUTES_DIR
        from routing.serialize import read_jsonl

        routes = read_jsonl(ROUTES_DIR / "corpus.jsonl")
    buckets: Dict[tuple, List] = defaultdict(list)
    for route in routes:
        key = (
            getattr(route, "network_group", None),
            getattr(route, "customer_folder", None),
            getattr(route, "base_instance", None),
            int(getattr(route, "vehicle_index", 0)),
            tuple(getattr(route, "customer_ids", ()) or ()),
        )
        buckets[key].append(route)
    groups = []
    for key, members in buckets.items():
        by_terrain = {}
        for route in members:
            by_terrain[str(getattr(route, "terrain_variant", ""))] = route
        if not set(REQUIRED_TERRAINS).issubset(by_terrain):
            continue
        groups.append(
            {
                "network_group": key[0],
                "customer_folder": key[1],
                "base_instance": key[2],
                "vehicle_index": key[3],
                "customer_ids": key[4],
                "route_ids": {terrain: by_terrain[terrain].route_id for terrain in REQUIRED_TERRAINS},
            }
        )
    return groups


def matched_terrain_route_ids(split: str | None = None) -> tuple[set[str], int]:
    """Route IDs in complete L/NL/VG sibling groups, optionally restricted to a split."""
    groups = matched_terrain_sibling_groups()
    if split:
        from experiments.dataset import load_split_routes

        split_ids = {route.route_id for route in load_split_routes(split)}
        groups = [group for group in groups if set(group["route_ids"].values()).issubset(split_ids)]
    ids = {route_id for group in groups for route_id in group["route_ids"].values()}
    return ids, len(groups)


def filter_matched_terrain_rows(rows: Sequence[dict], split: str | None = "test") -> List[dict]:
    ids, _ = matched_terrain_route_ids(split=split)
    return [row for row in rows if row.get("route_id") in ids]


def seed_feasibility_summary(rows: Sequence[dict]) -> dict:
    """Per-seed feasibility. Counts are seed-route episodes, not pooled unique routes."""
    tagged = attach_feas_rate(rows)
    by_seed: Dict[int, List[dict]] = defaultdict(list)
    for row in tagged:
        by_seed[int(row.get("seed", 0))].append(row)
    per_seed = {}
    for seed, group in sorted(by_seed.items()):
        n_routes = len({row.get("route_id") for row in group})
        n_feasible = sum(1 for row in group if row.get("feasible"))
        per_seed[str(seed)] = {
            "n_routes": n_routes,
            "n_feasible_routes": n_feasible,
            "n_episode_rows": len(group),
            "route_weighted_feasibility": route_weighted_mean(group, "feas_rate"),
            "parent_balanced_feasibility": parent_balanced_mean(group, "feas_rate"),
        }
    rw = [stats["route_weighted_feasibility"] for stats in per_seed.values() if stats["route_weighted_feasibility"] is not None]
    pb = [stats["parent_balanced_feasibility"] for stats in per_seed.values() if stats["parent_balanced_feasibility"] is not None]
    feas = [stats["n_feasible_routes"] for stats in per_seed.values()]
    n_routes = next((stats["n_routes"] for stats in per_seed.values()), 0)

    def _mean(values: List[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def _sd(values: List[float]) -> float | None:
        if len(values) <= 1:
            return 0.0 if values else None
        return float(np.std(values, ddof=1))

    return {
        "per_seed": per_seed,
        "n_seeds": len(per_seed),
        "n_routes_per_seed": n_routes,
        "mean_feasible_routes_per_seed": _mean([float(v) for v in feas]),
        "sd_feasible_routes_per_seed": _sd([float(v) for v in feas]),
        "mean_route_weighted_feasibility_across_seeds": _mean(rw),
        "sd_route_weighted_feasibility_across_seeds": _sd(rw),
        "mean_parent_balanced_feasibility_across_seeds": _mean(pb),
        "sd_parent_balanced_feasibility_across_seeds": _sd(pb),
        "note": "feasible-route counts are per-seed episodes (e.g. mean k/49), not a pooled 25/245 unique-route count",
    }


def logical_checkpoint_path(path: str, filename: str | None = None) -> str:
    """Replace absolute worktree paths with repository-relative checkpoint ids."""
    text = str(path).replace("\\", "/")
    marker = "checkpoints/"
    idx = text.lower().rfind(marker)
    if idx < 0:
        return text
    tail = text[idx + len(marker) :]
    parts = [part for part in tail.split("/") if part]
    if len(parts) >= 2:
        logical = f"checkpoints/{parts[0]}/{parts[1]}"
        if filename:
            return f"{logical}/{filename}"
        if len(parts) >= 3:
            return f"{logical}/{parts[2]}"
        return logical
    return f"checkpoints/{tail}"
