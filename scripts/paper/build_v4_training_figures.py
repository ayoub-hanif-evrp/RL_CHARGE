"""Build V4 training figures from saved curves.jsonl (no retraining).

Aggregation rules:
- Never extrapolate a seed beyond its last observed update.
- Aggregate only on the common update range shared by all loaded seeds.
- Uncertainty: Student-t 95% CI across seeds (n=5 => t_{0.975,4}).

Default final figures use V4_BASE_NO_L_FAIL (selected development reward).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "results" / "v4_reward" / "ablation"
FINAL = ROOT / "results" / "v4_reward" / "final_clean"
FIG = ROOT / "results_v4" / "figures"
DPI = 600
SEEDS = (42, 43, 44, 45, 46)
DEFAULT_VARIANT = "V4_BASE_NO_L_FAIL"
# two-sided 95% Student-t critical value for df=4
T_CRIT_5 = float(stats.t.ppf(0.975, df=4))


def _load_curves(variant: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    roots = (FINAL, ABLATION)
    for seed in SEEDS:
        path = None
        for base in roots:
            candidate = base / variant / f"seed_{seed}" / "curves.jsonl"
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            out[seed] = rows
    return out


def _series(curves: dict[int, list[dict]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± Student-t 95% CI on the intersection of seed update supports."""
    per_seed = []
    for rows in curves.values():
        by_u = {int(r["update"]): r.get(key) for r in rows if r.get(key) is not None}
        if by_u:
            per_seed.append(by_u)
    empty = (np.asarray([]), np.asarray([]), (np.asarray([]), np.asarray([])))
    if not per_seed:
        return empty
    # common updates present in EVERY seed (no forward-fill / extrapolation)
    common = set(per_seed[0].keys())
    for by_u in per_seed[1:]:
        common &= set(by_u.keys())
    grids = sorted(common)
    if not grids:
        return empty
    mat = np.asarray([[float(by_u[u]) for u in grids] for by_u in per_seed], dtype=float)
    mu = mat.mean(axis=0)
    n = mat.shape[0]
    if n > 1:
        se = mat.std(axis=0, ddof=1) / np.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        lo, hi = mu - tcrit * se, mu + tcrit * se
    else:
        lo, hi = mu.copy(), mu.copy()
    return np.asarray(grids), mu, (lo, hi)


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_reward_evolution(variant: str = DEFAULT_VARIANT) -> Path | None:
    curves = _load_curves(variant)
    if not curves:
        return None
    # Older ablation logs used base_episode_return_mean (= normalized objective for unshaped rewards).
    c_train = 10.0
    for rows in curves.values():
        if rows and rows[0].get("c_train") is not None:
            c_train = float(rows[0]["c_train"])
            break
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    # (key, ylabel, title, scale_if_fallback)
    panels = (
        (
            axes[0],
            "objective_episode_return_mean",
            "Mean objective episode return (−T / −H)",
            "(a) Objective return",
            c_train,
        ),
        (
            axes[1],
            "normalized_base_episode_return_mean",
            "Mean normalized base return",
            "(b) Normalized base return",
            1.0,
        ),
    )
    for ax, key, ylabel, title, fallback_scale in panels:
        xs, mu, (lo, hi) = _series(curves, key)
        if xs.size == 0:
            xs, mu, (lo, hi) = _series(curves, "base_episode_return_mean")
            if xs.size:
                mu = mu * fallback_scale
                lo = lo * fallback_scale
                hi = hi * fallback_scale
        if xs.size == 0:
            continue
        ax.plot(xs, mu, color="#0072B2", lw=1.8)
        ax.fill_between(xs, lo, hi, color="#0072B2", alpha=0.18)
        ax.set_xlabel("PPO update")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return _save(fig, "fig_v4_reward_evolution.png")


def fig_loss_evolution(variant: str = DEFAULT_VARIANT) -> Path | None:
    curves = _load_curves(variant)
    if not curves:
        return None
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for ax, key, ylabel, title, logy in (
        (axes[0], "policy_loss", r"Policy loss $L_\pi$", "(a) Policy loss", False),
        (axes[1], "value_loss", r"Value loss $L_V$", "(b) Value loss", True),
    ):
        xs, mu, (lo, hi) = _series(curves, key)
        if xs.size == 0:
            continue
        ax.plot(xs, mu, color="#0072B2", lw=1.8)
        ax.fill_between(xs, np.maximum(lo, 1e-12), np.maximum(hi, 1e-12), color="#0072B2", alpha=0.18)
        if logy and np.all(mu > 0):
            ax.set_yscale("log")
        ax.set_xlabel("PPO update")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return _save(fig, "fig_v4_loss_evolution.png")


def fig_validation_learning(variant: str = DEFAULT_VARIANT) -> Path | None:
    curves = _load_curves(variant)
    if not curves:
        return None
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs, mu, (lo, hi) = _series(curves, "val_parent_balanced_feasibility")
    if xs.size == 0:
        return None
    ax.plot(xs, 100.0 * mu, color="#0072B2", lw=1.8, label="VAL (parent-balanced)")
    ax.fill_between(xs, 100.0 * lo, 100.0 * hi, color="#0072B2", alpha=0.18)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("VALIDATION feasibility (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, "fig_v4_validation_learning.png")


def fig_gradient_norm(variant: str = DEFAULT_VARIANT) -> Path | None:
    curves = _load_curves(variant)
    if not curves:
        return None
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    xs, mu, (lo, hi) = _series(curves, "grad_norm_preclip")
    if xs.size == 0:
        return None
    ax.plot(xs, mu, color="#D55E00", lw=1.8)
    ax.fill_between(xs, np.maximum(lo, 1e-12), np.maximum(hi, 1e-12), color="#D55E00", alpha=0.18)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Pre-clip gradient norm")
    if np.all(mu > 0):
        ax.set_yscale("log")
    fig.tight_layout()
    return _save(fig, "fig_v4_gradient_norm.png")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    variant = DEFAULT_VARIANT
    if len(sys.argv) > 1:
        variant = sys.argv[1]
    written = []
    for fn in (fig_reward_evolution, fig_loss_evolution, fig_validation_learning, fig_gradient_norm):
        path = fn(variant)
        if path is not None:
            written.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha(path)})
    manifest = {
        "variant": variant,
        "aggregation": "common_update_intersection_no_extrapolation",
        "uncertainty": "student_t_95ci_across_seeds",
        "n_seeds_found": len(_load_curves(variant)),
        "figures": written,
        "note": "TRAIN/VAL development figures only. Not V3/V4 TEST.",
    }
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not written:
        print("no curves found yet; figures not written", file=sys.stderr)


if __name__ == "__main__":
    main()
