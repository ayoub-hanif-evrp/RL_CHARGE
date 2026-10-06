"""Paired/per-seed VAL comparison of V4 reward variants (no truncated axes)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / "results" / "v4" / "reward_development" / "SUMMARY.json"
FIG = ROOT / "results" / "v4" / "reward_development" / "figures"
ORDER = ("V3_TIME", "V4_BASE", "V4_PBRS", "V4_BASE_NO_L_FAIL")
COLORS = {
    "V3_TIME": "#999999",
    "V4_BASE": "#56B4E9",
    "V4_PBRS": "#0072B2",
    "V4_BASE_NO_L_FAIL": "#E69F00",
}


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    variants = payload["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    x = np.arange(len(ORDER))

    # Panel A: feasibility with seed points
    ax = axes[0]
    means = []
    for i, v in enumerate(ORDER):
        seeds = sorted(variants[v]["feas_per_seed"].items(), key=lambda kv: int(kv[0]))
        vals = [100.0 * float(val) for _, val in seeds]
        means.append(float(np.mean(vals)))
        ax.scatter(np.full(len(vals), i), vals, color=COLORS[v], s=28, zorder=3, edgecolors="white", linewidths=0.4)
        ax.errorbar(i, means[-1], yerr=100.0 * float(variants[v]["feas_sd"]), fmt="none", ecolor="#333333", capsize=3, elinewidth=1.0, zorder=2)
        ax.plot(i, means[-1], "D", color=COLORS[v], markersize=6, markeredgecolor="black", markeredgewidth=0.4, zorder=4)
    ax.set_xticks(x, ORDER, rotation=15, ha="right")
    ax.set_ylabel("VAL feasibility (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Parent-balanced VAL feasibility", loc="left")
    ax.axhline(100, color="#DDDDDD", lw=0.8, zorder=0)

    # Panel B: completion with seed points
    ax = axes[1]
    means = []
    for i, v in enumerate(ORDER):
        seeds = sorted(variants[v]["comp_per_seed"].items(), key=lambda kv: int(kv[0]))
        vals = [float(val) for _, val in seeds]
        means.append(float(np.mean(vals)))
        ax.scatter(np.full(len(vals), i), vals, color=COLORS[v], s=28, zorder=3, edgecolors="white", linewidths=0.4)
        ax.errorbar(i, means[-1], yerr=float(variants[v]["comp_sd"]), fmt="none", ecolor="#333333", capsize=3, elinewidth=1.0, zorder=2)
        ax.plot(i, means[-1], "D", color=COLORS[v], markersize=6, markeredgecolor="black", markeredgewidth=0.4, zorder=4)
    ax.set_xticks(x, ORDER, rotation=15, ha="right")
    ax.set_ylabel("Failure-retaining VAL completion")
    ymin = min(min(float(x) for x in variants[v]["comp_per_seed"].values()) for v in ORDER)
    ymax = max(max(float(x) for x in variants[v]["comp_per_seed"].values()) for v in ORDER)
    pad = 0.15 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_title("(b) Parent-balanced VAL completion", loc="left")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig_v4_reward_ablation_val.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
