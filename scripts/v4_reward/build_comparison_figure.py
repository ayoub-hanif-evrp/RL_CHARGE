"""Bar comparison of V4 reward variants from SUMMARY.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "v4_reward" / "SUMMARY.json"
FIG = ROOT / "results_v4" / "figures"
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
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    x = np.arange(len(ORDER))
    feas = [100.0 * variants[v]["feas_mean"] for v in ORDER]
    feas_sd = [100.0 * variants[v]["feas_sd"] for v in ORDER]
    comp = [variants[v]["comp_mean"] for v in ORDER]
    comp_sd = [variants[v]["comp_sd"] for v in ORDER]
    axes[0].bar(x, feas, yerr=feas_sd, color=[COLORS[v] for v in ORDER], capsize=3, width=0.7, edgecolor="none")
    for i, v in enumerate(feas):
        axes[0].text(i, v + 0.35, f"{v:.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(x, ORDER, rotation=15, ha="right")
    axes[0].set_ylabel("VAL feasibility (%)")
    axes[0].set_ylim(90, 102)
    axes[0].set_title("(a) Parent-balanced VAL feasibility", loc="left")

    axes[1].bar(x, comp, yerr=comp_sd, color=[COLORS[v] for v in ORDER], capsize=3, width=0.7, edgecolor="none")
    for i, v in enumerate(comp):
        axes[1].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks(x, ORDER, rotation=15, ha="right")
    axes[1].set_ylabel("VAL completion (failure-retaining)")
    axes[1].set_ylim(3.5, 4.3)
    axes[1].set_title("(b) Parent-balanced VAL completion", loc="left")
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
