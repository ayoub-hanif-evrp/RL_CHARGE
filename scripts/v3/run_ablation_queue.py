"""Queue runner for remaining B0–B3 × seeds 42–46 ablations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("B0", "B1", "B3", "B2")
SEEDS = (42, 43, 44, 45, 46)


def main() -> None:
    for variant in VARIANTS:
        for seed in SEEDS:
            dest = ROOT / "results" / "v3" / "ablation" / variant / f"seed_{seed}" / "validation.json"
            if dest.is_file():
                print(f"skip {variant} {seed}", flush=True)
                continue
            print(f"run {variant} {seed}", flush=True)
            proc = subprocess.run(
                [sys.executable, "scripts/v3/run_ablation.py", "--variant", variant, "--seed", str(seed)],
                cwd=ROOT,
            )
            if proc.returncode != 0:
                raise SystemExit(f"ablation failed: {variant} seed={seed} code={proc.returncode}")


if __name__ == "__main__":
    main()
