"""Queue runner for SynthCharge TRAIN/VAL envelope ablation (development only)."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("A_arrival_to_max", "B_energy_to_max", "C_energy_to_time")
SEEDS = (42, 43, 44, 45, 46)
LOG = ROOT / "results" / "development" / "envelope_ablation" / "queue.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    log("START envelope ablation queue (POST-HOC DEVELOPMENT / TRAIN-VAL ONLY)")
    for variant in VARIANTS:
        for seed in SEEDS:
            dest = (
                ROOT
                / "results"
                / "development"
                / "envelope_ablation"
                / variant
                / f"seed_{seed}"
                / "validation.json"
            )
            if dest.is_file():
                log(f"skip {variant} seed={seed}")
                continue
            log(f"run {variant} seed={seed}")
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/v3/development/run_envelope_ablation.py",
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                ],
                cwd=ROOT,
            )
            if proc.returncode != 0:
                log(f"FAIL {variant} seed={seed} code={proc.returncode}")
                raise SystemExit(proc.returncode)
            log(f"done {variant} seed={seed}")
    log("COMPLETE envelope ablation queue")


if __name__ == "__main__":
    main()
