"""Run the full V4 reward ablation matrix with limited parallelism."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("V3_TIME", "V4_BASE", "V4_PBRS", "V4_BASE_NO_L_FAIL")
SEEDS = (42, 43, 44, 45, 46)
LOG = ROOT / "results" / "v4_reward" / "ablation" / "PARALLEL_RUN.jsonl"


def _done(variant: str, seed: int) -> bool:
    dest = ROOT / "results" / "v4_reward" / "ablation" / variant / f"seed_{seed}" / "validation.json"
    ckpt = ROOT / "checkpoints_v4" / "reward_ablation" / variant / f"seed_{seed}" / "best.pt"
    return dest.is_file() and ckpt.is_file()


def _run(variant: str, seed: int) -> dict:
    started = time.time()
    if _done(variant, seed):
        row = {"variant": variant, "seed": seed, "status": "skipped", "runtime_s": 0.0}
        return row
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "v4_reward" / "run_reward_ablation.py"),
        "--variant",
        variant,
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    row = {
        "variant": variant,
        "seed": seed,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "runtime_s": time.time() - started,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(v, s) for v in VARIANTS for s in SEEDS]
    print(f"launching {len(jobs)} jobs with {args.workers} workers", flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futs = {pool.submit(_run, v, s): (v, s) for v, s in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(
                    f"{row['variant']} seed={row['seed']} -> {row['status']} "
                    f"({row['runtime_s']:.1f}s)",
                    flush=True,
                )
    n_ok = sum(1 for v, s in jobs if _done(v, s))
    print(json.dumps({"completed_cells": n_ok, "total": len(jobs)}, indent=2), flush=True)
    if n_ok < len(jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
