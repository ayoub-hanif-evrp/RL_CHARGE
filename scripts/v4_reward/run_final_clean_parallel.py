"""Parallel launcher for clean final V4 reward seeds (42–46)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44, 45, 46)
LOG = ROOT / "results" / "v4_reward" / "final_clean" / "PARALLEL_RUN.jsonl"


def _run(seed: int) -> dict:
    started = time.time()
    dest = ROOT / "results" / "v4_reward" / "final_clean" / "V4_BASE_NO_L_FAIL" / f"seed_{seed}" / "validation.json"
    ckpt = ROOT / "checkpoints_v4" / "final_reward" / "V4_BASE_NO_L_FAIL" / f"seed_{seed}" / "best.pt"
    if dest.is_file() and ckpt.is_file():
        row = {"seed": seed, "status": "skipped", "returncode": 0, "runtime_s": 0.0}
        return row
    cmd = [sys.executable, str(ROOT / "scripts" / "v4_reward" / "run_final_clean.py"), "--seed", str(seed)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    row = {
        "seed": seed,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "runtime_s": time.time() - started,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }
    return row


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"launching {len(SEEDS)} final_clean seeds", flush=True)
    results = []
    with LOG.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futs = {pool.submit(_run, s): s for s in SEEDS}
            for fut in as_completed(futs):
                row = fut.result()
                results.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(f"seed={row['seed']} -> {row['status']} ({row['runtime_s']:.1f}s)", flush=True)
    n_ok = sum(1 for r in results if r["returncode"] == 0)
    print(json.dumps({"n_ok": n_ok, "n": len(results)}, indent=2), flush=True)
    if n_ok < len(SEEDS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
