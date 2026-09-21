"""Seed-42 TRAIN/VAL sanity for DiscretePPO and AttentionPPO.

Not hyperparameter tuning. TEST is forbidden. Legacy DDQN is not a paper method.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SANITY = Path(__file__).resolve().parent
METHODS = (
    ("discrete_ppo", "discrete_ppo/seed_42", "DiscretePPO"),
    ("attention_ppo", "attention_ppo/seed_42", "AttentionPPO"),
)


def _finite_losses(curves_path: Path) -> dict:
    losses = []
    if not curves_path.is_file():
        return {"n_rows": 0, "finite": False, "nonzero": False, "last_loss": None}
    for line in curves_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("loss")
        if value is None:
            value = row.get("policy_loss")
        if value is not None:
            losses.append(float(value))
    finite = bool(losses) and all(math.isfinite(v) for v in losses)
    nonzero = any(abs(v) > 0.0 for v in losses) if losses else False
    return {
        "n_rows": len(losses),
        "finite": finite,
        "nonzero": nonzero,
        "last_loss": losses[-1] if losses else None,
        "n_nan": sum(1 for v in losses if not math.isfinite(v)),
    }


def main() -> int:
    python = sys.executable
    summary = {"methods": {}}
    for method, rel, label in METHODS:
        out_dir = SANITY / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = SANITY / f"{method}_seed42.log"
        print(f"START {label}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [
                    python,
                    str(ROOT / "scripts" / "train_rl.py"),
                    "--method",
                    method,
                    "--split",
                    "train",
                    "--seeds",
                    "42",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            print(f"CRASH {label} returncode={proc.returncode}", flush=True)
            summary["methods"][label] = {"status": "crash", "returncode": proc.returncode}
            (SANITY / "sanity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print("SANITY_DONE", flush=True)
            return proc.returncode
        inspect_script = SANITY / "inspect_learned.py"
        inspections = {}
        for tag in ("best", "last"):
            ckpt = out_dir / f"{tag}.pt"
            if not ckpt.is_file():
                continue
            dest = out_dir / f"policy_behavior_{tag}.json"
            insp = subprocess.run(
                [
                    python,
                    str(inspect_script),
                    "--ckpt",
                    str(ckpt),
                    "--split",
                    "validation",
                    "--out",
                    str(dest),
                ],
                cwd=ROOT,
            )
            if insp.returncode != 0:
                print(f"CRASH inspect {label} {tag}", flush=True)
                summary["methods"][label] = {"status": "inspect_crash", "tag": tag}
                (SANITY / "sanity_summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )
                print("SANITY_DONE", flush=True)
                return insp.returncode
            inspections[tag] = json.loads(dest.read_text(encoding="utf-8"))
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        losses = _finite_losses(out_dir / "curves.jsonl")
        best_beh = inspections.get("best") or {}
        summary["methods"][label] = {
            "status": manifest.get("status"),
            "best_update": manifest.get("best_update"),
            "best_val_feasibility": manifest.get("best_val_feasibility"),
            "best_val_completion_all": manifest.get("best_val_completion_all"),
            "best_val_route_weighted_feasibility": manifest.get("best_val_route_weighted_feasibility"),
            "checkpoint_selection": manifest.get("checkpoint_selection"),
            "split_used_for_learning": manifest.get("split_used_for_learning"),
            "split_used_for_selection": manifest.get("split_used_for_selection"),
            "max_env_transitions": manifest.get("max_env_transitions"),
            "normalizer_split": (manifest.get("normalizer_provenance") or {}).get("split"),
            "normalizer_n_routes": (manifest.get("normalizer_provenance") or {}).get("n_routes"),
            "losses": losses,
            "val_feasibility": best_beh.get("feasibility"),
            "val_n_feasible": best_beh.get("n_feasible"),
            "val_n_routes": best_beh.get("n_routes"),
            "frac_continue": best_beh.get("frac_continue"),
            "frac_charge": best_beh.get("frac_charge"),
            "repeated_station_actions": best_beh.get("repeated_station_actions"),
            "loop_guard_hits": best_beh.get("loop_guard_hits"),
            "n_station_revisit": best_beh.get("n_station_revisit"),
            "nan_or_inf": best_beh.get("nan_or_inf"),
        }
        print(f"OK {label} status={manifest.get('status')}", flush=True)
    (SANITY / "sanity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("SANITY_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
