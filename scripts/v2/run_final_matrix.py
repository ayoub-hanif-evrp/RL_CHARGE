"""Document / list the frozen 20-run final training matrix.

Does not train. Training already completed at TRAINING_EXECUTION_SHA.
Refuses to launch training if any completed final run exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v2"))

from final_common import DATASETS, METHODS, SEEDS, TRAINING_EXECUTION_SHA, checkpoint_dir  # noqa: E402


def main() -> None:
    print(f"training_execution_sha={TRAINING_EXECUTION_SHA}")
    print("matrix:")
    missing = []
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                folder = checkpoint_dir(dataset, method, seed)
                ok = (folder / "best.pt").is_file() and (folder / "manifest.json").is_file()
                print(f"  {dataset} {method} seed={seed}: {'OK' if ok else 'MISSING'}")
                if not ok:
                    missing.append(f"{dataset}/{method}/seed_{seed}")
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    print("20/20 present. Do not retrain. Proceed to freeze_final_checkpoints.py")


if __name__ == "__main__":
    main()
