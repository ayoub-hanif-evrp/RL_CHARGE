"""Preflight gate before any final V2 evaluation. Does not train or evaluate TEST."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v2"))

from final_common import (  # noqa: E402
    DATASETS,
    FINAL,
    METHOD_FREEZE_SHA,
    METHODS,
    SEEDS,
    TRAINING_EXECUTION_SHA,
    checkpoint_dir,
    git_head,
    git_porcelain,
    load_json,
    method_tree_diff,
    sha256,
)


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "missing"


def main() -> int:
    report = []
    ok = True

    def gate(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        report.append(f"{label}: {'PASS' if passed else 'FAIL'}{(' — ' + detail) if detail else ''}")

    porcelain = git_porcelain().strip()
    # Allow only untracked finalization outputs that are not yet committed; refuse
    # modifications under frozen dirs. For this gate we require clean porcelain
    # when --require-clean is passed (default for evaluation opening).
    require_clean = "--allow-dirty-tooling" not in sys.argv
    if require_clean:
        gate("working_tree_clean", not porcelain, porcelain[:200] if porcelain else "")
    else:
        gate("working_tree_clean", True, "skipped (--allow-dirty-tooling)")

    gate("method_tree_identical_to_freeze", not method_tree_diff().strip(), method_tree_diff().strip())
    freeze = load_json(FINAL / "METHOD_FREEZE.json")
    gate("method_freeze_sha_match", freeze["V2_METHOD_FREEZE_SHA"] == METHOD_FREEZE_SHA)
    mismatches = []
    for rel, digest in freeze["ppo_hyperparameters"]["config_sha256"].items():
        if sha256(ROOT / rel) != digest:
            mismatches.append(rel)
    gate("config_hashes", not mismatches, ",".join(mismatches))

    pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--tb=line"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    gate("pytest", pytest.returncode == 0, pytest.stdout.strip().splitlines()[-1] if pytest.stdout else "")
    frvcpy = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/baselines/test_frvcpy_parity.py", "-q", "--tb=short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    gate("frvcpy_parity", frvcpy.returncode == 0, frvcpy.stdout.strip().splitlines()[-1] if frvcpy.stdout else "")

    lock = subprocess.run(
        [sys.executable, "scripts/v2/write_test_lock.py", "--verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    gate("TEST_LOCK", lock.returncode == 0, lock.stdout.strip())
    audit = subprocess.run(
        [sys.executable, "scripts/v2/audit_final_data.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    gate("data_audit", audit.returncode == 0 and '"n_failures":0' in audit.stdout.replace(" ", ""))

    gate("FINAL_PROTOCOL_exists", (FINAL / "FINAL_PROTOCOL.json").is_file())
    complete = 0
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                folder = checkpoint_dir(dataset, method, seed)
                if (folder / "best.pt").is_file() and (folder / "manifest.json").is_file():
                    manifest = load_json(folder / "manifest.json")
                    if manifest.get("best_update") is not None and not manifest.get("git_dirty"):
                        complete += 1
    gate("learned_runs_20_of_20", complete == 20, f"{complete}/20")

    env = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        "pyvrp": _version("pyvrp"),
        "frvcpy": _version("frvcpy"),
        "cuda": False,
        "git_head": git_head(),
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "training_execution_sha": TRAINING_EXECUTION_SHA,
    }
    expected = freeze["packages"]
    env_ok = (
        env["python"].startswith(expected["python"])
        and env["torch"] == expected["torch"]
        and env["numpy"] == expected["numpy"]
        and env["pyvrp"] == expected["pyvrp"]
        and env["frvcpy"] == expected["frvcpy"]
    )
    gate("environment_matches_method_freeze", env_ok, json_dumps := str(env))

    print("\n".join(report))
    print("---")
    print(json_dumps)
    if not ok:
        return 1
    print("PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
