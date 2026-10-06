"""Quick post-migration hash checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


checks = {
    "raw": lf(ROOT / "results/v4/test/raw/synthcharge_v4_test.jsonl"),
    "lock": lf(ROOT / "results/v4/test/TEST_LOCK.json"),
    "consumed": sha(ROOT / "results/v4/test/EVALUATION_CONSUMED.json"),
    "reward": sha(ROOT / "results/v4/reward_development/FINAL_REWARD_FREEZE.json"),
}
exp = {
    "raw": "431014b397eba9191b09312f83ccb7523641e353f3d37ae9c6d57bb2a31b00dc",
    "lock": "c0055ba29beaa9b36590ef7e84cc9a60735f52adabaf7ca51dfdde5b89444d58",
    "consumed": "3cdcef0bb339f63a21466d5dd26d27e8c0d518820ed402de21eec78a02bc54b7",
    "reward": "c6a520e1a588445c880056777c8298de5fbbf1ae1cdfeac3865b104e036dc82f",
}
print({k: checks[k] == exp[k] for k in checks})
freeze = json.loads((ROOT / "results/v4/test/CHECKPOINT_FREEZE.json").read_text(encoding="utf-8"))
ok = all(
    sha(ROOT / "models/v4/fa_hppo" / f"seed_{e['seed']}" / "best.pt") == e["checkpoint_sha256"]
    for e in freeze["checkpoints"]
    if e.get("method") == "V4_FA_HPPO"
)
print("fa_hppo", ok)
print("fig03", sha(ROOT / "results/v4/paper/figures/fig03_soc_envelope.png"))
print("files", sum(1 for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts))
print("top", sorted(p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")))
