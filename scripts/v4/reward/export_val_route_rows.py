"""Reevaluate VAL (90 routes) for all V4 ablation variants × seeds. No TEST. No retrain."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from domain.load_convention import LoadConvention  # noqa: E402
from experiments.dataset import parse_route_instance  # noqa: E402
from experiments.evaluate import evaluate_policy  # noqa: E402
from rl.ablation import AblationConfig  # noqa: E402
from rl.checkpoint import load_hybrid_actor, sha256_file  # noqa: E402
from rl.train_loop import assert_learning_split  # noqa: E402
from routing.serialize import read_jsonl  # noqa: E402

VARIANTS = ("V3_TIME", "V4_BASE", "V4_PBRS", "V4_BASE_NO_L_FAIL")
SEEDS = (42, 43, 44, 45, 46)
CKPT_ROOT = ROOT / "models" / "v4" / "reward_ablation"
OUT = ROOT / "results" / "v4" / "reward_development" / "val_route_rows"


def _sha_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert_learning_split("validation")
    corpus = ROOT / "data" / "routes_v2" / "synthcharge_final" / "validation" / "corpus.jsonl"
    routes = read_jsonl(corpus)
    if len(routes) != 90:
        raise SystemExit(f"expected 90 VAL routes, got {len(routes)}")
    OUT.mkdir(parents=True, exist_ok=True)
    ablation = AblationConfig(name="FULL", time_aware=True, soc_interval="continuation_to_max")
    n_rows = 0
    for variant in VARIANTS:
        out_path = OUT / f"{variant}_val_rows.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for seed in SEEDS:
                ckpt = CKPT_ROOT / variant / f"seed_{seed}" / "best.pt"
                if not ckpt.is_file():
                    raise SystemExit(f"missing checkpoint {ckpt}")
                ckpt_sha = sha256_file(ckpt)
                actor = load_hybrid_actor(ckpt)
                actor.ablation = ablation
                for route in routes:
                    instance = parse_route_instance(route)
                    result = evaluate_policy(
                        instance=instance,
                        route=route,
                        policy=actor,
                        load_convention=LoadConvention.OFFICIAL_REFERENCE_PICKUP,
                        profile_name=route.physics_profile or "synthcharge_linear",
                        eval_mode=True,
                    )
                    row = result.to_record(
                        variant=variant,
                        seed=int(seed),
                        base_instance=route.base_instance,
                        checkpoint_sha256=ckpt_sha,
                        checkpoint_relpath=str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                        split="validation",
                    )
                    handle.write(json.dumps(row) + "\n")
                    n_rows += 1
        print(f"wrote {out_path}", flush=True)
    meta = {
        "n_rows": n_rows,
        "n_routes": len(routes),
        "n_variants": len(VARIANTS),
        "n_seeds": len(SEEDS),
        "val_corpus": str(corpus.relative_to(ROOT)).replace("\\", "/"),
        "val_corpus_sha256": _sha_text(corpus),
        "confirmatory_test_consumed": False,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
