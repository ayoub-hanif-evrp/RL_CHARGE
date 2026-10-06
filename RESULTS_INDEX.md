# Results index

## Current publication package (V4, standalone)

| Artifact | Path |
|----------|------|
| Figures (PNG) + tables | `results_v4_paper/` |
| Captions | `results_v4_paper/captions/CAPTIONS.md` |
| Manifest | `results_v4_paper/MANIFEST.json` |
| Builder | `scripts/paper/build_v4_results_paper.py` |
| Claims | `paper/V4_CLAIMS.md` |
| Benchmark wording | `docs/V4_BENCHMARK_WORDING.md` |

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

## V4 frozen scientific evidence

| Artifact | Path |
|----------|------|
| Reward freeze | `results/v4_reward/FINAL_REWARD_FREEZE.json` |
| Authoritative training | `results/v4_reward/final_authoritative/` |
| Reward ablation (VAL) | `results/v4_reward/ablation/`, `SUMMARY.json` |
| TEST protocol | `results/v4_test/PAPER_PROTOCOL.json` |
| Checkpoint freeze | `results/v4_test/CHECKPOINT_FREEZE.json` |
| TEST lock | `results/v4_test/TEST_LOCK.json` |
| Evaluation consumed | `results/v4_test/EVALUATION_CONSUMED.json` |
| Raw TEST rows | `results/v4_test/raw/` |

## Historical packages (untouched; not mixed into V4 paper outputs)

| Stage | Path |
|-------|------|
| V3 publication displays | `results_paper/` |
| V3 confirmatory | `results/v3_hppo/` |
| V2 | `results/v2/final/` |
| V1 | `results/final/` |

Do **not** reopen consumed TESTs. Do **not** put prior-version comparisons into `results_v4_paper/`.
