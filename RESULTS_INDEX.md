# Results index

## Current publication package (V4, standalone)

| Artifact | Path |
|----------|------|
| Figures (PNG) + tables | `results/v4/paper/` |
| Captions | `results/v4/paper/captions/CAPTIONS.md` |
| Manifest | `results/v4/paper/MANIFEST.json` |
| Builder | `scripts/paper/build_v4_results_paper.py` |
| Authoritative models | `models/v4/fa_hppo/` + `models/v4/fa_hppo/CHECKSUMS.json` |
| Claims | `paper/V4_CLAIMS.md` |
| Benchmark wording | `docs/methodology/V4_BENCHMARK_WORDING.md` |

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

## V4 frozen scientific evidence

| Artifact | Path |
|----------|------|
| Reward freeze | `results/v4/reward_development/FINAL_REWARD_FREEZE.json` |
| Authoritative training | `results/v4/reward_development/final_authoritative/` |
| Reward ablation (VAL) | `results/v4/reward_development/ablation/`, `SUMMARY.json` |
| TEST protocol | `results/v4/test/PAPER_PROTOCOL.json` |
| Checkpoint freeze | `results/v4/test/CHECKPOINT_FREEZE.json` |
| TEST lock | `results/v4/test/TEST_LOCK.json` |
| Evaluation consumed | `results/v4/test/EVALUATION_CONSUMED.json` |
| Raw TEST rows | `results/v4/test/raw/` |

## Historical packages (not mixed into V4 paper outputs)

| Stage | Path |
|-------|------|
| V3 publication displays | `results/v3/paper/` |
| V3 claims map | `results/v3/paper/claims/` |
| V3 confirmatory + development | `results/v3/` (incl. `development/`) |
| V2 | `results/v2/final/` |
| V1 | `results/v1/final/` |

Do **not** reopen consumed TESTs. Do **not** put prior-version comparisons into `results/v4/paper/`.
