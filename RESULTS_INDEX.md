# Results index

## Paper confirmatory (V3-HPPO)

| Artifact | Path |
|----------|------|
| Protocol (pre-TEST) | `results/v3_hppo/PAPER_PROTOCOL.json` |
| Checkpoint freeze | `results/v3_hppo/CHECKPOINT_FREEZE.json` |
| TEST lock | `results/v3_hppo/TEST_LOCK.json` |
| Evaluation consumed | `results/v3_hppo/EVALUATION_CONSUMED.json` |
| Environment | `results/v3_hppo/ENVIRONMENT.json` |
| Raw TEST rows | `results/v3_hppo/raw/` |
| Statistics | `results/v3_hppo/statistics/` |
| Ablation (development) | `results/v3_hppo/ablation/` |
| Paper figures | `paper/figures/` |
| Paper tables | `paper/tables/` |
| Claims discipline | `paper/CLAIMS.md` |
| Results map | `paper/RESULTS_MAP.md` |

## Historical V2 (preserved; includes DiscretePPO)

| Artifact | Path |
|----------|------|
| Protocol / freezes | `results/v2/final/FINAL_PROTOCOL.json`, `METHOD_FREEZE.json`, `CHECKPOINT_FREEZE.json`, `TEST_LOCK.json`, `EVALUATION_CONSUMED.json` |
| Raw / stats / figures | `results/v2/final/raw/`, `statistics/`, `tables/`, `figures/` |

## Historical V1

| Artifact | Path |
|----------|------|
| Final package | `results/final/` |

## Regenerate paper displays only

```bash
python scripts/paper/build_paper_artifacts.py --verify
```

Do **not** re-open the consumed TEST.
