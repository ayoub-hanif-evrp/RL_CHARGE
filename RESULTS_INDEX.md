# Results index

## Publication outputs (single source of truth)

| Artifact | Path |
|----------|------|
| Figures (PNG) + tables | `results_paper/` |
| Manifest | `results_paper/MANIFEST.json` |
| Builder | `scripts/paper/build_results_paper.py` |

## Paper confirmatory (V3-HPPO)

| Artifact | Path |
|----------|------|
| Protocol (pre-TEST) | `results/v3_hppo/PAPER_PROTOCOL.json` |
| Checkpoint freeze | `results/v3_hppo/CHECKPOINT_FREEZE.json` |
| TEST lock | `results/v3_hppo/TEST_LOCK.json` |
| Evaluation consumed | `results/v3_hppo/EVALUATION_CONSUMED.json` |
| Raw TEST rows | `results/v3_hppo/raw/` |
| Statistics | `results/v3_hppo/statistics/` |
| Ablation (development) | `results/v3_hppo/ablation/` |
| Claims discipline | `paper/CLAIMS.md` |

## Historical V2 (preserved; includes DiscretePPO)

`results/v2/final/`

## Historical V1

`results/final/`

## Regenerate paper displays only

```bash
python scripts/paper/record_illustrative_val_episode.py   # VAL case study JSON (optional if present)
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Do **not** re-open the consumed TEST. Fig. 1 is VAL methodology illustration only.
