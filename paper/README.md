# Paper area

Manuscript claim discipline and results map. **Publication figures/tables live in `results_paper/`.**

- `CLAIMS.md` — claim discipline
- `RESULTS_MAP.md` — every figure/table → raw source (`results_paper/`)

Regenerate publication outputs:

```bash
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Do **not** retrain or re-open TEST from this directory.
