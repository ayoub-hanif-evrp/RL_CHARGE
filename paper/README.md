# Paper area

Publication-facing maps and regenerated figures/tables for the FA-HPPO paper.

- `CLAIMS.md` — claim discipline
- `RESULTS_MAP.md` — every figure/table → raw source
- `figures/` — PDF/SVG + PNG (generated, not hand-edited)
- `tables/` — Markdown + CSV + LaTeX

Regenerate with:

```bash
python scripts/paper/build_paper_artifacts.py
python scripts/paper/build_paper_artifacts.py --verify
```

Do **not** retrain or re-open TEST from this directory.
