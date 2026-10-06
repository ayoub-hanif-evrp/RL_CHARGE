# Paper area

## Current (V4, standalone)

- `V4_CLAIMS.md` — standalone claim discipline and limitations
- Publication figures/tables: `results/v4/paper/`
- Benchmark wording: `docs/V4_BENCHMARK_WORDING.md`

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

## Historical (do not mix into V4 manuscript package)

- `CLAIMS.md` / `RESULTS_MAP.md` — historical V3 claim map
- `results/v3/paper/` — historical V3 publication displays

Do **not** retrain or re-open a consumed TEST from this directory.
