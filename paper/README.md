# Paper area (active = V4)

## Current

- `V4_CLAIMS.md` — standalone claim discipline and limitations
- Publication figures/tables: `results/v4/paper/`
- Benchmark wording: `docs/methodology/V4_BENCHMARK_WORDING.md`
- Authoritative models: `models/v4/fa_hppo/` (binaries local/Release; checksums tracked)

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

## Historical V3 (do not mix into V4)

- Claims / results map: `results/v3/paper/claims/`
- Publication displays: `results/v3/paper/`

Do **not** retrain or re-open a consumed TEST from this directory.
