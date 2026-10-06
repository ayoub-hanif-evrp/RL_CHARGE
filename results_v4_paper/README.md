# V4 paper-facing package (standalone)

This directory contains **only** V4 manuscript-facing evidence.

## Primary TEST result

FA-HPPO achieves **94.56%** mean feasibility (5 seeds) on the fresh
independently generated held-out SynthCharge TEST and substantially outperforms
One-step lookahead, Greedy full charge, and Greedy minimum charge under the same
feasibility-aware environment.

## Reproduce displays (no TEST rerun)

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

## Benchmark strata

Balanced quotas are **layout × frozen-route-length bin** (20 routes/cell).
See `docs/V4_BENCHMARK_WORDING.md`.

## Claims / limitations

See `paper/V4_CLAIMS.md`.
