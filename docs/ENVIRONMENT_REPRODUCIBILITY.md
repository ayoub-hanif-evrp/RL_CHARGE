# Environment reproducibility

## Known versions (from archived experiment manifests)

Recovered from committed experiment metadata — **not invented**:

| Source | Field | Value |
|--------|-------|-------|
| `results/v3_hppo/ENVIRONMENT.json` | python | 3.12.10 |
| `results/v3_hppo/ENVIRONMENT.json` | torch | 2.14.0+cpu |
| `results/v3_hppo/ENVIRONMENT.json` | os / device | Windows-11 / cpu |
| `results/v2/final/ENVIRONMENT.json` | python | 3.12.10 |
| `results/v2/final/ENVIRONMENT.json` | torch | 2.14.0+cpu |
| `results/v2/final/ENVIRONMENT.json` | numpy | 2.5.3 |
| `results/v2/final/ENVIRONMENT.json` | matplotlib | 3.11.2 |
| `results/v2/final/ENVIRONMENT.json` | pyvrp | 0.14.0 |
| `results/v2/final/ENVIRONMENT.json` | frvcpy | 0.1.1 |
| `pyproject.toml` | pyvrp pin | 0.14.0 |

Method freeze SHA recorded in V2 env: `df35705c012b4ce88674a5b8906176c97838b06c`.

## Unknown / not recovered

No full `pip freeze` from the original V3 training machine was found in the
repository. Therefore a complete bit-for-bit lockfile **cannot** be
reconstructed without fabrication.

Unknown examples include (non-exhaustive): exact transitive dependency pins
beyond the packages listed above, CUDA builds (experiments used CPU), and
Windows-specific binary wheels.

## Partial lock hint

`requirements-experiment-lock.txt` lists **verified** package versions only.
It is a reconstruction aid, not a claim of complete environment fidelity.

## Capture for future runs

```bash
python scripts/repro/capture_environment.py
```

Writes `ENVIRONMENT_CAPTURE.json` (or a path you pass) with python, platform,
key package versions, and `pip freeze`.

## CI

GitHub Actions runs unit tests on **Python 3.11 and 3.12** (CPU torch).
CI versions may differ from the Windows experiment host; they validate
portability, not bit-identical numerics.
