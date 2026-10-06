# Table 4 — Reward development ablation (VALIDATION)

TRAIN/VAL development evidence only. TEST not used for selection.
Uncertainty is mean ± SD across five training seeds (not a 95% Student-t CI).
Full definitions: Progress = Time + progress failure; Normalized = Normalized equivalent;
PBRS = Potential-shaped; Time-horizon = selected normalized time-horizon reward.

| Reward | VAL feasibility (mean ± SD) | VAL completion (mean ± SD) | Mean best update | Selected |
|---|---:|---:|---:|---|
| Progress | 98.2% ± 1.7 | 3.915 ± 0.123 | 220 |  |
| Normalized | 98.2% ± 1.7 | 3.915 ± 0.123 | 220 |  |
| PBRS | 97.8% ± 1.1 | 3.891 ± 0.082 | 222 |  |
| Time-horizon | 98.2% ± 1.0 | 3.878 ± 0.046 | 292 | yes |
