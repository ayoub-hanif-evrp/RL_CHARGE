# Table 3 — Development methodology ablation

DEVELOPMENT / GOLD VALIDATION ONLY — not V3 TEST. Descriptive names primary; B0/B1/B3/B2 are provenance labels. Five seeds 42–46.

| Variant | Time-aware cap | Return scale | VAL feas. | SD | Charge-req feas. | Completion | Value loss | Grad norm |
|---|---|---|---|---|---|---|---|---|
| Base HPPO (B0) | off | off | 0.467 | 0.020 | 0.500 | 1089.4 | 331420.7465 | 2637.141 |
| + Time-aware cap (B1) | on | off | 0.554 | 0.052 | 0.565 | 1060.5 | 314350.6094 | 2000.386 |
| + Return scaling (B3) | off | on | 0.801 | 0.055 | 0.659 | 941.5 | 0.0019 | 0.512 |
| FA-HPPO full (B2) | on | on | 0.900 | 0.039 | 0.771 | 919.1 | 0.0017 | 0.605 |
