# Table 4 — Ablation

Development/validation ablation unless marked eval-on-TEST amount ablation.

| Variant | Time cap | Return scale | Seeds done | Mean parent-bal. feas | Mean value loss |
|---|---|---|---|---|---|
| B0 | off | off | 5 | 0.467 | 331420.7465 |
| B1 | on | off | 5 | 0.554 | 314350.6094 |
| B3 | off | on | 5 | 0.801 | 0.0019 |
| B2 | on | on | 5 | 0.900 | 0.0017 |
| FA-HPPO-Min | on | on (frozen) | 5 (eval) | 0.079 | n/a (eval) |
| FA-HPPO-Max | on | on (frozen) | 5 (eval) | 0.943 | n/a (eval) |
