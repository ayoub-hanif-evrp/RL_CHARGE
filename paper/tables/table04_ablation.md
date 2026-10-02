# Table 4 — Ablation

Development/validation ablation unless marked eval-on-TEST amount ablation.

| Variant | Time cap | Return scale | Seeds done | Mean parent-bal. feas | Mean value loss |
|---|---|---|---|---|---|
| B0 | off | off | 3 | 0.458 | 335459.4526 |
| B1 | on | off | 0 | NA | NA |
| B3 | off | on | 0 | NA | NA |
| B2 | on | on | 0 | NA | NA |
| FA-HPPO-Min | on | on (frozen) | 5 (eval) | 0.079 | n/a (eval) |
| FA-HPPO-Max | on | on (frozen) | 5 (eval) | 0.943 | n/a (eval) |
