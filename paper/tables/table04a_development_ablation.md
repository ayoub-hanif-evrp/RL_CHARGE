# Table 4A — Development B0/B1/B3/B2

Development/validation ablation on gold TRAIN/VAL only — not fresh TEST evidence. B2 = full FA-HPPO (time-aware cap + TRAIN-only return scaling).

| Variant | Time cap | Return scale | Seeds | Mean parent-bal. VAL feas | Mean value loss |
|---|---|---|---|---|---|
| B0 | off | off | 5 | 0.467 | 331420.7465 |
| B1 | on | off | 5 | 0.554 | 314350.6094 |
| B3 | off | on | 5 | 0.801 | 0.0019 |
| B2 | on | on | 5 | 0.900 | 0.0017 |
