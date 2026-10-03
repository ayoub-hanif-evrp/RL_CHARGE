# Table 4B — TEST amount-policy sensitivity

Same frozen SynthCharge FA-HPPO checkpoints on the V3 TEST. Max ≈ free amount → continuous fine-tuning is not the main driver; Min often collapses via ZERO_CHARGE_NOOP and must not be over-interpreted.

| Method | u policy | Feasibility | 95% CI | Completion | Role |
|---|---|---|---|---|---|
| FA-HPPO | learned Beta | 0.942 | [0.920, 0.960] | 4.2 | primary FA-HPPO |
| FA-HPPO-Max | forced u=1 (upper SOC) | 0.943 | [0.923, 0.961] | 4.2 | envelope upper-bound sensitivity |
| FA-HPPO-Min | forced u=0 (lower SOC) | 0.079 | [0.053, 0.107] | 9.4 | degenerate lower-bound stress test |
