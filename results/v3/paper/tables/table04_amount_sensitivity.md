# Table 4 — TEST amount-policy sensitivity

Same frozen checkpoints on V3 TEST. Visual summary in Fig. 14.

| Method | u policy | Feasibility | 95% CI | Completion | Route vs Max | Interpretation |
|---|---|---|---|---|---|---|
| FA-HPPO | learned Beta | 0.942 | [0.920, 0.961] | 4.200 | better=3, tied=173, worse=4 | primary |
| FA-HPPO-Max | forced u=1 | 0.943 | [0.921, 0.961] | 4.179 | — | ≈ free |
| FA-HPPO-Min | forced u=0 | 0.079 | [0.054, 0.108] | 9.440 | — | degenerate stress |
