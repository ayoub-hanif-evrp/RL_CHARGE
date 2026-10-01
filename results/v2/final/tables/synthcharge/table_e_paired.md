# SynthCharge Table E — Paired HybridPPO vs DiscretePPO

Primary family; Holm within two comparisons.

| benchmark | family | comparison | effect | ci95_lo | ci95_hi | p_raw | p_holm | n_units | test |
|---|---|---|---|---|---|---|---|---|---|
| SynthCharge fresh TEST | feasibility | HybridPPO - DiscretePPO | -0.0044 | -0.0267 | 0.0156 | 0.8505 | 0.8505 | 90 | paired sign-flip / permutation on seed-averaged parent/route differences |
| SynthCharge fresh TEST | completion_all_routes | HybridPPO - DiscretePPO | 0.1158 | -0.0035 | 0.2474 | 0.06355 | 0.1271 | 90 | paired sign-flip / permutation on seed-averaged parent/route differences |
