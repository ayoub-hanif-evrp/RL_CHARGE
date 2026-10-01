# PAPER FACTS (claims supported by final evidence only)

## DEVELOPMENT (not TEST)
- B2 was frozen from gold TRAIN/VAL diagnostics (seeds 42/43).

## FRESH SynthCharge TEST
- HybridPPO feasibility mean across 5 seeds: 0.902
- DiscretePPO feasibility mean across 5 seeds: 0.907
- HybridPPO charging-required feasibility: 0.881
- DiscretePPO charging-required feasibility: 0.883
- HybridPPO failure-retaining completion: 4.395
- DiscretePPO failure-retaining completion: 4.279
- See table_e_paired.md for Holm-adjusted HybridPPO vs DiscretePPO tests.

## LEGACY same-domain challenge (not fresh)
- HybridPPO feasibility: 0.815
- DiscretePPO feasibility: 0.915

