# Paper claims discipline (FA-HPPO)

## Supported confirmatory claims (only after V3 TEST)

Allowed forms if evidence supports them:

- FA-HPPO achieved X feasibility on the fresh V3 SynthCharge TEST (mean across five seeds, with CI).
- FA-HPPO improved feasibility relative to baseline Y by Z percentage points; paired test produced p=… (Holm-adjusted).
- On the charging-required subset (144 routes), FA-HPPO feasibility was …

If significance is absent: say so.

## Development-only observations

- B0/B1/B2/B3 five-seed gold TRAIN/VAL ablation (why B2 was frozen).
- Existing two-seed V2 diagnostics under `results/v2/diagnostics/`.
- Certified-PyVRP route-construction diagnostics.
- V2 HybridPPO vs DiscretePPO comparison (historical; not V3 confirmatory).

## Limitations (must appear)

- SynthCharge uses **linear** energy; it does **not** validate EVRPTW-GR terrain/payload physics.
- Customer sequences are fixed; FA-HPPO does **not** solve routing.
- Charging certificates are positive witnesses; timeout/exhaustion is **unverified**, not infeasible.
- Native frvcpy is exact only for its FRVCP assumptions, **not** globally exact for EVRPTW-GR.
- Legacy EVRPTW-GR challenge (V2) is not a fresh TEST.

## Claims that MUST NOT be made

- HybridPPO globally solves EV routing / chooses customer order.
- Certificate search proves infeasibility on timeout.
- frvcpy is exact for the EVRPTW-GR formulation.
- SynthCharge validates official terrain/payload physics.
- Restricted label setting / certificate search is globally optimal.
- Generic Hybrid PPO is novel.
- “State of the art” / “superior to all methods” without exact supporting tests.
- DiscretePPO was removed because it was invalid (it was not; it is historical V2 evidence).
