# Paper claims discipline (FA-HPPO)

## Central contribution (supported)

**Feasibility-aware charging control** for fixed-route EV charging, implemented as FA-HPPO / HybridPPO (B2):

- energy-continuation lower SOC bound + optimistic time-feasibility upper SOC bound;
- TRAIN-only return scaling that removes a severe optimization pathology;
- Hybrid discrete station / continuous amount action suited to the problem.

Generic Hybrid PPO is **not** claimed as novel.

## Supported confirmatory claims (V3 TEST)

Allowed forms if evidence supports them:

- FA-HPPO achieved X feasibility on the fresh independently generated SynthCharge TEST (mean across five seeds, with CI).
- FA-HPPO improved feasibility / completion relative to GreedyMin, GreedyFull, and OneStepLookahead; report Holm-adjusted p-values for the predeclared family.
- On the charging-required subset (144 routes), FA-HPPO feasibility was …

If significance is absent: say so.

## Supported methodological claims (development VAL only)

- Return scaling reduces value-loss / gradient-norm pathology by orders of magnitude (B0/B1 vs B3/B2).
- Time-aware cap further improves parent-balanced VAL feasibility (B3 → B2).
- Full B2 is best among the 2×2 on gold VAL.

## Amount-policy interpretation (must be honest)

On the V3 TEST, **FA-HPPO-Max** (\(u=1\)) matches free FA-HPPO within noise (~94.3% vs ~94.2%; 173/180 routes tied).

Therefore the paper **must not** claim that learning the continuous charge quantity is what drives the strong result.

Preferred framing:

> The feasibility-aware SOC envelope accounts for much of the observed
> performance. Once a time-aware upper SOC bound is imposed, targeting the upper
> bound performs nearly identically to learned continuous amount control on this
> benchmark.

Preferred central contribution statement:

> We introduce a feasibility-aware action formulation for fixed-route EV
> charging that combines an energy-continuation lower SOC bound, an optimistic
> time-feasibility upper SOC bound, and PPO-based station/continue control.

Then report honestly that the learned continuous amount head does not
significantly improve over forcing the time-aware upper bound on this benchmark.

**FA-HPPO-Min** (\(u=0\)) collapses (~7.9%), largely via `ZERO_CHARGE_NOOP`. Keep it (predeclared) in secondary/appendix tables; do **not** use it to argue that “learning how much is crucial.”

## Development-only observations

- B0/B1/B2/B3 five-seed gold TRAIN/VAL ablation.
- Existing two-seed V2 diagnostics under `results/v2/diagnostics/`.
- Certified-PyVRP route-construction diagnostics.
- V2 HybridPPO vs DiscretePPO comparison (historical; not V3 confirmatory).

## Experimental-history transparency (DiscretePPO)

An earlier frozen stage (`results/v2/final/`) additionally studied a discretized PPO comparator. Those results are retained in the public repository but fall outside the predeclared V3 confirmatory comparison (FA-HPPO vs three heuristics). Do not imply DiscretePPO was invalid.

## Limitations (must appear)

- SynthCharge uses **linear** energy; it does **not** validate EVRPTW-GR terrain/payload physics.
- Customer sequences are fixed; FA-HPPO does **not** solve routing.
- Charging certificates are positive witnesses; timeout/exhaustion is **unverified**, not infeasible.
- Native frvcpy is exact only for its FRVCP assumptions, **not** globally exact for EVRPTW-GR.
- V3 is a **stratified, certificate-filtered** SynthCharge stress benchmark (~6% candidate acceptance), not the natural frequency of generator outputs.
- V3 is a **fresh held-out / independently generated SynthCharge TEST**, not distribution-shift generalization to a different external domain.
- Legacy EVRPTW-GR challenge (V2) is not a fresh TEST.

## Claims that MUST NOT be made

- Continuous amount learning is clearly superior to targeting the feasibility-aware upper SOC bound on this TEST.
- HybridPPO globally solves EV routing / chooses customer order.
- Certificate search proves infeasibility on timeout.
- frvcpy is exact for the EVRPTW-GR formulation.
- SynthCharge validates official terrain/payload physics.
- Restricted label setting / certificate search is globally optimal.
- Generic Hybrid PPO is novel.
- V3 demonstrates external domain generalization beyond SynthCharge.
- DiscretePPO was removed because it was invalid.
- “State of the art” / “superior to all methods” without exact supporting tests.
