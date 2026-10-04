# Figure captions (`results_paper/`)

Use PDF figures in the LaTeX manuscript; PNG for preview/GitHub.

## Main paper

**Figure 1.** Conceptual overview of feasibility-aware hybrid PPO (FA-HPPO) for fixed-route charging control. (a) Customer visit order is fixed; the policy may insert a charging detour C_i -> S_j -> C_{i+1} and does not reorder customers. (b) At a selected station, departure SOC is restricted to a feasibility-aware interval; u in [0,1] maps to SOC_target = SOC_lower + u (SOC_upper - SOC_lower). (c) A discrete head selects CONTINUE/station under masking; a Beta continuous head yields an explicit u that maps into the interval. Schematic only — not empirical evidence.

**Figure 2.** Confirmatory performance on the independently generated V3 SynthCharge TEST (180 routes). FA-HPPO aggregates five independently trained seeds; small points are seed means and the larger marker is the across-seed mean. Error bars are the predeclared hierarchical 95% bootstrap interval (seed -> route). Baselines are deterministic. Failure-retaining completion assigns horizon H to infeasible episodes. Higher feasibility and lower completion are better. Charging-required subset metrics appear in Table 2.

**Figure 3.** Paired route-level effects of FA-HPPO versus each baseline on V3 SynthCharge TEST (n=180; five FA-HPPO seeds). Points show seed-averaged paired differences with 95% CIs; numeric labels give the point estimates. Positive feasibility / negative completion favor FA-HPPO. Holm-adjusted permutation p-values are in Table A2 (Monte Carlo floor <5e-5).

**Figure 4.** Robustness on charging-required V3 TEST routes by layout x length. (a) FA-HPPO feasibility (%). (b) FA-HPPO improvement over OneStepLookahead (percentage points) on the same cells; sequential 0-max scale because all cells are nonnegative in the frozen data. Computed from frozen V3 raw rows only.

**Figure 5.** Gold development B0/B1/B3/B2 ablation on official EVRPTW-GR VAL (five seeds 42-46): parent-balanced feasibility. Variant definitions (time-aware upper cap / return scaling) are in Table 3 and the caption — not embedded in the plot. Development evidence only; not V3 TEST.

**Figure 6.** V3 TEST amount-policy sensitivity with the same frozen FA-HPPO checkpoints: learned continuous u, forced u=1 (Max), and forced u=0 (Min). Small points are seeds; error bars are hierarchical 95% CIs. Learned vs Max ties on 173/180 routes. Prefer the interpretation that the feasibility-aware envelope accounts for much of the observed performance; do not claim continuous amount learning is the main driver.

## Appendix

**Figure A1.** Illustrative SynthCharge VAL charging trajectory; not TEST evidence. (a) Fixed-route geometry with charging insertions. (b) SOC along the visit sequence with feasible intervals at charges.

**Figure A2.** Illustrative hybrid decision from the same VAL episode; not TEST evidence. (a) Masked discrete CONTINUE/station probabilities. (b) Continuous Beta density for u and resulting target SOC.

**Figure A3.** Gold development learning curves (parent-balanced VAL feasibility, %): mean +/- SD across five seeds for B0/B1/B3/B2. Line styles distinguish variants; not V3 TEST evidence.

**Figure A4.** Effect of methodology components on PPO optimization stability (gold development curves). (a) Value loss. (b) Pre-clip gradient norm. Shared legend; bands are mean +/- SD (not CIs). Log y-scale. Development VAL only.

**Figure A5.** Post-hoc SynthCharge TRAIN/VAL envelope ablation (Arrival-to-Max, EnergyLower-to-Max, EnergyLower-to-TimeUpper); five seeds 42-46 shown as paired seed trajectories with the mean overlay. Y-axis is zoomed (94-101%). Post-hoc development study — not confirmatory V3 TEST. Do not claim each envelope component monotonically improves feasibility.

**Figure A6.** Frozen-raw FA-HPPO failure consistency on V3 TEST (no policy replay). The 20 routes with at least one failing seed, abbreviated R01-R20 (full IDs in Table A3 / ROUTE_ALIASES.json), sorted by number of failing seeds. 160/180 routes succeed under all five seeds. All recorded reasons are NO_FEASIBLE_ACTION.

**Figure A7.** Charging-required failure rate (%) by layout x length on frozen V3 TEST FA-HPPO rows. Independent of Fig. A6 for readability.

**Figure A8.** FA-HPPO per-seed robustness on V3 TEST. (a) Feasibility (%; zoomed y-axis; shaded hierarchical 95% CI). (b) Failure-retaining completion.

**Figure A9.** Native FRVCP reference benchmark (separate archive; n=133). Not a SynthCharge/EVRPTW-GR exact comparison. Do not treat frvcpy as an exact oracle for the paper's SynthCharge formulation.
