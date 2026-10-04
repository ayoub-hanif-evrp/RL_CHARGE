# Figure captions (`results_paper/`)

Use PDF figures in the LaTeX manuscript; PNG for preview/GitHub.

## Main paper

**Figure 1.** Conceptual overview of feasibility-aware hybrid PPO (FA-HPPO) for fixed-route charging control. (a) The customer visit order is fixed; the policy chooses CONTINUE or an optional charging-station insertion and does not reorder customers. (b) When a station is selected, the admissible departure SOC is restricted to a feasibility-aware interval, and the continuous action u in [0,1] maps to SOC_target = SOC_lower + u (SOC_upper - SOC_lower). (c) A discrete head selects CONTINUE/station under action masking; a Beta continuous head produces u. Schematic only — not empirical evidence.

**Figure 2.** Confirmatory performance on the independently generated V3 SynthCharge TEST (180 routes). FA-HPPO results aggregate five independently trained seeds; small points denote individual seed results and the larger marker denotes the across-seed mean. Error bars show the predeclared hierarchical 95% bootstrap interval over training seed and route. Baselines are deterministic. Failure-retaining completion assigns horizon H to infeasible episodes. Higher feasibility and lower completion are better. Charging-required subset metrics appear in Table 2.

**Figure 3.** Paired route-level effects of FA-HPPO versus each baseline on the V3 SynthCharge TEST (n=180 routes; five FA-HPPO training seeds). Points show seed-averaged paired differences with 95% confidence intervals; the vertical line marks zero. Positive feasibility differences and negative completion differences favor FA-HPPO. Holm-adjusted permutation p-values for the predeclared primary comparisons are reported in Table A2 (Monte Carlo floor <5e-5).

**Figure 4.** Robustness of FA-HPPO on charging-required V3 TEST routes by layout x route-length cell. (a) FA-HPPO feasibility (%; mean over route x seed rows in each cell). (b) FA-HPPO minus OneStepLookahead feasibility (percentage points) on the same cells, using a diverging scale centered at zero. Computed from frozen V3 raw rows only.

**Figure 5.** Mechanism evidence. (a) Gold development B0/B1/B3/B2 ablation on official EVRPTW-GR VAL (five seeds 42-46): parent-balanced feasibility with compact variant codes; the component matrix lists time-aware upper cap and return scaling. Development evidence only — not V3 TEST. (b) V3 TEST amount-policy sensitivity with the same frozen FA-HPPO checkpoints: learned continuous u, forced u=1 (Max), and forced u=0 (Min). Small points are seeds; error bars are hierarchical 95% CIs. Learned vs Max ties on 173/180 routes. Prefer the interpretation that the feasibility-aware envelope accounts for much of the observed performance; do not claim continuous amount learning is the main driver.

## Appendix

**Figure A1.** Illustrative SynthCharge VAL episode; not TEST evidence. Fixed-route geometry with charging insertions, SOC evolution, and one hybrid discrete/continuous decision with the feasible SOC interval.

**Figure A2.** Gold development learning curves (parent-balanced VAL feasibility, %): mean +/- SD across five seeds for B0/B1/B3/B2. Line styles distinguish variants; not V3 TEST evidence.

**Figure A3.** Effect of return scaling / methodology components on PPO optimization stability (gold development curves). (a) Value loss vs update. (b) Pre-clip gradient norm vs update. Bands are mean +/- SD (not confidence intervals). Log y-scale. Development VAL only.

**Figure A4.** Post-hoc SynthCharge TRAIN/VAL envelope ablation (Arrival-to-Max, EnergyLower-to-Max, EnergyLower-to-TimeUpper); five seeds 42-46. Points are seed values; bars mark means. Post-hoc development study — not confirmatory V3 TEST evidence. Do not interpret as a monotonic causal claim that each envelope component improves feasibility.

**Figure A5.** Frozen-raw FA-HPPO failure analysis on V3 TEST (no policy replay). (a) The 20 routes with at least one failing seed, sorted by number of failing seeds; annotation notes that 160/180 routes succeed under all five seeds. (b) Charging-required failure rate (%) by layout x length. All recorded failure reasons are NO_FEASIBLE_ACTION. Step-level pre-failure trajectories were not stored.

**Figure A6.** FA-HPPO per-seed robustness on V3 TEST. (a) Feasibility (%; y-axis truncated for readability, with hierarchical 95% CI band). (b) Failure-retaining completion. Seed values also appear in the main result plots and Table A1.

**Figure A7.** Native FRVCP reference benchmark (separate archive; n=133). This is a separate native FRVCP benchmark and is not a SynthCharge/EVRPTW-GR exact comparison. Do not treat frvcpy as an exact oracle for the paper's SynthCharge formulation.
