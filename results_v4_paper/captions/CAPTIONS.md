# V4 figure captions (standalone)

All TEST figures use the fresh independently generated held-out SynthCharge TEST.
TRAIN/VAL figures are development evidence; TEST was not used for training or checkpoint selection.

**Fig. 1.** Fixed-route EV charging use case: the customer sequence is fixed while charging-station insertion and charge amount are learned.

**Fig. 2.** FA-HPPO control loop with feasibility-aware shielding, hybrid discrete/continuous decisions, SOC envelope mapping, and the normalized time-horizon reward.

**Fig. 3.** Feasibility-aware SOC envelope: continuation lower bound, optimistic time-aware upper bound, and target SOC parameterized by continuous amount \(u\).

**Fig. 4.** Illustrative TRAIN/VAL SOC trajectory with charging stops on a fixed customer route (methodology illustration; not TEST).

**Fig. 5.** Main TEST feasibility for FA-HPPO and three heuristic/planning baselines under the same feasibility-aware environment. FA-HPPO shows five-seed Student-t uncertainty; baselines are deterministic.

**Fig. 6.** Failure-retaining completion on all TEST routes (infeasible episodes retain the route horizon \(H\)). Lower is better.

**Fig. 7.** Feasibility on the charging-required TEST subset (144 routes). FA-HPPO reaches 100% on the complementary 36 no-charge-required routes.

**Fig. 8.** Empirical CDF of failure-retaining completion for FA-HPPO and baselines on all TEST routes.

**Fig. 9.** TEST feasibility by layout (`C`, `R`, `RC`), exposing regime dependence.

**Fig. 10.** FA-HPPO TEST feasibility by balanced frozen-route-length bin (`short`, `medium`, `long`).

**Fig. 11.** Layout × route-length feasibility heatmap for FA-HPPO on the locked TEST (balanced 3×3 strata).

**Fig. 12.** VALIDATION reward-development ablation. Potential-based shaping did not improve validation feasibility; the simpler time-horizon reward was selected.

**Fig. 13.** TRAIN objective episode return (\(G=-T\) on success, \(G=-H\) on failure), mean across seeds with Student-t 95% interval on the common update support.

**Fig. 14.** TRAIN PPO policy and value losses (diagnostics; decreasing loss is not claimed as convergence).

**Fig. 15.** VALIDATION parent-balanced feasibility versus PPO update.

**Fig. A1.** Pre-clip gradient norm during authoritative training.

**Fig. A3.** Per-seed FA-HPPO feasibility on the locked TEST.

**Fig. A4.** Pooled FA-HPPO failure-reason counts on the locked TEST.
