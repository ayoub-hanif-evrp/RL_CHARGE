# V4 figure captions (standalone)

All TEST figures use the fresh independently generated held-out SynthCharge TEST.
TRAIN/VAL figures are development evidence; TEST was not used for training or checkpoint selection.
Uncertainty conventions are stated explicitly per figure (never as generic "error bars").

**Fig. 1 (methodology).** Fixed-route EV charging use case. The solid path is the frozen customer sequence (depot → customers → depot). Candidate charging stations are shown separately; the dashed orange path illustrates one charging insertion (customer → station → next frozen customer). The customer order itself is unchanged. Charge insertion is a learned decision; the figure only depicts the geometric idea.

**Fig. 2 (methodology).** FA-HPPO control loop: fixed-route state → feasibility shield → features → hybrid PPO → CONTINUE/station → continuous amount $u$ with SOC mapping → simulator transition. Equations below the workflow state the normalized time-horizon reward and the SOC-target map $SOC_{\mathrm{target}}=SOC_{\mathrm{lower}}+u(SOC_{\mathrm{upper}}-SOC_{\mathrm{lower}})$.

**Fig. 3 (methodology).** Single-decision SOC envelope (not a simulated trajectory). Vertical scale shows arrival SOC, energy-continuation lower bound $SOC_{\mathrm{lower}}$, optimistic time-feasibility upper bound $SOC_{\mathrm{upper}}$, and selected target SOC with $SOC_{\mathrm{lower}}\le SOC_{\mathrm{target}}\le SOC_{\mathrm{upper}}$ and $SOC_{\mathrm{target}}=SOC_{\mathrm{lower}}+u(SOC_{\mathrm{upper}}-SOC_{\mathrm{lower}})$.

**Fig. 4 (VALIDATION).** Real SOC trajectory from frozen V4 evidence (not TEST). Selection rule: First charging-required VALIDATION route in stable route_id order that is feasible under the frozen V4 seed-42 checkpoint and contains at least one charging action. Selected route `v2sc_sc_C_n15_s200037_01`, seed 42. Markers show station arrivals and SOC envelopes at charge decisions.

**Fig. 5 (TEST).** Main TEST feasibility for FA-HPPO and three deterministic baselines under the same feasibility-aware environment. FA-HPPO whiskers are the half-width of the 95% Student-t interval across five seeds; baselines have no uncertainty artists.

**Fig. 6 (TEST).** Failure-retaining completion on all TEST routes (infeasible episodes retain the route horizon $H$). Lower is better. FA-HPPO whiskers are the 95% Student-t half-width across five seeds; baselines are deterministic point values.

**Fig. 7 (TEST).** Feasibility on the charging-required TEST subset (144 routes). FA-HPPO whiskers are the 95% Student-t half-width across five seeds; baselines are deterministic. FA-HPPO reaches 100% on the complementary 36 no-charge-required routes.

**Fig. 8 (TEST).** Empirical CDF of failure-retaining completion. Every method contributes exactly 180 route-level values: FA-HPPO values are seed-averaged per route before constructing the ECDF; deterministic baselines contribute their single value per route.

**Fig. 9 (TEST).** TEST feasibility by layout (`C`, `R`, `RC`). FA-HPPO whiskers are 95% Student-t half-widths across five seeds within each layout; baselines have no uncertainty artists.

**Fig. 10 (TEST; supplemental candidate).** FA-HPPO TEST feasibility by balanced frozen-route-length bin (`short`, `medium`, `long`). Whiskers are 95% Student-t half-widths across five seeds. Bins are frozen-route-length strata, not customer-scale cells.

**Fig. 11 (TEST).** Layout × frozen-route-length feasibility heatmap for FA-HPPO on the locked TEST (balanced 3×3 strata). Color scale is fixed to 0%–100%.

**Fig. 12 (VALIDATION).** Reward-development ablation on VALIDATION. Display labels: Progress, Normalized, PBRS, Time-horizon (selected, diamond marker). Uncertainty is mean ± SD across five training seeds (not a 95% Student-t CI). Potential-based shaping did not improve validation feasibility; the simpler time-horizon reward was selected.

**Fig. 13 (TRAIN; supplemental).** TRAIN objective episode return ($G=-T$ on success, $G=-H$ on failure), mean across seeds with Student-t 95% interval on the common update support (no extrapolation after early stopping).

**Fig. 14 (TRAIN; supplemental).** TRAIN PPO policy and value losses with Student-t 95% intervals on the common update support. Diagnostics only; decreasing loss is not claimed as convergence. Policy loss may be negative.

**Fig. 15 (VALIDATION; supplemental).** VALIDATION parent-balanced feasibility versus PPO update, mean with Student-t 95% interval across five seeds on the common update support.

**Fig. A1 (TRAIN; supplemental).** Pre-clip gradient norm during authoritative training (Student-t 95% interval; log axis only where valid).

**Fig. A3 (TEST; supplemental).** Per-seed FA-HPPO feasibility on the locked TEST.

**Fig. A4 (TEST; supplemental).** Pooled FA-HPPO failure-reason counts on the locked TEST (route×seed). Display labels are human-readable; underlying reason codes are preserved in the raw evidence.
