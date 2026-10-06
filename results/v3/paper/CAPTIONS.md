# Figure captions (`results/v3/paper/figures/`)

PNG only (600 dpi). Flat folder (no appendix/). Captions carry interpretation.

**Figure 1 — Problem setting.** Illustrative SynthCharge VAL episode (not TEST). Depot (square), numbered customers in fixed order, unused/used charging stations (triangles with IDs), dashed fixed customer sequence, and solid executed FA-HPPO path with charging detours.

**Figure 2 — Method overview.** FA-HPPO hybrid control: discrete CONTINUE/station choice, continuous charge fraction $u$ when charging, mapped through a feasibility-aware SOC envelope, then environment step.

**Figure 3 — SOC envelope.** Arrival SOC, energy-continuation lower bound, time-aware upper bound, and target SOC $= lower + u(upper-lower)$.

**Figure 4 — Illustrative SOC trajectory.** Same VAL episode as Fig. 1. SOC along the executed visit sequence; shaded intervals mark feasible departure bands at charging stops.

**Figure 5 — Main TEST feasibility.** Locked V3 SynthCharge TEST (180 routes). Vertical bars; FA-HPPO whiskers are hierarchical 95% bootstrap intervals. Higher is better.

**Figure 6 — Main TEST completion.** Failure-retaining completion on the same TEST (infeasible episodes keep horizon $H$). Lower is better.

**Figure 7 — Charging-required subset.** Feasibility restricted to routes that require charging (harder stratum).

**Figure 8 — Performance profiles.** Share of routes that are feasible and finished by a completion-time threshold. Curves asymptote at each method’s feasibility rate.

**Figure 9 — Feasibility by layout.** Charging-required TEST routes stratified by R / C / RC.

**Figure 10 — Feasibility by length.** Charging-required TEST routes stratified by short / medium / long.

**Figure 11 — Difficulty heatmap.** FA-HPPO feasibility (%) on charging-required cells (layout × length).

**Figure 12 — Gain vs Lookahead.** Feasibility improvement (pp) of FA-HPPO over OneStepLookahead by layout × length.

**Figure 13 — Development ablation.** Gold EVRPTW-GR VAL means for B0–B2 (time-aware cap × return scaling). Development evidence only.

**Figure 14 — Amount policies.** Same frozen checkpoints with learned $u$, forced $u=1$ (Max), and forced $u=0$ (Min).

**Figure 15 — Charging effort.** Mean station visits among feasible TEST routes.

**Figure 16 — Runtime.** Mean evaluation runtime per route (ms).

**Figure 17 — Failure regimes.** FA-HPPO failure rate (%) on charging-required cells by layout and length (aggregate; not per-seed).

**Figure 18 — Envelope variants.** Post-hoc SynthCharge VAL means for Arrival→Max / EnergyLower→Max / EnergyLower→TimeUpper. Development nuance only.

**Figure 19 — Charge decision snapshot.** One VAL charging decision: (a) discrete action probabilities (masked illegal stations); (b) continuous target inside the local SOC envelope.

**Figure 20 — Native FRVCP reference.** Separate FRVCP archive (n=133). Not an exact SynthCharge/EVRPTW-GR comparator.
