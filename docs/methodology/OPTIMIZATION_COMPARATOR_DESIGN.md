# Optimization comparator design

**Status:** design / feasibility study only.  
**Companion:** `docs/STRONG_COMPARATOR_FEASIBILITY_STUDY.md`

Do **not** insert a new comparator into the consumed V3 confirmatory family.

## Goal

A stronger fixed-route charging comparator under **SynthCharge / V3 mathematical assumptions**, beyond reactive GreedyMin/Full and shallow OneStepLookahead.

## Candidates (priority order)

### 1. Dynamic programming / label-setting on the frozen route

| Aspect | Notes |
|--------|-------|
| State | position in customer sequence × time × SOC |
| Actions | CONTINUE vs station insert; charge amount continuous or discretized |
| Exactness | Exact only with proved dominance / breakpoint structure |
| Continuous charge | Requires discretization or analytical breakpoints |
| Shield | Must encode energy continuation + optional time SOC cap + loop guards |
| Runtime | Sensitive to SOC/time discretization; may be OK for short/medium routes |

**Open question:** Does SynthCharge linear energy + linear charging admit exact breakpoints? Not proved in-repo.

### 2. Controlled-discretization shortest path

Same as DP with explicit SOC grid / charge grid. Exact for the discretized model; approximate for continuous \(u\).

### 3. MILP

| Aspect | Notes |
|--------|-------|
| Variables | station visit indicators along sequence; SOC/time continuous |
| Exactness | Exact for the formulated model |
| Risk | Silent mismatch vs shield (ZERO_CHARGE_NOOP, revisit, time-cap optimism) |
| Runtime | Likely OK for small \(n\); 180 mixed lengths need timeouts |

### 4. Beam search / MPC

Heuristic; stronger than one-step lookahead if depth/beam large. Development-friendly. Not exact.

### 5. frvcpy

Exact only for **native FRVCP**. Not an EVRPTW-GR / SynthCharge-TW oracle unless equivalence is proved. Keep as appendix reference only.

## Shared-shield requirement

Any “same-problem” claim must match:

- frozen customer order;
- CONTINUE vs charge actions;
- linear SynthCharge energy/charge;
- time windows / waiting / service;
- energy-continuation legality;
- optional optimistic time SOC upper bound;
- station revisit / loop guards;
- failure-retaining completion convention for scoring (if used).

## Evaluation policy

| Path | Label |
|------|-------|
| TRAIN/VAL validation now | POST-HOC DEVELOPMENT |
| Consumed V3 TEST | POST-HOC / EXPLORATORY COMPARATOR only; never edit `paired_primary.json` |
| Prefer | New predeclared **V4** TEST with comparator in the primary family |

## Recommendation

1. Prototype restricted DP / beam on SynthCharge VAL.  
2. If competitive and equivalence-checked, open V4 protocol (V4-C + V4-D).  
3. Do not silently upgrade the V3 main table.
