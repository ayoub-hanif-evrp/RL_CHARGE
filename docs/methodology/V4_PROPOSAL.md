# V4 proposal (do not execute without explicit approval)

V3 remains frozen and valid. The items below are **proposals only**.

| ID | Proposal | Scientific purpose | Code change | Retrain? | New TEST? | Compute (order) | Hypothesis |
|----|----------|--------------------|-------------|----------|-----------|-----------------|------------|
| **V4-A** | Correct station-arrival feature (distinct arrival / lower / upper) | Fix accidental duplicate index 6=9 | `features.py` (+ possibly STATION_DIM docs) | Yes | Yes | ~5 seeds × paper budget | Distinct arrival feature improves or changes VAL/TEST feasibility |
| **V4-B** | Endpoint-capable continuous head | Allow natural \(u\to\{0,1\}\) | `policy.py` action head | Yes | Yes | ~5 seeds × paper budget | Endpoint-capable head differs from Max/free pattern |
| **V4-C** | Strong planning/optimization comparator | Address weak baseline critique | new `src/planning/` (or MILP/DP) | No (comparator) | Prefer yes for confirmatory | Prototype on VAL; full 180 if accepted | Planner matches/exceeds FA-HPPO under same assumptions |
| **V4-D** | Fresh untouched TEST | Confirmatory family including V4 method ± planner | protocol/lock only | Uses V4 ckpts | Yes (new) | Corpus gen + 180×methods | Predeclared V4 claims without consuming V3 |
| **V4-E** | Optional nonlinear / OOD benchmark | Limit of linear SynthCharge | physics/data | Yes | Yes | Large | FA-HPPO envelope still helps under nonlinear charge/energy |
| **V4-F** | Reward study: `V3_TIME` / `V4_BASE` / `V4_PBRS` / `V4_BASE_NO_L_FAIL` | Clear completion-time objective; PBRS as ablation only; no arbitrary weights | `src/rl/rewards.py` + env wiring | Yes (TRAIN/VAL) | **No** until freeze + dedicated TEST protocol | ~20 cells × paper budget | **Selected:** `V4_BASE_NO_L_FAIL` (lexicographic VAL). PBRS did not improve mean VAL feasibility vs unshaped base; result retained as ablation |

V4-F design document: `docs/V4_REWARD_DESIGN.md`.  
Artifact root: `results/v4/reward_development/` (does not modify V3).

## Hard rules if V4 starts

- New method / protocol identifiers.
- No reuse of consumed V3 TEST for selection or retuning.
- Do not silently patch `src/` and keep claiming V3 numbers.
- Report unfavorable results.

## Title note (manuscript only)

Prefer emphasizing **feasibility-aware** control over “hybrid continuous superiority.” Keep FA-HPPO as the method name in the body. No mass rename of code artifacts.
