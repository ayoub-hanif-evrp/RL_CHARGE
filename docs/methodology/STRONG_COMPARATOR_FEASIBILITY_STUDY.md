# Strong comparator feasibility study (analysis only)

**Status:** documentation / design analysis. No new confirmatory baseline is
inserted into the consumed V3 TEST family.

**Context:** V3 confirmatory baselines are GreedyMin, GreedyFull, and
OneStepLookahead. Reviewers may ask for a stronger planning/optimization
comparator under the **same** fixed-route charging assumptions.

---

## Problem statement (V3 / SynthCharge fixed route)

- Frozen customer sequence (no routing decisions).
- Hybrid decisions: CONTINUE vs charge at a station; continuous target SOC in a
  feasibility interval.
- Linear energy / linear charging (SynthCharge profile).
- Time windows and service/waiting apply in the simulator.
- Transition-level shield: energy continuation + optional optimistic time SOC cap.
- Objective used by RL: maximize feasibility / minimize failure-retaining
  completion (horizon on failure).

Any “same-problem” optimizer must state whether it matches **all** of the above
or a restriction.

---

## Candidate: restricted label-setting (existing `src/exact/`)

| Item | Assessment |
|------|------------|
| State | Restricted discrete charge amounts (continuation-min / full) on frozen sequence |
| Actions | Station insert + restricted SOC choices |
| Objective | Feasibility / schedule existence under restricted action set |
| Charging | Continuous physics, but **restricted** decision set |
| Time windows | Partial / search-dependent; not claimed globally exact |
| SOC dynamics | Simulator energy model |
| Exact? | **No** for continuous FA-HPPO; exact only for its restricted action set |
| Continuous charge? | No (restricted) |
| Differs from V3 | Action set ≠ Beta / free \(u\) in envelope |
| Runtime on 180 | Certificate search already used for corpus construction; timeouts common |

**Verdict:** useful certificate / restricted witness; **not** an oracle for V3.

---

## Candidate: frvcpy / native FRVCP

| Item | Assessment |
|------|------------|
| State | Classic FRVCP on fixed route |
| Exact? | Exact under **FRVCP** assumptions |
| Matches V3? | **No** without a proof of mathematical equivalence to SynthCharge + TW + shield |
| Continuous charge? | FRVCP-specific |
| Runtime | Feasible on small FRVCP instances; already archived as appendix reference |

**Verdict:** keep as native FRVCP reference only. Do **not** claim frvcpy is an
exact oracle for V3.

---

## Candidate: dynamic programming / label-setting with continuous SOC

| Item | Assessment |
|------|------------|
| State | Position in sequence × time × SOC (continuous SOC → discretization or breakpoints) |
| Exactness | Exact only under a proved breakpoint property |
| SynthCharge linear energy/charge | May admit piecewise structure; **not proved here** |
| Time windows | Expand state; curse of dimensionality |
| Continuous charge | Needs SOC discretization or analytical breakpoints |
| Runtime on 180 | Uncertain without implementation + scaling study |

**Verdict:** promising research direction; **not** ready to declare equivalence.

---

## Candidate: MILP

| Item | Assessment |
|------|------------|
| Formulation | Binary station visits / ordering along frozen sequence; SOC / time linear inequalities |
| Exact? | Exact for the formulated relaxation/model |
| Continuous charge? | Yes if SOC variables continuous |
| Differences | Must encode the same shield legality, loop guards, ZERO_CHARGE_NOOP, time cap optimism |
| Runtime | Likely solvable for small \(n\); 180 mixed lengths may need timeouts |

**Verdict:** implementable under a **new** namespace after careful equivalence
spec; prefer V4 predeclared TEST if used confirmatory.

---

## Candidate: beam search / MPC

| Item | Assessment |
|------|------------|
| Exact? | No (heuristic) |
| Continuous charge? | Can sample \(u\) grid or use Max/Min |
| Relation to V3 | Stronger than one-step lookahead if depth/beam large |
| Runtime | Controllable |

**Verdict:** good **development** comparator; if added to confirmatory tables,
requires predeclared V4 protocol (not silent V3 insertion).

---

## Recommendation

| Path | When |
|------|------|
| **A. Development-only comparison now** | Implement beam/MPC or MILP under `src/planning/` / `scripts/v3/development/`; evaluate TRAIN/VAL only |
| **B. Post-hoc V3 analysis** | Only if labeled *POST-HOC EXPLORATORY*; never edit `paired_primary.json` |
| **C. New predeclared V4 TEST** | **Preferred** for strongest confirmatory evidence if a same-problem optimizer is established |

**Default preference: C.**

Do not invent a “strong baseline” and place it in the V3 main table as if
predeclared.

---

## Next engineering step (optional, non-V3)

1. Write a short equivalence checklist (simulator events vs optimizer constraints).
2. Prototype MILP or deep beam on SynthCharge VAL only.
3. If competitive and well-specified, open a V4 protocol draft.
