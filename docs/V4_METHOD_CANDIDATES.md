# V4 method candidates (not implemented)

These candidates are **design notes only**. None are implemented in the V3
freeze. Implementing any of them requires a new method identifier, retraining,
new checkpoints, a fresh untouched TEST, and no reuse of the consumed V3 TEST
for model-selection claims.

---

## C1. Correct station-arrival SOC feature

### Problem (frozen V3)

In `_station_features`, index 6 is assigned `arrival = interval.soc_lower`,
duplicating index 9. The physical arrival SOC from `_arrival_soc` is never
exposed as a distinct feature (see `docs/FROZEN_METHOD_NOTES.md`).

### Proposed corrected representation

Keep `STATION_DIM` meaningful (possibly still 12, or explicitly 13 if needed)
with **distinct** semantics:

| Feature | Definition |
|---------|------------|
| actual predicted arrival SOC | `_arrival_soc(sim, station)` |
| feasibility-aware continuation lower departure SOC | energy-continuation lower (without overwriting arrival) |
| time-aware upper SOC | optimistic time-feasibility upper (when enabled) |

Plus the existing geometric/energy/detour/slack channels.

### Requirements if adopted

- new method identifier (e.g. FA-HPPO-V4 / HybridPPO-V4);
- retrain all seeds;
- new checkpoint freeze;
- fresh independently generated / locked TEST;
- **no** reuse of V3 TEST for selection or retuning claims.

---

## C2. Endpoint-capable continuous amount parameterizations

Frozen V3 uses \(\alpha,\beta > 1\) via `softplus + 1`, eval mean
\(u=\alpha/(\alpha+\beta)\), with sample clamp away from exact \(\{0,1\}\`.

Candidates (none claimed superior a priori):

1. Beta without forced `+1` where numerically stable;
2. mixture / hurdle model with explicit boundary masses at \(u=0\) and \(u=1\);
3. alternative bounded continuous distributions (e.g. squashed Gaussian).

Motivation: FA-HPPO-Max ≈ free FA-HPPO on V3 suggests the envelope upper bound
is highly competitive; endpoint-capable heads are a reasonable research
direction. Do **not** assert that C2 would have changed V3 confirmatory
outcomes without a new protocol.

---

## C3. Stronger same-problem planning comparator

See `docs/STRONG_COMPARATOR_FEASIBILITY_STUDY.md`.

If a mathematically equivalent optimizer exists for SynthCharge fixed-route
charging under V3 assumptions, the preferred confirmatory path is:

> **C. new predeclared V4 TEST** including that comparator in the primary family.

Do not silently insert a post-hoc optimizer into the consumed V3 confirmatory
family.

---

## C4. Broader / nonlinear benchmark

Optional future protocol:

- OOD or non-linear energy / heterogeneous chargers;
- do not overclaim SynthCharge linear energy as EVRPTW-GR terrain validation.

---

## Title note (no mass rename)

| Option | Emphasis |
|--------|----------|
| *Feasibility-Aware Hybrid PPO for Fixed-Route Electric Vehicle Charging* | Architecture (hybrid discrete/continuous PPO) |
| *Feasibility-Aware PPO for Fixed-Route Electric Vehicle Charging* | Contribution (feasibility envelope + PPO control) |

**Recommendation:** prefer the second for the manuscript title if space is
tight, because V3 evidence attributes most performance to the
feasibility-aware envelope rather than fine continuous amount control. Keep
“Hybrid” in the method name (FA-HPPO) in the body. Do **not** mass-rename
code/protocol artifacts for a title change.
