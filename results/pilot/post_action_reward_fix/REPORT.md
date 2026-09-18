# Second Hybrid PPO TRAIN/VAL pilot (post action/reward fix)

TEST was not used. Corpus, 20/6/6 split, physics, and PPO hyperparameters
were not changed. The failed first pilot remains in
`results/pilot/pre_action_fix/`.

**Decision: STOP. Do not freeze a paper `budget_updates`. Do not run
seeds 44–46, TEST, or final ablations.**

Seed 42 now has a real VAL learning plateau. Seed 43 does not share it.
Choosing 40, 160, or 200 would be a seed-specific spike or a one-seed
plateau, which the protocol forbids.

Do not automatically factor the action hierarchy, change the learning
rate, add curriculum, change the corpus, or change the reward again.

---

## What was corrected

1. **Station revisits** between two frozen customers are masked
   (`STATION_REVISIT`). Same-station drip charging and `S1→S2→S1` cycles
   are illegal. The programming loop-guard stays a backstop.
   `ZERO_CHARGE_NOOP` remains a first-visit backstop.
2. **Failure progress:** `r_fail = -(H - t) - L_remaining`.
   `L_remaining` is a time-unit lower bound on remaining frozen-route
   travel + service (no charging, no waiting). Success is still
   `G = -completion_time`. This is a horizon-derived failure-progress
   term, not a second operational objective.
3. **VAL checkpoint selection** is lexicographic on
   parent-balanced feasibility, then parent-balanced all-routes
   completion. Route-weighted VAL numbers stay in the logs. The 20/6/6
   split is unchanged.
4. Duplicate first-pilot files outside `pre_action_fix/` were removed.

Tests: **186 passed**.

---

## Deterministic TRAIN/VAL diagnostics (after the fix)

Not an exact feasibility oracle. Union is a lower bound on “easy” routes.

### TRAIN (132)

| Method | Feas. | Rate | Parent-bal. | Visits | Revisits | Loop-guard | ZERO_CHARGE_NOOP | NO_FEASIBLE_ACTION |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy min | 3 | 2.3% | 2.1% | 2.30 | 0 | 0 | 22 | 107 |
| Greedy full | 8 | 6.1% | 7.5% | 4.20 | 0 | 0 | 33 | 91 |
| Lookahead | 9 | 6.8% | 6.1% | 6.00 | 0 | 0 | 0 | 123 |

Union: **14 / 132 = 10.6%** (unchanged).

### VALIDATION (77)

| Method | Feas. | Rate | Parent-bal. | Visits | Revisits | Loop-guard | ZERO_CHARGE_NOOP | NO_FEASIBLE_ACTION |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy min | 21 | 27.3% | 9.7% | 1.79 | 0 | 0 | 9 | 47 |
| Greedy full | 18 | 23.4% | 8.5% | 2.74 | 0 | 0 | 23 | 36 |
| Lookahead | 12 | 15.6% | 5.3% | 5.27 | 0 | 0 | 0 | 65 |

Union: **23 / 77 = 29.9%** (was 22 / 77).

Pre-fix loop-guard hits were 37–76 on TRAIN and 16–44 on VAL, with
lookahead median visits = 50. After the revisit mask: **0 revisits,
0 loop-guard** for all three methods. Mean visits fell from ~15–31 to
~2–6. Legal baseline behavior no longer cycles stations.

Parent-balanced VAL is much lower than route-weighted VAL because `r102`
is 41 of 77 routes. Greedy-min parent-balanced VAL is **9.7%** vs
route-weighted **27.3%**. That is why checkpoint selection now averages
parents, not routes.

Greedy `ZERO_CHARGE_NOOP` is a first-visit target = arrival-SOC backstop,
not a reselection loop.

---

## PRE-FIX vs POST-FIX Hybrid PPO

Pre-fix parent-balanced VAL was not logged (selection was route-weighted).
Route-weighted VAL is the apples-to-apples trajectory comparison.
Post-fix selection uses parent-balanced metrics.

### Seed 42

| | PRE | POST |
| --- | ---: | ---: |
| Status | early_stop @ 200 | early_stop @ 360 |
| Best update | 0 | **160** |
| Best route-weighted VAL feas. | **0 / 77 (0%)** | **21 / 77 (27.3%)** |
| Best parent-balanced VAL feas. | (not logged; 0 throughout) | **9.7%** |
| Best parent-balanced completion | — | 1170.78 |
| CONTINUE fraction (best) | 0.14% | **42.5%** |
| CHARGE fraction (best) | 99.9% | **57.5%** |
| target-u mean / sd | 0.483 / 0.012 | 0.478 / 0.028 |
| Mean station visits | 37.8 | **4.79** |
| Repeated-station actions | 804 | **0** |
| Loop-guard hits | (audit: common) | **0** |
| ZERO_CHARGE_NOOP | 29 | **3** |
| NO_FEASIBLE_ACTION | 48 | 53 |
| Loss finite / NaN | finite / no | finite / no |
| Runtime | 2318 s | 3971 s |

POST VAL route-weighted trajectory (eval every 10):

`0, 0, 0.039, 0.182, 0.182, 0.247, 0.208, 0.182, 0.221, 0.195, 0.221, 0.221, 0.195, 0.247, 0.260, 0.273, 0.273, 0.260, 0.260, 0.260, 0.260, …` then a slow decline to 0.104 at update 360.

Parent-balanced stays in **6–10%** from update 30 through 200 (peak 9.7%
at 150–160). Last.pt is still 8/77, not a ZERO_CHARGE_NOOP collapse.

This is a **stable region**, not a one-point spike. Best parent-balanced
VAL equals greedy-min on this split (21/77, 9.7%). Hybrid PPO did not
beat the heuristic; it reached it.

### Seed 43

| | PRE | POST |
| --- | ---: | ---: |
| Status | early_stop @ 399 | early_stop @ 240 |
| Best update | 200 | **40** |
| Best route-weighted VAL feas. | 16 / 77 (20.8%) | 14 / 77 (18.2%) |
| Best parent-balanced VAL feas. | (not logged) | **6.5%** |
| CONTINUE fraction (best) | 18.4% | **36.5%** |
| CHARGE fraction (best) | 81.6% | **63.5%** |
| target-u mean / sd | 0.539 / 0.018 | 0.506 / 0.032 |
| Mean station visits | 13.6 | **5.34** |
| Repeated-station actions | 982 | **0** |
| Loop-guard hits | — | **0** |
| ZERO_CHARGE_NOOP (best / last) | 36 / **77** | **3 / 0** |
| NO_FEASIBLE_ACTION (best / last) | 25 / 0 | 60 / 64 |
| Loss finite / NaN | finite / no | finite / no |
| Runtime | 4295 s | 2722 s |

POST parent-balanced VAL:

- 0 through update 30
- 6.5% / 5.7% / 4.8% at 40 / 50 / 60
- **0 from 70 through 190**
- 5.2% at 200, 0 at 210–220, 2.4% at 230, 6.1% at 240 (last)

Last.pt is 13/77 with 0 ZERO_CHARGE_NOOP (pre-fix last was 0/77, 100%
CHARGE, 77 ZERO_CHARGE_NOOP). The drip-charge death is gone. The
**learning trajectory is still not a shared stable region**: a
three-eval bump at 40–60, a long zero stretch, then a late noisy
recovery. That is not enough to freeze a budget.

---

## Shared-budget test

| Candidate | Seed 42 | Seed 43 |
| --- | --- | --- |
| 40 | rising, not yet peak | **best**, then collapse to 0 |
| 160 | **best plateau** | 0 |
| 200 | still ~26% / 9.3% parent-bal. | one-point 14% then 0 |
| 300 / 400 | 42 already early-stopped in decline | 43 already early-stopped |

No shared TRAIN/VAL region. **No paper budget.**

---

## Side effect worth recording

TRAIN greedy-min dynamic normalizer states: 2402 (pre) → 742 (post).
Greedy-min no longer walks 50-visit loops, so the TRAIN-only normalizer
sees shorter trajectories. Hyperparameters were not changed.

---

## Stop rule (item 8)

The action/reward correction is **not a failure**: station cycles are
gone, CONTINUE is used, seed 42 matches greedy-min VAL, and seed 43 no
longer dies on 77/77 ZERO_CHARGE_NOOP.

It is **not a success under the freeze rule**: both seeds must show a
reasonably stable TRAIN/VAL region. Seed 43 does not.

Do **not**:
- factor the action hierarchy
- change learning rate or entropy
- add curriculum
- change the route corpus
- change the reward again
- run TEST or five-seed paper experiments

Machine summary: `pilot_comparison.json`.
