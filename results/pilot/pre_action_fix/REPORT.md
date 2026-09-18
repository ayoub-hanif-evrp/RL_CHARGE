# Failed pre-paper Hybrid PPO pilot — root-cause audit

**Diagnostic only.** TEST was not used. PPO was not retrained. Reward, lr,
entropy, architecture, corpus, and splits were not changed.

Archive: `results/pilot/pre_action_fix/`  
Route-level CSV: `baseline_route_level.csv`  
Machine summary: `audit_summary.json`

---

## 1. Deterministic TRAIN feasibility

132 routes. Completion times are **feasible routes only**.

| Method | Feasible | Rate | Mean / median completion (feas.) | Mean / median station visits | Consecutive same-station (sum) | Loop-guard routes |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| GreedyMinimumSufficientCharge | 3 / 132 | 2.3% | 196.5 / 187.9 | 14.9 / 1 | 0 | 37 |
| GreedyFullCharge | 8 / 132 | 6.1% | 954.4 / 451.9 | 20.7 / 3 | 0 | 52 |
| OneStepLookahead | 9 / 132 | 6.8% | 257.7 / 189.4 | 30.7 / 50 | 865 | 76 |

Greedy min/full never repeat the same station (they skip zero-ΔSOC
same-station actions). Lookahead often does, and its median visit count is
the programming loop-guard (50).

`MASK_VIOLATION` on greedy-min (6 TRAIN, 3 VAL) is a **baseline chooser
artifact**: when CONTINUE is illegal and every remaining station is a
zero-charge no-op, `_greedy_choose` returns CONTINUE anyway. Not a PPO
change and not an exact infeasibility label.

---

## 2. Deterministic VALIDATION feasibility

77 routes.

| Method | Feasible | Rate | Mean / median completion (feas.) | Mean / median station visits | Consecutive same-station (sum) | Loop-guard routes |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| GreedyMinimumSufficientCharge | 20 / 77 | 26.0% | 178.1 / 176.9 | 11.1 / 1 | 0 | 16 |
| GreedyFullCharge | 18 / 77 | 23.4% | 183.4 / 184.3 | 18.9 / 1 | 0 | 28 |
| OneStepLookahead | 12 / 77 | 15.6% | 170.5 / 164.9 | 29.9 / 50 | 388 | 44 |

VAL is **much easier for greedy** than TRAIN. Seed 43’s peak 16/77 (20.8%)
is in the same band as greedy-full, not above it. Seed 42’s 0/77 is
**worse** than all three baselines.

---

## 3. Union-of-baselines (diagnostic lower bound only)

| Split | Routes | Completed by ≥1 of the 3 | Union rate |
| --- | ---: | ---: | ---: |
| TRAIN | 132 | 14 | **10.6%** |
| VALIDATION | 77 | 22 | **28.6%** |

This is **not** an exact feasibility oracle. Routes outside the union are
**not** claimed infeasible. A better charger than these three heuristics
could still succeed. It is only a lower bound on “easy” routes.

TRAIN being ~4× harder than VAL for these heuristics is a real training
problem: PPO samples TRAIN, where even greedy almost never finishes.

---

## 4. Failure-reason distributions

Dominant failure for the heuristics is **`NO_FEASIBLE_ACTION`** (shield
dead-end: no legal CONTINUE and no legal station). Secondary:
**`ZERO_CHARGE_NOOP`**. Loop-guard (50 same-customer station hops) is hit
often by lookahead and by greedy-full on TRAIN.

PPO checkpoints (below) die on the same two reasons, with collapse
concentrating on `ZERO_CHARGE_NOOP`.

---

## 5. Station-action cardinality

States: every TRAIN/VAL reset plus up to 8 greedy-min steps (828 TRAIN,
448 VAL). TEST unused.

Instance station count **m**:

| Split | m min / mean / max | Typical |
| --- | --- | --- |
| TRAIN | 2 / 10.17 / 21 | many Small (3–7) and Large (`m=21`, 288 states) |
| VAL | 4 / 10.46 / 21 | 4, 5, 8, 21 |

Legal unmasked stations (after the shield): mean **≈4.8** on both splits
(range 0–21). CONTINUE is legal in **52.5%** of TRAIN states and **55.6%**
of VAL states.

**Structural flat categorical** over `{CONTINUE, S1, …, Sm}` (no mask):

```
P(CONTINUE) = 1/(m+1)
P(CHARGE)   = m/(m+1)
```

| Split | Mean P(CONTINUE) unmasked | Mean P(CHARGE) unmasked |
| --- | ---: | ---: |
| TRAIN | 0.143 | 0.857 |
| VAL | 0.126 | 0.874 |

For Large_Network `m=21`: **P(CONTINUE)=1/22 ≈ 4.5%**, **P(CHARGE)≈95.5%**
if logits are equal and the mask is ignored.

The **implemented** policy does mask illegal bits (`-1e9`). Uniform over
*legal* actions is `1 / (1[CONTINUE]+n_legal_stations)`, which is less
extreme when few stations are legal, but still charge-heavy whenever many
stations remain unmasked.

---

## 6. Untrained vs failed-pilot CONTINUE vs CHARGE mass

Softmax of **masked** logits on reset + subsampled greedy states.
Untrained = fresh `HybridPolicy` (d_model 64), seed 0.

| Policy | Split | Mean P(CONTINUE) | Mean P(CHARGE) | Fraction P(CHARGE)>0.8 |
| --- | --- | ---: | ---: | ---: |
| Untrained | TRAIN | 0.665 | 0.335 | 0.29 |
| Untrained | VAL | 0.563 | 0.437 | 0.38 |
| Seed 42 best | TRAIN | 0.145 | 0.855 | 0.65 |
| Seed 42 best | VAL | 0.111 | 0.889 | 0.83 |
| Seed 43 best | TRAIN | 0.479 | 0.521 | 0.29 |
| Seed 43 best | VAL | 0.399 | 0.601 | 0.38 |
| Seed 43 last | TRAIN | 0.112 | 0.888 | 0.84 |
| Seed 43 last | VAL | **0.083** | **0.917** | **0.91** |

Initialization does **not** start as a charger: masked untrained mass
prefers CONTINUE (separate continue vs station heads; many stations
masked). Collapse toward CHARGE is **learned**. Seed 43 best is the only
checkpoint that still puts substantial mass on CONTINUE; `last.pt` wipes
it out.

---

## 7. Same-station repeats

### Semantics (not changed)

One `ChargeAction(station_id, target_soc)` already does:

1. travel current → station (no-op if already there)
2. linear charge `dt = g * ΔE` to `target_soc`

So choosing the **same station again without moving** only adds energy if
the previous target left SOC below max. Under this linear model that is
**never scientifically required**: the first action’s `u=1` already maps
to `soc_upper` (usually max SOC). Repeats consume the loop-guard counter
(50). After SOC is maxed, the shield/simulator reject the next same-station
action as `ZERO_CHARGE_NOOP`.

Greedy min/full skip those no-ops (0 consecutive repeats). PPO does not.

### Quantities

| Checkpoint | Split | Feasible | Consecutive same-station (sum / mean) | Energy added by repeats | Time added by repeats | Failures | CONTINUE action fraction |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| Seed 42 best | TRAIN | 0/132 | 1445 / 10.9 | 5528 | 9012 | 97 dead-end, 35 `ZERO_CHARGE_NOOP` | 0.7% |
| Seed 42 best | VAL | 0/77 | 789 / 10.2 | 3182 | 3507 | 48 dead-end, 29 `ZERO_CHARGE_NOOP` | 0.1% |
| Seed 43 best | TRAIN | 2/132 | 2092 / 15.8 | 1029 | 1548 | 77 `ZERO_CHARGE_NOOP`, 53 dead-end | 15.2% |
| Seed 43 best | VAL | 16/77 | 943 / 12.2 | 314 | 372 | 36 `ZERO_CHARGE_NOOP`, 25 dead-end | 18.9% |
| Seed 43 last | TRAIN | 0/132 | 2139 / 16.2 | 758 | 1377 | **132/132 `ZERO_CHARGE_NOOP`** | **0%** |
| Seed 43 last | VAL | 0/77 | 1519 / 19.7 | 543 | 464 | **77/77 `ZERO_CHARGE_NOOP`** | **0%** |

Collapsed `last.pt`: never CONTINUE; sit at a station; drip-charge with
Beta mean `u≈0.5` until max; die on `ZERO_CHARGE_NOOP`.

---

## 8. Reward invariance: `G_failure = -H`

Initial time is 0. Feasible steps sum to `-t_fail`. Terminal failure adds
`-(H - t_fail)`. Therefore

```
G_failure = -t_fail - (H - t_fail) = -H
```

**Confirmed on 36/36 failed TRAIN rollouts** (greedy-min, random-legal,
always-first-station), `max |G+H| = 0`.

Examples on `c102_50_21_NL` (`H=1236`):

| Policy | Steps | Customers served (index) | G | Equals -H? |
| --- | ---: | ---: | ---: | --- |
| greedy_min | 53 | 3 | -1236 | yes |
| random_legal | 50 | 0 | -1236 | yes |
| always station u=0.5 | 0 | 0 | -1236 | yes |
| greedy_min (other vehicle) | 58 | 6 | -1236 | yes |

Serving six customers then dying is **identical** in return to dying at
the depot on the first action. With TRAIN greedy union ≈10% feasible,
almost every PPO episode is this indistinguishable `-H`. That is a sparse
learning signal.

---

## 9. Diagnosis (most likely cause of PPO collapse)

**Primary mechanism of collapse:** same-station charging remains legal
after a partial charge. Hybrid PPO’s Beta mean sits near `u≈0.5`, so each
`ChargeAction` only fills part of `[soc_lower, max]`. The agent can
(and does) select the same station again, adding a little energy and a
little time, until the battery is full and the next action is
`ZERO_CHARGE_NOOP`. Seed 43 `last.pt` is this policy in pure form:
0% CONTINUE, 100% `ZERO_CHARGE_NOOP`.

That is possible because, under **linear** charging, travel+charge is
already one action; repeats are redundant, not necessary, but **not
masked** unless ΔSOC is ~0.

**Why it is learned rather than an init bug:** an untrained masked
softmax still prefers CONTINUE. Training inverts that.

**Why it does not recover:** every failed trajectory on a route has
`G=-H`, so long charge-loops are not ranked worse than “almost finished”.
The categorical `{CONTINUE}∪{S1..Sm}` prior is also charge-heavy on
Large instances (`m=21` → 95% charge if logits are flat and unmasked).

**Contributing difficulty, not the collapse mechanism itself:** TRAIN is
a harsh population for even greedy (union 14/132). VAL is easier (22/77).
A method that never matches greedy-min on TRAIN cannot be expected to
hold a VAL spike (seed 43 at update 200) against later collapse.

**Not claimed:** that routes outside the 14/22 union are infeasible;
that the shield is an exact oracle; that TEST would look like VAL.

No method change was made. STOP.
