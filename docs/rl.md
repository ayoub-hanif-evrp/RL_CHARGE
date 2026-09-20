# Reinforcement learning (Part 3A)

The method is **one Hybrid PPO** with a shared encoder and a single optimizer.
There is no KNN+GCN+CNN stack and no pair of DDQNs owning one encoder.

## MDP

- **State:** simulator state, frozen remaining customer sequence, station
  candidate features. Station attention uses the shield legality mask, so
  visited/illegal stations contribute no attention mass. See
  `src/rl/features.py`.
- **Action:** discrete `{CONTINUE} ∪ stations`. If a station is chosen,
  continuous `u ∈ (0, 1)` is mapped through the shield SOC interval.
- **Transition:** `FixedRouteSimulator`. Customer order is immutable.
- **Discount:** `γ = 1`.
- **Horizon:** depot due date `H`.
- **Success:** return to depot, all customers served.
- **Failure:** the shield leaves no action, or a transition is infeasible.

**Eval objective:** minimize `route_completion_time`. Energy and distance are
reported separately and never mixed into a scalar cost.

## Reward

Feasible step: `r = -(t' - t)` (travel, wait, service, charge).

Successful episode: `G = -route_completion_time`.

Terminal failure: `r_fail = -(H - t0) - L_remaining(failure_state)`, so
`G_failure = -H - L_remaining(state_failure)` from initial time zero.
`t0` is the pre-action clock even if the failed transition mutated time.
`L_remaining` is a method-independent lower bound in time units on the
remaining frozen-route travel and service. It is **not** a second reported
objective. Among failures, more route progress (smaller `L_remaining`) is
preferred. Failures remain strictly worse than a feasible completion unless
`L_remaining = 0`.

## Policy

```
encoder(s) → h
  Transformer over the frozen remaining route
  candidate attention over stations
discrete head: logits over CONTINUE+stations, softmax over unmasked actions
if station: Beta(α, β) with α, β = softplus(MLP([h; station_embed])) + 1
value V(s)
```

`log π = log π_disc + 1[station] log π_Beta(u)`.

Eval mode: greedy discrete action and the Beta mean (deterministic).

Feature normalization is **fit on all TRAIN routes** (reset features) plus a
TRAIN-only greedy-min dynamic pass. Validation and test statistics never enter
the fit. Provenance (`n_routes`, `n_dynamic_states`, seed, git SHA) is stored
in the checkpoint.

Model selection on validation is lexicographic on **parent-balanced**
metrics: maximize mean-over-parents feasibility, then minimize mean-over-parents
all-routes completion (`H` for failures). Route-weighted VAL numbers stay in
the logs. Paper/pilot evaluate the full validation population.

## Training budget

Paper Hybrid PPO: `configs/rl/hybrid_ppo.toml` (`budget_updates=200`,
`rollout_steps=256`, val every 10 updates, patience 8). Smoke:
`configs/rl/hybrid_ppo_smoke.toml`. `scripts/train_ppo.py` remains a CPU
smoke helper. Paper training is `scripts/train_rl.py`.

Seeds cover Python, NumPy, PyTorch, and hierarchical episode sampling
(`parent → vehicle → terrain`). Paper seeds are `[42, 43, 44, 45, 46]`.

## AttentionPPO (architectural, not a reproduction)

`AttentionPPO` adds a HetGAT-style node-type embedding (depot / customer /
station) on the same frozen-route Hybrid PPO actor. Public joint
routing+charging DRL papers are **not** reproduced. See `docs/experiments.md`.

## Baselines (new simulator only)

| Name | Behaviour |
| --- | --- |
| `GreedyMinimumSufficientCharge` | CONTINUE if legal; else min-detour reachable station to `soc_lower`. |
| `GreedyFullCharge` | Same station rule; `soc_upper`. |
| `OneStepLookahead` | Score CONTINUE vs each unmasked station+endpoint SOC by time. |
| `DiscretePPO` | Same encoder; **categorical** station action plus six charge levels `{0,0.2,...,1}` conditioned on the chosen station. No Beta sampling or snapping. Simulator stays continuous. |
| `LegacyTwoStageDDQN` | Same features/encoder; two discrete heads; six SOC levels `0.5…1.0` **conditioned on the selected station**; one optimizer; frozen target encoder **and** both target heads. Double DQN: online selects, target evaluates. TRAIN for learning, VALIDATION for checkpoints. This is the old *method*, not the old environment. |

None of these use `check_charging_needed` or `MAX_RL_STOPS`.

frvcpy is an optional adapter. The EVRPTW-GR export is labeled `not_equivalent`.
It is **not** exact for payload-dependent directed energy plus customer TWs.
