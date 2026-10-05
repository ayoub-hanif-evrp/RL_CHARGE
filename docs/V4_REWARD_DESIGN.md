# V4 reward design (FA-HPPO)

**Status:** development protocol for a new experimental stage.  
**Frozen V1/V2/V3 artifacts are not modified or reinterpreted.**

Primary operational objective (unchanged):

\[
\min T_{\mathrm{completion}}
\quad\text{among feasible fixed-route charging schedules.}
\]

Hard constraints (battery, min SOC, customer time windows, depot horizon,
capacity, reachability, legal revisits, valid charge targets) remain in the
simulator, shield, and feasibility-aware SOC envelope. A time-window
violation is an **infeasible / terminal failure**, not a soft lateness
penalty in the reward.

---

## 1. Normalization constant \(C_{\mathrm{train}}\)

Computed **once from TRAIN routes only** before learning:

\[
C_{\mathrm{train}}
=
\operatorname{median}_{r\in\mathrm{TRAIN}} H_r,
\qquad
H_r=\text{depot due date / horizon of route }r.
\]

- Never fit on VAL or TEST.
- On SynthCharge TRAIN, all horizons equal \(10\), so
  \(C_{\mathrm{train}}=10\) (median = max).
- Recorded in protocol, manifests, checkpoint metadata, and curves.

Code: `rl.rewards.compute_c_train`.

---

## 2. Reward variants

| Kind | Training reward (feasible) | Failure | PPO `return_scale` |
|------|----------------------------|---------|---------------------|
| `V3_TIME` | \(r=-\Delta t\) | \(-(H-t_0)-L\) | historical TRAIN max \(H\) (V3) |
| `V4_BASE` | \(r=-\Delta t/C_{\mathrm{train}}\) | \([-(H-t_0)-L]/C_{\mathrm{train}}\) | \(1\) (scale inside reward) |
| `V4_PBRS` | \(r=-\Delta t/C+\Phi(s')-\Phi(s)\) | see §4 | \(1\) |
| `V4_BASE_NO_L_FAIL` | same as `V4_BASE` | \(-(H-t_0)/C\) only | \(1\) |

where \(\Delta t=t_{t+1}-t_t\) and
\(L=L_{\mathrm{remaining}}(s)=\texttt{remaining_time_lower_bound(simulator)}\).

Potential (V4_PBRS only, \(\gamma=1\)):

\[
\Phi(s)=-\frac{L_{\mathrm{remaining}}(s)}{C_{\mathrm{train}}},\qquad
\Phi_{\mathrm{absorbing}}=0.
\]

Shaping:

\[
F(s_t,s_{t+1})=\Phi(s_{t+1})-\Phi(s_t).
\]

Boxed V4 proposal:

\[
r_t^{V4}
=
-\frac{\Delta t}{C_{\mathrm{train}}}
+
\Phi(s_{t+1})-\Phi(s_t).
\]

---

## 3. Cumulative-return derivations

Notation: \(C=C_{\mathrm{train}}\), \(L_t=L_{\mathrm{remaining}}(s_t)\),
\(\Phi_t=-L_t/C\), initial state \(s_0\), horizon \(H\).

### 3.1 Successful trajectory (unshaped `V4_BASE`)

\[
G_{\mathrm{success}}^{\mathrm{base}}
=
\sum_t(-\Delta t_t/C)
=
-T_{\mathrm{completion}}/C.
\]

### 3.2 Failed trajectory (unshaped `V4_BASE`)

After feasible prefix reaching time \(t_k\), then fail:

\[
G_{\mathrm{fail}}^{\mathrm{base}}
=
-t_k/C
+
\bigl(-(H-t_k)-L_k\bigr)/C
=
-(H+L_k)/C.
\]

### 3.3 Successful trajectory (shaped `V4_PBRS`)

\[
G_{\mathrm{success}}^{\mathrm{PBRS}}
=
\sum_t\Bigl(-\frac{\Delta t_t}{C}+\Phi_{t+1}-\Phi_t\Bigr)
=
-\frac{T}{C}+\Phi_N-\Phi_0.
\]

Completed / absorbing: \(L_N=0\Rightarrow\Phi_N=0\), so

\[
G_{\mathrm{success}}^{\mathrm{PBRS}}
=
\frac{-T+L_0}{C}.
\]

Since \(T\ge L_0\), \(G_{\mathrm{success}}^{\mathrm{PBRS}}\le 0\). Maximizing \(G\) is
equivalent to minimizing \(T\) (constant \(L_0/C\)).

### 3.4 Failed trajectory (shaped `V4_PBRS`, no double-count of \(L\))

**Base failure term omits \(L\)** (avoids counting \(L\) twice with shaping):

\[
r_{\mathrm{fail}}^{\mathrm{base}}=-(H-t_k)/C,\qquad
F=\Phi_{\mathrm{abs}}-\Phi_k=-\Phi_k=L_k/C.
\]

Prefix + fail:

\[
G_{\mathrm{fail}}^{\mathrm{PBRS}}
=
\frac{-t_k+L_0-L_k}{C}
+
\frac{-(H-t_k)+L_k}{C}
=
\frac{L_0-H}{C}.
\]

\(L_k\) cancels. All failures on a route share return \((L_0-H)/C\).

### 3.5 Feasible completions dominate failures (`V4_PBRS`)

For any feasible completion with \(T\le H\):

\[
G_{\mathrm{success}}^{\mathrm{PBRS}}-G_{\mathrm{fail}}^{\mathrm{PBRS}}
=
\frac{-T+L_0-L_0+H}{C}
=
\frac{H-T}{C}\ge 0.
\]

Equality only if \(T=H\). Hence every strictly early feasible completion beats
every failure on the same route.

### 3.6 Failure ablation `V4_BASE_NO_L_FAIL`

\[
G_{\mathrm{fail}}^{\mathrm{noL}}= -H/C.
\]

Still \(G_{\mathrm{success}}=-T/C\ge -H/C\), but failures no longer depend on
remaining progress \(L_k\).

**Exact-horizon boundary.** If a successful route finishes at \(T=H\), then
\(G_{\mathrm{success}}=G_{\mathrm{fail}}=-H/C\). Empirically, no successful
TRAIN/VAL evaluation under the development checkpoints satisfies
\(|T-H|<10^{-6}\) (see `results/v4_reward/analysis/EXACT_HORIZON_AUDIT.json`).
This remains a theoretical boundary; no epsilon penalty is added.

---

## 4. What we deliberately do **not** put in the reward

| Temptation | Why omitted |
|------------|-------------|
| Soft lateness weights | Hard TW: late service ⇒ infeasible |
| Explicit distance penalty | Travel time already in \(\Delta t\) under SynthCharge kinematics |
| \(-\lambda N_{\mathrm{stations}}\) | Detours/charging already increase \(\Delta t\) |
| Terminal-SOC / energy weights | No universal “wasted SOC”; report as metrics |
| Success/fail constants \(\pm 100\) | Dominated by horizon-derived construction above |
| Arbitrary \(w_1 d+w_2 E+\cdots\) | Not scientifically identified |

Distance, energy charged, station visits, terminal SOC remain **evaluation
metrics**, not primary objectives.

---

## 5. PBRS ablation status (development result)

`V4_PBRS` is an **ablation**, not the selected reward.

On the completed SynthCharge TRAIN/VAL 5-seed matrix:

- mean parent-balanced VAL feasibility did **not** improve vs `V4_BASE` /
  `V3_TIME` (97.8% vs 98.2%);
- the correct unshaped comparator for PBRS is **`V4_BASE_NO_L_FAIL`**
  (same base failure without \(L\), plus shaping);
- the negative/neutral PBRS result remains reported and is not overwritten.

No new reward weights were introduced after seeing these VAL numbers.

## 6. Selected development reward

Under the **development selection rule**
(max mean parent-balanced VAL feasibility, then min mean failure-retaining completion):

**`V4_BASE_NO_L_FAIL`** — normalized time-horizon reward:

\[
r_t=-\Delta t/C_{\mathrm{train}},\qquad
r_{\mathrm{fail}}=-(H-t)/C_{\mathrm{train}}.
\]

With \(C_{\mathrm{train}}=10\) on SynthCharge.

`V4_BASE_NO_L_FAIL` matched the best mean development feasibility and obtained the
lowest mean failure-retaining completion among the tied variants. Effects across
seeds were variable, so the reward is selected primarily for its simple
objective-consistent formulation rather than claimed as a large performance
improvement.

## 7. Experimental isolation

First V4 reward study uses **frozen V3 architecture** (FA-HPPO features /
Beta head / envelope as in V3 paper method) and only changes the reward kind.

Do **not** mix with V4-A feature fix, V4-B endpoint head, or new planners
until the reward is selected on TRAIN/VAL and frozen.

No V3 TEST consumption for reward selection. A fresh V4 TEST requires a
separate committed protocol after freeze.

---

## 8. Artifact namespace

```
results/v4_reward/
results/v4_reward/final_clean/
checkpoints_v4/reward_ablation/
checkpoints_v4/final_reward/
docs/V4_REWARD_DESIGN.md
scripts/v4_reward/
scripts/paper/build_v4_training_figures.py
```

V3 paths under `results/v3_hppo/`, `checkpoints_v2/final/`, `checkpoints_v3/`
remain read-only evidence.

## 9. Reviewer-defensible methodology statement

FA-HPPO uses an objective-aligned time reward rather than a weighted mixture of
arbitrary penalties. Feasible transitions incur normalized elapsed-time cost,
while failure maps the episode to the route horizon. Hard EV and time-window
requirements are enforced structurally. Distance contributes through travel
time, while energy charged, charging stops, and terminal SOC are reported
separately as operational metrics.
