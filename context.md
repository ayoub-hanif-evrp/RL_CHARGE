# Project context — RL_CHARGE / EVRPTW-GR fixed-route charging

This file is the **handoff document for other code agents**. Read it before
editing, training, or “improving” anything. It is a lab briefing, not the
paper manuscript.

If this file disagrees with older `docs/*.md` comments, **this file plus the
pinned hashes and the frozen SHA** win for the current stage.

---

## 0. Read this first if you are another agent

**Scientific goal.** Learn a charging policy for an electric vehicle whose
**customer visit order is already frozen**. The agent may insert charging-station
stops and choose how much to charge. It must **never** reorder customers, never
solve joint routing+charging, and never invent a mixed-unit cost.

**Proposed method (only one).** Hybrid PPO for fixed-route charging:
WHEN + WHERE + HOW MUCH. DiscretePPO is a comparator (categorical charge
amount), not a second proposed method. AttentionPPO is an optional
architectural comparator. **Legacy DDQN is not part of the paper.**

**Current stage (21 Sep 2026).** Methodology, simulator, shield, Hybrid PPO,
and the TRAIN/VAL pilot protocol are **frozen**. The third Hybrid PPO
TRAIN/VAL pilot succeeded sufficiently to freeze:

- max updates 400
- rollout steps 256
- validation every 10
- patience 20
- per-seed best parent-balanced validation checkpoint

Seeds 42 and 43 of that Hybrid PPO pilot are archived under
`results/pilot/post_correctness_fix/`. TEST has **not** been used for final
evaluation. The five-seed paper matrix, ablations, and manuscript have
**not** started. Do not tune against VALIDATION further. No more
methodology development unless a verified implementation bug occurs.

**Default stance.** Do **not** redesign the MDP, Hybrid PPO, shield, corpus,
splits, physics, or reward. Do **not** regenerate frozen routes. Do **not**
touch TEST until the user starts the final experiment phase. Do **not**
train, evaluate, or tabulate DDQN for the paper. If something looks broken,
**stop and report**.

**Repo.** https://github.com/ayoub-hanif-evrp/RL_CHARGE.git  
**Package.** `rl-charge`, `src/` layout, Python **≥ 3.11**  
**Branch.** `main`  
**Methodology freeze SHA (historical).** `a40b5853f25308ac13af94f308fe8fcdfb258f21`  
**Paper-code freeze (`paper_code_sha`).** `a175ee43548a5d2d5154a9ae0731a01f7642a7f6`  
All final training and evaluation must start from this SHA.  
`results/summaries/experiment_freeze.json` records that `paper_code_sha`. A
later bookkeeping commit of that JSON is `snapshot_commit_sha` and is **not**
the training SHA.

**User constraints that survive across chats.** They are hard, not suggestions:

1. Do not redesign MDP / Hybrid PPO / corpus / splits.
2. Do not regenerate frozen routes.
3. Do not use TEST for tuning, budget, or method decisions.
4. Do not write the manuscript until results exist.
5. Do not start the 5-seed paper experiment until the user approves.
6. Keep pinned corpus and split hashes (section 3) unchanged.
7. Do not include Legacy DDQN in paper training, TEST evaluation, tables,
   or manuscript experiment plans.
8. If a verified implementation bug appears: stop and report; do not
   silently retune lr / entropy / architecture / shield / reward.

---

## 1. What this project is about

### 1.1 The application

An electric delivery vehicle must serve a sequence of customers with **time
windows**, **payload**, and a **limited battery**. The road network has
**altitude**, so energy is **directional** (uphill costs more than downhill)
and **payload-dependent**. Charging stations exist; charging takes time.

The public dataset is **EVRPTW-GR** (Electric Vehicle Routing Problem with
Time Windows given altitude / gradient information):

- DOI: [10.17632/srfdbp2twv.1](https://doi.org/10.17632/srfdbp2twv.1)
- Authors: Rastani, Keskin, Yüksel, Çatay
- Companion paper: Rastani et al., *European Journal of Operational Research*
  (2025/2026)
- Geometry: Schneider, Stenger, Goeke (2014) EVRPTW instances, plus altitude

The canonical description of the file format is
`data/raw/EVRPTW_GR/EVRPTW-GR_Dataset_Description.pdf`.

### 1.2 What is **not** being solved

This is **not**:

- a joint routing+charging RL paper (the agent does not choose the next
  customer)
- a reconstruction of public HetGAT / eTruckRouting / EV-GNN / HeVRPMD papers
- the deleted legacy `Training/` + `Testing/` stack (price-based NS/CS/ES,
  mixed-unit `total_cost`, `check_charging_needed`, `MAX_RL_STOPS`)
- an exact EVRPTW-GR MILP solver
- an exact FRVCP solver on EVRPTW-GR (frvcpy is exact only on **native**
  FRVCP instances)

`AttentionPPO` in this repo is only a **node-type embedding** on the same
frozen-route Hybrid PPO actor. Do not describe it as a reproduction of those
papers.

### 1.3 Why the paper was rebuilt

An earlier manuscript / code stack was rejected. The rebuilt claim is
narrower and cleaner:

> Given **precomputed customer routes**, learn a **Hybrid PPO charging
> policy** that minimizes **route completion time** under EVRPTW-GR physics
> and a method-independent feasibility shield.

Everything else (greedy heuristics, DiscretePPO, AttentionPPO, native frvcpy,
restricted label-setting on tiny routes) is a **baseline, comparator, or
ablation**, not a second proposed method. **Legacy DDQN is not a paper
baseline.**

### 1.4 Two-stage methodology (frozen)

**Stage A — routing (done, frozen).** PyVRP 0.14.0 solves a **customer-only
pickup-VRPTW**. Stations, battery, altitude, and energy are omitted on
purpose. The customer sequences are written once and never edited.

**Stage B — charging (this code).** `FixedRouteSimulator` executes those
sequences. Between two consecutive frozen customers the agent may visit zero
or more stations. Customer order cannot change.

### 1.5 Primary scientific objective

**Route completion time.** Energy, distance, waiting, service, and charge
time are logged **separately**. There is **no** mixed-unit `total_cost`.

Failed episodes stay in every table (`feasible=false`). The all-routes
completion metric substitutes depot due date **H** for failures so dropping
infeasible routes cannot hide losses.

Model selection (TRAIN/VAL): lexicographic **parent-balanced** metrics

1. maximize mean-over-parents validation feasibility
2. then minimize mean-over-parents all-routes completion time (`H` for failures)

Route-weighted VAL numbers stay in the logs. Evaluate **`best.pt`**, never
`last.pt`, on the paper/pilot. Each seed may select its own best validation
checkpoint.

---

## 2. Dataset facts agents must not “fix”

Location: `data/raw/EVRPTW_GR/` — **immutable**. The parser is read-only. Tests
hash the raw tree before and after a full scan. **Nothing may write here.**

124 instance files, **zero** parser errors. Documented issues are **warnings**,
not bugs to correct:

| Issue | Fact |
| --- | --- |
| Incomplete download | Docs imply 156 files (3 terrains × all sizes). Medium and Large have only `Nearly_Level`. 32 files missing. |
| Small_Network capacity | Demands were scaled; `C` was not. 54/108 Small files have a customer demand > `C`. |
| VG altitude rule | 3 files break the 1.5× Nearly_Level rule. |
| Missing curb `M` | Only Medium/Large have `M`. Small leaves it `None`. Official physics uses Model-2 curb 6350 kg, not a silent fill-in of file `M`. |
| Altitude unit | Not documented as metres. Values are O(1). |

Node types: `d` depot, `f` station, `c` customer. Canonical time-window field
is **`due_date`** (raw `DueDate`). Never invent `close_time` / `due_time`.

Small_Network battery `Q ≈ 77.75`. File `r` (consumption rate) is recorded and
**unused**. Energy is never `r * distance`.

Default load convention for RL: `LoadConvention.OFFICIAL_REFERENCE_PICKUP`
(payload starts at 0 and **increases** after customer service). `DELIVERY` is
sensitivity only.

---

## 3. Frozen scientific objects (do not regenerate)

### 3.1 Git SHA for the paper-code freeze

The historical methodology freeze (pre-pilot) was:

```
a40b5853f25308ac13af94f308fe8fcdfb258f21
```

**All final paper training and evaluation must use `paper_code_sha`** from
`results/summaries/experiment_freeze.json` after the paper-scope cleanup is
committed. That SHA is the code version. A later commit that only stores the
preflight JSON is `snapshot_commit_sha` (bookkeeping) and must not be treated
as the training SHA.

Corpus and split hashes below are unchanged and still pinned.

### 3.2 Frozen route corpus

Generated once (`configs/routing/pyvrp.toml`):

- `demand_mapping=pickup`
- `Stop=MaxIterations` (not wall-clock)
- unrestricted fleet with dominating fixed cost `F = 2 * n_customers * Dmax + 1`
- seed 42
- iterations: Small 10k, Medium 20k, Large 40k
- terrain reuse: solve once per parent sibling set, copy customer IDs onto
  other terrains so terrain experiments differ **only** in altitude/energy

Charging feasibility at generation time is **`unverified`**. Routes are **not**
dropped because a future agent might fail to charge them.

| File | SHA-256 (pinned) |
| --- | --- |
| `data/routes/corpus.jsonl` | `1796d817ff058fe0559021ab79ca97087ef3646c57744c2c883ecc37e85451e0` |
| `data/routes/manifest.csv` | `41bb8be3245f8f3b0800db4ba74cb19a2003236b6fbd0dbc1237e7cdfaee104d` |
| `data/routes/corpus_metadata.json` | `1ca8725199842f0a8d4a4e6fac2a31931ea3c9aaca833c2d253c6147469d9a96` |

Counts: **124 instances, 258 routes, seed 42**.

These files are `-text` in `.gitattributes`. Git must **not** rewrite CRLF/LF.
Pinned hashes are of **working-tree bytes**. CI failed once because Windows
CRLF ≠ Linux LF.

### 3.3 Splits (20 / 6 / 6 parents)

Split unit = Solomon–Schneider **parent** `base_instance`. Every size and
terrain of `c101` stays together. Seed **42**.

Algorithm:

1. Stratify by `(customer_distribution, schedule_type)` — six strata.
2. Sort names, `random.Random(42).shuffle`, last → test, second-last → val,
   rest → train.
3. Large_Network repair: swap TEST parent with a same-stratum Large parent
   from VAL if TEST has zero Large files. On this download: `c108` ↔ `c101`.

| File | SHA-256 (pinned) |
| --- | --- |
| `data/splits/train.json` | `3aecd73e9472b8572e899a7e2da3f1f874f0e5a13250f5cd6d76d812002913ce` |
| `data/splits/validation.json` | `1379aef7e92c308f90ab78674dec9994db72078c96cb1020ecc49d33eef69af4` |
| `data/splits/test.json` | `9bf39fbf1f27aa57146879fb71922b0216f24f1eb37a4fd9a9a43b44b6c4b6d0` |
| `data/splits/split_metadata.json` | `4740ab71737bfd9c04d37f258e57bfa4854a526da7c03b218ffe877818128900` |

Expected counts:

- Parents: train 20, validation 6, test 6
- Instances: train 73, validation 30, test 21
- **Routes used by RL:** train **132**, validation **77**

Leakage tests forbid any shared `base_instance`, `instance_id`, or `route_id`
across splits.

**TEST is forbidden** for training, checkpoint selection, hyperparameter
decisions, budget choice, and the current pilot. `scripts/train_rl.py` raises
if `--split test`.

Hashes are also listed in `scripts/preflight_experiments.py` as `PINNED_HASHES`.

---

## 4. Physics (do not invent units)

Typed quantities live in `src/domain/quantities.py`. Subtracting a `Distance`
from a `BatteryEnergy` is a `TypeError` on purpose.

| Quantity | Meaning |
| --- | --- |
| Distance | Planar Euclidean `hypot(Δx, Δy)`. **Not** 3-D slope length. |
| Travel time | `distance / v`. In every published file `v = 1`, so time = distance in Schneider units. |
| Energy | Normalized Schneider units, same space as `Q`. Positive = consumption, negative = regeneration (clipped by SOC ceiling). |
| SOC | `battery_energy / Q`, benchmark bounds `[0, 1]`. |
| Payload | Pickup: starts 0, `+= demand` after service. Stations do not change payload. |
| Gradient | `sin(arctan(Δh / d))` matching EVRPTW-GR Model 2. |

Energy is Demir / Model-2 linearized (`official_evrptwgr` profile):

- curb 6350 kg, cargo cap 3650 kg (from Model 2, **not** Small_Network `C`/`M`)
- empty-flat baseline ≈ 1 energy unit per distance
- **directional:** `energy(i,j,payload)` need not equal `energy(j,i,payload)`
- time windows use file `v`; the kW formula uses 60 km/h only inside the
  physical power expression, then normalizes

Default initial SOC = 1 (overnight full charge). Charging on EVRPTW-GR is
**linear** in time via inverse rate `g`. Piecewise Montoya charging is a
**separate** native-FRVCP / sensitivity profile, not the EVRPTW-GR default.

Configs:

- `configs/physics/official_evrptwgr.toml` — default experiments
- `configs/physics/raw_file.toml` — file `M`/`C` (missing `M` is an error)
- `configs/physics/montoya_piecewise.toml` — native FRVCP sensitivity

---

## 5. Simulator, shield, MDP, methods

### 5.1 `FixedRouteSimulator`

Deterministic physics engine. **No Gym, no torch.**

Actions (`src/simulation/actions.py`):

- `ContinueAction` — travel to the next frozen customer, or depot if done.
  Applies load-dependent directional energy, waiting, service, then the load
  convention. **Advances** the customer index.
- `ChargeAction(station_id, target_soc)` — travel to that station and charge
  continuously to `target_soc ∈ [current_soc, max_soc]`. Does **not** advance
  the customer index. Multiple stations between two customers are legal.

`target_soc` is continuous. There is no six-level grid inside the simulator.
Discrete grids exist only inside comparator **policies** (DiscretePPO).

Failed transitions return `feasible=False` plus an explicit
`InfeasibilityReason` (`INSUFFICIENT_ENERGY`, `TIME_WINDOW_VIOLATION`,
`CAPACITY_VIOLATION`, `INVALID_TARGET_SOC`, `UNKNOWN_STATION`,
`INVALID_STATE`, `LOOP_GUARD`). Silent success is forbidden.

`loop_guard_station_visits` (default 50) is a programming guard against
infinite charge cycles. It is **not** a scientific “max three stops” rule.

Time:

```
arrival       = departure_previous + travel_time
waiting_time  = max(0, ready_time - arrival)
service_start = arrival + waiting_time      # must be <= due_date
departure     = service_start + service_time
```

Station arrival is **not** judged against an invented station time window.
The next `ContinueAction` fails if the subsequent customer window is missed.

### 5.2 Feasibility shield

Method-independent **mask**. It does **not** choose when to charge.

Three labels stay separate:

| Label | Meaning |
| --- | --- |
| `transition_feasible` | Simulator accepted the last action. |
| `shield_mask` | Which discrete actions are legal now. |
| `charging_feasibility_status` | Global existence of a charging schedule. Stays `unverified`. |

CONTINUE is legal iff the next frozen node is a legal `ContinueAction`.

A station is legal iff travel to it is energy-feasible **and** it has a
**finite continuation hop rank** toward the next frozen node, using
full-battery hops on the same directed energy model:

- rank 0 = next frozen customer/depot
- rank 1 = stations that can reach rank 0 at **full battery**
- rank k+1 = stations that can reach at least one lower-rank station

For rank `r`, min departure SOC is over feasible first hops to the frozen
node or a **strictly lower-rank** station. That blocks station cycles without
assuming Euclidean “must get closer”.

FULL / Hybrid PPO SOC interval: **`continuation_to_max`**
`[max(arrival SOC, continuation min departure SOC), max_soc]`.

Ablation A2: **`arrival_to_max`** `[arrival, max_soc]` only.

Map `u ∈ [0,1]` by `target_soc = soc_lower + u * (soc_upper - soc_lower)`.

This continuation bound is **energy-only**. Time windows and charging duration
are **not** included. It is **not** a globally exact certificate.

Same-station ΔSOC below `ZERO_CHARGE_EPS` is masked as `ZERO_CHARGE_NOOP`.

If every discrete bit is false, the episode is a terminal failure with
`r = -(H - t)`.

### 5.3 MDP (frozen)

| Piece | Definition |
| --- | --- |
| State | Simulator state + remaining frozen customers + station features (`src/rl/features.py`) |
| Discrete action | `{CONTINUE} ∪ stations`, shield-masked |
| Continuous action | `u ∈ (0,1)` mapped through the shield SOC interval (Hybrid PPO / FULL) |
| Transition | `FixedRouteSimulator` |
| Reward | Feasible: `r = -(t' − t)`. Dead-end: `r = -(H − t)` |
| Discount | **γ = 1** so a successful return ≈ **−completion time** |
| Horizon | Depot due date `H` |
| Success | Back at depot, all customers served |

Feature dims: global 8, next 13, customer 11, station 12.

Normalizer: fit on **TRAIN only** (all TRAIN resets + TRAIN-only greedy-min
rollouts). VAL/TEST statistics never enter the fit. Provenance is stored with
the checkpoint.

Episode sampling: `HierarchicalSampler` — **parent → route → terrain**.

### 5.4 Hybrid PPO (the proposed method)

Shared Transformer encoder, **one optimizer**.

```
encoder(s) → h
  Transformer over remaining frozen customers
  attention over stations
discrete head: logits over CONTINUE+stations, softmax on unmasked actions
if station: Beta(α, β), α,β = softplus(MLP([h; station_embed])) + 1
value V(s)
log π = log π_disc + 1[station] log π_Beta(u)
```

CONTINUE has **no** charge / Beta term in the log-prob.

Eval mode: greedy discrete action + Beta mean (deterministic).

Training log-prob is of the **executed** action after the env/shield snap of
`u`. GAE bootstraps `V(s')` on truncated (non-terminal) rollouts.

Configs:

| File | Role |
| --- | --- |
| `configs/rl/hybrid_ppo.toml` | Frozen paper protocol: 400 updates, 256 steps, val every 10, patience 20 |
| `configs/rl/hybrid_ppo_pilot.toml` | Same ceiling used by the accepted TRAIN/VAL pilot |
| `configs/rl/hybrid_ppo_smoke.toml` | Smoke only |

Shared knobs: `lr=3e-4`, `rollout_steps=256`, `minibatch=32`, `epochs=4`,
`clip=0.2`, `gae_lambda=0.95`, `entropy_coef=0.01`, `value_coef=0.5`,
`max_grad_norm=0.5`, `d_model=64`, `n_heads=4`, `n_layers=1`.

### 5.5 Ablations and other learned methods

| Name | Meaning |
| --- | --- |
| FULL | Hybrid PPO, `continuation_to_max` |
| A1 / DiscretePPO | **True categorical**: station + six charge levels `{0, 0.2, …, 1}` **conditioned on the chosen station**. **No Beta + snap sampling.** Simulator stays continuous. |
| A2 | `arrival_to_max` SOC interval |
| A3 | Zero remaining-route features |
| A4 | Zero terrain/load features |
| A5 | Mean-pool stations instead of attention |
| AttentionPPO | Same Hybrid PPO + depot/customer/station type embedding. **Not** a paper clone. Architectural comparator only. |

`LegacyTwoStageDDQN` exists under `src/baselines/` as **historical/internal
code**. It is **not** in the paper experiment matrix.

### 5.6 Stateless baselines (new simulator only)

- `GreedyMinimumSufficientCharge` — CONTINUE if legal, else min-detour station to `soc_lower` (`u=0`)
- `GreedyFullCharge` — same station rule, `u=1`
- `OneStepLookahead` — one-step time score over CONTINUE and meaningful stations

Do **not** reimplement `check_charging_needed` or `MAX_RL_STOPS`.

### 5.7 frvcpy (native FRVCP only)

`frvcpy.solver.Solver` is exact for **native FRVCP** (published energy matrix +
piecewise charging). It is **not** exact for EVRPTW-GR (payload-dependent
directed energy + Schneider customer TWs). `evrptwgr_to_frvcp_surrogate()`
stays `not_equivalent`. Those IDs are **never** joined into EVRPTW-GR splits.

**Official reviewer-facing reference** (vendored from [e-VRO/frvcpy](https://github.com/e-VRO/frvcpy)):

- `data/external/frvcpy/frvcpy-instance.json`
- `data/external/frvcpy/testdata.json` — **133** routes with known objectives

Parity: every testdata route vs real Solver, upstream
`assertAlmostEqual(..., places=3)` → abs tolerance **5×10⁻⁴**.
Measured max abs error ≈ **4.86×10⁻⁷**.

Tiny `routes.json` is smoke. Sequential node-ID tours in
`data/external/frvcpy/benchmark/` are **optional nonlinear sensitivity**,
**not** official published FRVCP tours. Do not treat them as the primary
benchmark.

Native FRVCP code must not import torch. `src/baselines/__init__.py` uses
lazy `__getattr__`. A previous CI failure was an eager DiscretePPO/torch
import inside the frvcpy job.

---

## 6. Code map (where to look)

Python packages live under `src/` (`package-dir = {"" = "src"}` in
`pyproject.toml`). Tests add `src` to `pythonpath`.

### 6.1 Source modules

| Path | Role / key types |
| --- | --- |
| `src/data/` | Canonical parser only. `parser.py`, `models.py`, `validation.py`, `paths.py`. Do not add a second parser. |
| `src/domain/` | `LoadConvention`, typed quantities |
| `src/physics/` | Demir energy, battery, linear charging, profiles, travel time, network |
| `src/routing/` | PyVRP generator, `FrozenRoute`, corpus serialize/audit |
| `src/simulation/` | `FixedRouteSimulator`, `ContinueAction`/`ChargeAction`, `shield.py`, metrics |
| `src/rl/` | Hybrid PPO: `env.py`, `features.py`, `encoder.py`, `policy.py`, `ppo.py`, `train_loop.py`, `normalization.py`, `sampler.py`, `ablation.py` |
| `src/baselines/` | greedy, lookahead, `discrete_ppo.py`, `legacy_ddqn.py`, `ddqn_train.py`, frvcpy adapter/solver/reference |
| `src/exact/` | Label-setting RCSPP for tiny (≤5 customer) routes |
| `src/experiments/` | splits, evaluate, isolation, stats, actors, provenance, seeds |

### 6.2 Scripts (CLIs)

| Script | Purpose |
| --- | --- |
| `scripts/inspect_dataset.py` | Scan raw EVRPTW-GR (never writes raw) |
| `scripts/generate_routes.py` | **Do not rerun** for the paper |
| `scripts/audit_corpus.py` | Hash / integrity of frozen corpus |
| `scripts/make_splits.py` | **Do not rerun** |
| `scripts/preflight_experiments.py` | Mandatory freeze check (`--paper`) |
| `scripts/train_rl.py` | Paper/pilot training CLI (rejects TEST) |
| `scripts/train_ppo.py` | CPU smoke helper only |
| `scripts/run_ablations.py` | A1–A5 |
| `scripts/run_baselines.py` | Stateless methods |
| `scripts/evaluate.py` | Learned methods on a split+scenario |
| `scripts/evaluate_policy.py` | Single checkpoint |
| `scripts/run_frvcpy_benchmark.py` | Native FRVCP + parity |
| `scripts/run_exact_small.py` | Label-setting on ≤5-customer TEST later |
| `scripts/run_soc_reserve.py` | Separate `soc_reserve` scenario |
| `scripts/run_size_generalization.py` | Extra, never the headline TEST number |
| `scripts/run_nonlinear_sensitivity.py` | Optional Montoya |
| `scripts/analyze_results.py` | Stats |
| `scripts/make_tables.py` / `make_figures.py` | Paper artifacts only after freeze |
| `scripts/simulate_route.py` | Debug a single frozen route |
| `scripts/validate_physics.py` | Physics checks |

### 6.3 Tests (mirror the science)

| Path | Guards |
| --- | --- |
| `tests/test_parser.py`, `test_dataset_scan.py`, `test_validation.py` | Raw data immutable, parse 124 files |
| `tests/physics/` | Energy, battery, charging, Demir replica, quantities |
| `tests/routing/` | PyVRP wrapper, serialize, **pinned corpus hashes** |
| `tests/splits/test_splits.py` | 20/6/6, no leakage |
| `tests/simulation/test_simulator.py` | Transitions, TWs, load convention |
| `tests/simulation/test_shield.py` | Hop ranks, continuation, ZERO_CHARGE_NOOP |
| `tests/rl/test_ppo.py` | Hybrid PPO pieces |
| `tests/rl/test_discrete_ppo.py` | Categorical A1, no Beta snap |
| `tests/rl/test_gae_entropy.py` | GAE bootstrap `V(s')` |
| `tests/baselines/test_baselines.py` | Greedy / lookahead |
| `tests/baselines/test_ddqn.py` | Double DQN select/eval |
| `tests/baselines/test_frvcpy_parity.py` | Official testdata.json 133 routes |
| `tests/experiments/` | Isolation, expected counts, no SOC-reserve leak into `main_test` |

Run `python -m pytest tests -q --tb=short` after any code change.

### 6.4 Data / results layout

```
data/raw/                  immutable EVRPTW-GR
data/processed/            gitignored generated metadata
data/routes/               frozen corpus (3 tracked files)
data/splits/               frozen 20/6/6
data/external/frvcpy/      native FRVCP + testdata.json

configs/physics/           energy/charging profiles
configs/routing/pyvrp.toml frozen generator settings
configs/rl/                PPO TOML (legacy_ddqn.toml is internal, not paper)
configs/experiments/       seeds.toml, splits.toml, sampling.toml

checkpoints/               gitignored local training artifacts
results/pilot/             TRAIN/VAL pilot only (keep paper tables empty)
results/smoke/             already-present smoke tables; NOT paper
results/tables|figures|summaries/  paper outputs — do not fill from pilot
```

`results/summaries/experiment_freeze.json` is written by `--paper` preflight.
After a successful freeze it **dirties the tree** unless already committed.

---

## 7. How one training step actually works

1. `train_rl.py` loads TRAIN routes and VALIDATION routes. Rejects TEST.
2. Fit `Normalizer` on TRAIN resets + TRAIN greedy-min dynamics.
3. `HierarchicalSampler` draws a parent, then a route, then a terrain sibling.
4. `ChargingEnv` resets `FixedRouteSimulator` on that frozen customer sequence.
5. `extract_features` builds tensors + shield `discrete_mask`.
6. Policy samples a masked discrete action. If station: sample Beta `u`, map
   through `soc_interval_for_station`.
7. Env applies `ContinueAction` or `ChargeAction`. Reward is minus elapsed
   time, or `-(H-t)` on shield dead-end.
8. After `rollout_steps` (256), PPO updates with GAE (γ=1, λ=0.95), clipping,
   entropy, value loss, grad clip 0.5.
9. Every `eval_interval` (10) updates: evaluate **full VAL** (pilot/paper).
   Keep checkpoint if feasibility is better, or feasibility ties and
   completion-all is better.
10. Early stop if VAL selection metric does not improve for `patience`
    intervals.

Hardware: CUDA if `torch.cuda.is_available()`, else CPU. Device is recorded
in each run `manifest.json`. Seed 42 pilot ran on **CPU**.

---

## 8. Experiment protocol (Part 3B)

Scenario names **must never be mixed** in one paper table
(`src/experiments/isolation.py`):

`main_test`, `ablation`, `soc_reserve`, `terrain_analysis`, `frvcpy_native`,
`exact_small`, `nonlinear_sensitivity`, `smoke`, `pilot`.

Every eval CLI requires `--scenario`.

Seeds (`configs/experiments/seeds.toml`):

- Paper: `[42, 43, 44, 45, 46]` — do **not** pick the best seed
- Extended: `42…51`
- Ablations: `[42, 43, 44]`

Statistics (when TEST eventually runs):

1. Per-seed metrics
2. Mean and SD **across training seeds**
3. Parent-cluster uncertainty
4. Hierarchical bootstrap resampling unit: **training seed, then `base_instance`**
5. Paired Hybrid vs baseline tests match parents
6. Holm correction over comparisons

SOC-reserve rows use `scenario=soc_reserve` and **never** enter Hybrid PPO
`scenario=main_test` sample sizes.

Size-generalization is extra, never the headline TEST number.

Smoke (`--smoke`) caps VAL (16 routes) and is **not paper**. Existing
`results/smoke/tables/` must not be copied into `results/tables/`.

---

## 9. Hard rules and agent SOP

### 9.1 Never do these

- Write to `data/raw/`
- Regenerate `data/routes/` or change `configs/routing/pyvrp.toml`
- Rerun `scripts/make_splits.py`
- Change physics formulas, typed quantities, shield hop ranks, Hybrid PPO
  architecture, or the reward without explicit user approval
- Train or select checkpoints on TEST
- Launch ` --seeds paper ` or ablations before the user freezes the budget
- Write the manuscript
- Put pilot numbers into `results/tables/` or `results/figures/`
- Silently change lr / entropy / width / patience after seeing VAL
- Reintroduce the deleted `Training/` stack, mixed-unit `total_cost`,
  `close_time`, or “max three charging stops”
- Claim frvcpy optimality on EVRPTW-GR
- Treat sequential-ID Montoya tours as official FRVCP
- Eagerly import torch from native FRVCP code paths
- Use PowerShell `&&` (invalid). Use `;` or separate commands
- Commit unless the user asks. Do not force-push. Do not update git config

### 9.2 Stop-and-report conditions

Stop immediately and tell the user (do not auto-tune) if:

- VAL feasibility stays near 0
- losses become NaN/Inf
- the policy collapses to one action
- pinned hashes drift
- pytest or CI fails for a scientific reason rather than environment

### 9.3 Allowed without asking

- Read code/docs, add comments only if the user asked
- Run pytest / preflight (preflight `--paper` requires a clean tree)
- Expand `context.md` when the user asks for agent briefing
- Fix a **verified implementation bug** that contradicts the frozen spec.
  That is a bugfix, not a redesign. Still do not then start TEST.

### 9.4 Windows / Git pitfalls

- Workspace is Windows; CI is Ubuntu. Frozen files are `-text`.
- `core.autocrlf` must not rewrite corpus/splits/testdata hashes.
- Python stdout is buffered when not a TTY: use `python -u`.
- `requires-python >= 3.11`. CI must not use 3.10.
- `pip install -e . --no-deps` is used in CI after installing torch/pyvrp or
  numpy/frvcpy separately.

---

## 10. Commands

All commands assume the repo root and Python 3.11+.

### 10.1 Install

```bash
python -m pip install --upgrade pip setuptools
pip install -e ".[dev]"
pip install -e ".[dev,frvcpy]"
pip install -e ".[plots]"
```

CI CPU torch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy "pyvrp==0.14.0" pytest
pip install -e . --no-deps
```

Frvcpy CI job:

```bash
pip install numpy pytest frvcpy
pip install -e . --no-deps
```

### 10.2 Tests and freeze

```bash
python -m pytest tests -q --tb=short
python -m pytest tests/baselines/test_frvcpy_parity.py -q --tb=short
python scripts/preflight_experiments.py --paper
```

`--paper` fails if git is dirty, pinned hashes drift, pytest fails, corpus
audit fails, or split leakage is detected. It writes
`results/summaries/experiment_freeze.json`.

### 10.3 Dataset / corpus / splits (already done)

```bash
python scripts/inspect_dataset.py
python scripts/generate_routes.py          # DO NOT regenerate frozen corpus
python scripts/audit_corpus.py --routes data/routes
python scripts/make_splits.py --out data/splits   # DO NOT rerun
```

### 10.4 TRAIN / VALIDATION Hybrid PPO pilot (done)

The accepted protocol is frozen in `configs/rl/hybrid_ppo.toml`. Archive:
`results/pilot/post_correctness_fix/`. Do **not** rerun the pilot. Do **not**
tune against VALIDATION further. `--split test` is rejected by `train_rl.py`.

### 10.5 Paper training (only when the user starts the experiment phase)

```bash
python scripts/train_rl.py --method hybrid_ppo --split train --seeds paper
python scripts/train_rl.py --method discrete_ppo --seeds paper
python scripts/train_rl.py --method attention_ppo --seeds paper
python scripts/run_ablations.py --seeds ablation
```

Do **not** add `--method legacy_ddqn`. DDQN is not in the paper matrix.

### 10.6 Evaluation / tables (TEST only after training from paper_code_sha)

```bash
python scripts/run_baselines.py --split test --scenario main_test
python scripts/evaluate.py --split test --scenario main_test --methods hybrid_ppo,discrete_ppo,attention_ppo --seeds paper
python scripts/evaluate.py --split test --scenario ablation --methods A1,A2,A3,A4,A5 --seeds ablation
python scripts/run_soc_reserve.py --levels 0,0.05,0.10,0.15
python scripts/run_exact_small.py --split test --max-customers 5
python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy --scenario frvcpy_native
python scripts/run_frvcpy_benchmark.py --parity-only
python scripts/run_frvcpy_benchmark.py --fixtures-only
python scripts/run_restricted_search.py --splits train,validation
python scripts/analyze_results.py --scenario main_test --split test
python scripts/make_tables.py --scenario main_test
python scripts/make_figures.py
```

### 10.7 GitHub Actions

`.github/workflows/tests.yml`

| Job | Must pass |
| --- | --- |
| `unit-tests` | Ubuntu, Python 3.11, CPU torch, `pytest tests` |
| `frvcpy` | Ubuntu, Python 3.11, `frvcpy` installed, testdata parity + tiny smoke |

Both jobs are required (`continue-on-error` was removed). Actions:
`checkout@v7`, `setup-python@v6`.

---

## 11. Current status (21 Sep 2026)

| Item | State |
| --- | --- |
| Proposed method | Hybrid PPO only |
| Paper Hybrid PPO protocol | 400 updates × 256 steps, val every 10, patience 20, per-seed parent-balanced `best.pt` |
| Third TRAIN/VAL Hybrid PPO pilot | Accepted; archive `results/pilot/post_correctness_fix/` |
| Seed 42 pilot best | update 200, 21/77 VAL feasible, parent-balanced ~9.7%, CONTINUE/CHARGE ~49/51, 0 revisits, 0 loop-guard, no NaNs |
| Seed 43 pilot best | update 280, 22/77 VAL feasible, parent-balanced ~10.1%, last checkpoint stayed close |
| Freeze-prep sanity (seed 42) | DiscretePPO and AttentionPPO end-to-end; DDQN sanity untracked from the paper archive |
| DDQN | **Not in the paper.** Code/tests may remain as historical/internal. |
| Corpus / split hashes | Unchanged, pinned in preflight |
| TEST | **Not used** for final evaluation |
| 5-seed paper run | **Not started** |
| Ablations | **Not started** |
| Manuscript | **Not started** |
| Methodology tuning | **Stopped** unless a verified implementation bug appears |

`paper_code_sha` = `a175ee43548a5d2d5154a9ae0731a01f7642a7f6`
(in `results/summaries/experiment_freeze.json`). Do not train from a later
bookkeeping snapshot commit of that JSON.

---

## 12. Exact next commands (only when the user starts the experiment phase)

Train from `paper_code_sha`, TRAIN/VAL only, no TEST until evaluation:

```bash
python scripts/preflight_experiments.py --paper
python scripts/train_rl.py --method hybrid_ppo --split train --seeds paper
python scripts/train_rl.py --method discrete_ppo --seeds paper
python scripts/train_rl.py --method attention_ppo --seeds paper
python scripts/run_ablations.py --seeds ablation
```

Then, and only then, evaluate the untouched TEST population, native `frvcpy`,
restricted-search reference, statistics, tables, and figures. Do **not**
include `legacy_ddqn`.

Do **not** run `--seeds paper` or `--split test` until the user asks.

---

## 13. Longer design docs

| File | Topic |
| --- | --- |
| `data/README.md` | EVRPTW-GR format and parser warnings |
| `docs/physics.md` | Energy, charging, battery |
| `docs/simulator.md` | Transition engine |
| `docs/feasibility.md` | Shield / legality |
| `docs/routes.md` | PyVRP corpus |
| `docs/splits.md` | 20/6/6 parents |
| `docs/rl.md` | Hybrid PPO and baselines |
| `docs/experiments.md` | Part 3B protocol, seeds, stats |
| `docs/frvcpy_env.md` | Optional sidecar env |
| `results/smoke/README.md` | Smoke tables are not paper |
| `data/external/frvcpy/README.md` | Official testdata vs optional tours |

---

## 14. Glossary

| Term | Meaning here |
| --- | --- |
| Parent / `base_instance` | Solomon–Schneider family id (`c101`, `r203`, …). Split unit. |
| Frozen route | Immutable customer ID tuple from PyVRP. No stations stored. |
| CONTINUE | Discrete action: go to the next frozen customer or depot. |
| u | Continuous charge fraction inside the shield SOC interval. |
| H | Depot due date; used as failure completion time. |
| FULL | Proposed Hybrid PPO with `continuation_to_max`. |
| A1…A5 | Ablations of FULL. A1 = DiscretePPO. |
| FRVCP | Fixed-Route Vehicle Charging Problem (Froger / Montoya). Native only. |
| EVRPTW-GR | This paper’s dataset / physics. Not equivalent to native FRVCP. |
| Pilot | TRAIN/VAL Hybrid PPO budget search. Scenario `pilot`. |
| Paper seeds | 42–46. Not cherry-picked. |
| `best.pt` | VAL-selected checkpoint. The one to evaluate. |
| `last.pt` | Last update. Not the reported model. |
