# Project context — RL_CHARGE / EVRPTW-GR fixed-route charging

This file is a working briefing for the current PhD code, not the paper.
It describes **what the project is**, **what is frozen**, **how to run it**,
and **where the work currently stands**. Update the “Current status” section
when a new frozen SHA or experiment stage is reached.

Repository: https://github.com/ayoub-hanif-evrp/RL_CHARGE.git  
Local workspace: `RL code/` under the PhD folder.  
Python package name: `rl-charge` (`src/` layout). Requires **Python ≥ 3.11**.

---

## 1. What this project is

The scientific problem is **fixed-route electric vehicle charging scheduling**
on the **EVRPTW-GR** dataset (Rastani et al., EJOR 2026 companion data;
Schneider EVRPTW geometry with altitude).

Customer **visit order is frozen** by a customer-only pickup-VRPTW generator
(PyVRP 0.14.0). The RL agent does **not** reroute customers. At each step it
chooses:

- **CONTINUE** to the next frozen customer/depot, or
- a **charging station** plus a **target state of charge**.

The proposed method is **one Hybrid PPO**:

- masked categorical action over `{CONTINUE} ∪ stations`
- if a station is chosen, a **Beta** distribution over `u ∈ (0,1)` mapped
  through a shield SOC interval
- shared Transformer encoder, **one optimizer**
- discount **γ = 1**, so a successful episode return is **−completion time**

This is **not** joint routing+charging, not the deleted `Training/` stack, and
not a reproduction of public HetGAT / eTruckRouting papers.

Primary reported objective: **route completion time**. Energy, distance, and
time components are logged separately. Failures stay in the tables
(`feasible=false`); the all-routes metric uses depot due date **H** for
failures so dropping infeasible routes cannot hide losses.

---

## 2. Frozen scientific objects (do not regenerate)

### 2.1 Git SHA (methodology freeze for the current pilot)

```
a40b5853f25308ac13af94f308fe8fcdfb258f21
```

Paper preflight and the seed-42 pilot were run from this commit.

### 2.2 Dataset

- **Name:** EVRPTW-GR  
- **DOI:** 10.17632/srfdbp2twv.1  
- **Location:** `data/raw/EVRPTW_GR/` (immutable, tracked)  
- **124** instance files, **zero** parser errors  
- Small_Network Q ≈ **77.75**; Medium/Large Q and curb weight as published  
- Load convention for RL/simulation: `LoadConvention.OFFICIAL_REFERENCE_PICKUP`  
- Energy: Demir-style; empty-flat ≈ 1 energy unit per distance  

**Nothing may write to `data/raw/`.**

### 2.3 Frozen route corpus

Generated once with PyVRP (`demand_mapping=pickup`, `Stop=MaxIterations`,
unrestricted fleet with dominating fixed cost
`F = 2 * n_customers * Dmax + 1`). Charging feasibility is **unverified** at
generation time.

| File | SHA-256 (pinned) |
| --- | --- |
| `data/routes/corpus.jsonl` | `1796d817ff058fe0559021ab79ca97087ef3646c57744c2c883ecc37e85451e0` |
| `data/routes/manifest.csv` | `41bb8be3245f8f3b0800db4ba74cb19a2003236b6fbd0dbc1237e7cdfaee104d` |
| `data/routes/corpus_metadata.json` | `1ca8725199842f0a8d4a4e6fac2a31931ea3c9aaca833c2d253c6147469d9a96` |

Metadata counts: **124 instances, 258 routes, seed 42**.

Line endings for these files are stored **byte-for-byte** (`.gitattributes`
`-text`) so Linux CI matches the pinned hashes.

### 2.4 Splits (20 / 6 / 6 parents)

Unit = Solomon–Schneider **parent** `base_instance` (size and terrain of
`c101` stay together). Seed **42**. Algorithm: stratify by
`(customer_distribution, schedule_type)`, shuffle, last→test, second-last→val,
rest→train; Large repair swap `c108` ↔ `c101`.

| File | SHA-256 (pinned) |
| --- | --- |
| `data/splits/train.json` | `3aecd73e9472b8572e899a7e2da3f1f874f0e5a13250f5cd6d76d812002913ce` |
| `data/splits/validation.json` | `1379aef7e92c308f90ab78674dec9994db72078c96cb1020ecc49d33eef69af4` |
| `data/splits/test.json` | `9bf39fbf1f27aa57146879fb71922b0216f24f1eb37a4fd9a9a43b44b6c4b6d0` |
| `data/splits/split_metadata.json` | `4740ab71737bfd9c04d37f258e57bfa4854a526da7c03b218ffe877818128900` |

Expected counts:

- Parents: train 20, validation 6, test 6  
- Instances: train 73, validation 30, test 21  
- **Routes used by RL:** train **132**, validation **77** (test not loaded for
  the pilot)

**TEST is forbidden** for training, checkpoint selection, hyperparameter
decisions, and the current pilot.

---

## 3. Method (frozen for the pilot)

### 3.1 MDP

| Piece | Definition |
| --- | --- |
| State | Simulator state + remaining frozen customers + station features |
| Discrete action | `{CONTINUE} ∪ stations`, shield-masked |
| Continuous action | `u ∈ (0,1)` mapped to `[soc_lower, soc_upper]` |
| Transition | `FixedRouteSimulator` (customer order immutable) |
| Reward | Feasible step: `r = −(t' − t)`. Dead-end: `r = −(H − t)` |
| Discount | γ = 1 |
| Horizon | Depot due date `H` |
| Success | Return to depot, all customers served |

Feature normalizer is fit on **TRAIN only** (all TRAIN resets + TRAIN-only
greedy-min rollouts). VAL/TEST statistics never enter the fit.

### 3.2 Feasibility shield

Method-independent mask. It does **not** choose when to charge.

Station SOC interval for FULL / Hybrid PPO:

- **`continuation_to_max`:** lower bound = max(arrival SOC, energy-continuation
  minimum departure SOC)
- Ablation **A2** uses `arrival_to_max` (physical `[arrival, max]` only)

Continuation is **graph-theoretic hop rank**, not Euclidean “must get closer”:

- rank 0 = next frozen customer/depot  
- rank 1 = stations that can reach rank 0 at **full battery**  
- rank k+1 = stations that can reach at least one rank-k/lower station  

A station is legal only with **finite rank**. For rank `r`, the min departure
SOC is over feasible first hops to the frozen node or a **strictly lower-rank**
station. That blocks station cycles without assuming Euclidean progress.

This is an **energy** continuation bound. Time windows and charging duration
are **not** included. It is **not** a globally exact feasibility certificate.

Same-station ΔSOC below `ZERO_CHARGE_EPS` is masked as `ZERO_CHARGE_NOOP`.

### 3.3 Hybrid PPO (proposed method)

Config paper: `configs/rl/hybrid_ppo.toml`  
Config pilot: `configs/rl/hybrid_ppo_pilot.toml`

Shared settings: `lr=3e-4`, `rollout_steps=256`, `minibatch=32`, `epochs=4`,
`clip=0.2`, `gae_lambda=0.95`, `entropy_coef=0.01`, `value_coef=0.5`,
`max_grad_norm=0.5`, `d_model=64`, `n_heads=4`, `n_layers=1`.

| | Paper | Pilot ceiling |
| --- | --- | --- |
| `budget_updates` | 200 | 400 |
| `eval_interval` | 10 | 10 |
| `early_stopping_patience` | 8 | 20 |

Checkpoint selection is **lexicographic on VALIDATION**: maximize feasibility
rate, then minimize all-routes completion time (`H` for failures). Evaluate
**`best.pt`**, not `last.pt`.

GAE bootstraps `V(s')` on truncated (non-terminal) rollouts. Log-prob is of
the **executed** action after the env/shield snap of `u`.

Sampling of training episodes is hierarchical:
**parent → route → terrain** (`HierarchicalSampler`).

### 3.4 Ablations and other learned methods

| Name | Meaning |
| --- | --- |
| FULL | Hybrid PPO, `continuation_to_max` |
| A1 / DiscretePPO | Categorical station + **six charge levels** `{0,0.2,…,1}` conditioned on the chosen station. **No Beta + snap.** Simulator stays continuous. |
| A2 | `arrival_to_max` SOC interval |
| A3 | Zero remaining-route features |
| A4 | Zero terrain/load features |
| A5 | Mean-pool stations instead of attention |
| AttentionPPO | Same Hybrid PPO + depot/customer/station type embedding. **Not** a paper clone. |
| LegacyTwoStageDDQN | Two-head Q on the **new** simulator. True Double DQN: **online selects**, **target evaluates**. Charge-level head is conditioned on the selected station. Six historical SOC levels `0.5…1.0`. TRAIN for learning, VAL for checkpoints. |

### 3.5 Stateless baselines (new simulator only)

- `GreedyMinimumSufficientCharge` — CONTINUE if legal, else min-detour station to `soc_lower` (`u=0`)
- `GreedyFullCharge` — same station rule, `u=1`
- `OneStepLookahead` — one-step time score over CONTINUE and meaningful stations

None of these use the deleted `check_charging_needed` / `MAX_RL_STOPS` stack.

### 3.6 frvcpy (native FRVCP only)

`frvcpy.solver.Solver` is exact for **native FRVCP** (published energy matrix +
piecewise charging). It is **not** exact for EVRPTW-GR. The EVRPTW-GR export is
labeled `not_equivalent` and must never be reported as EVRPTW-GR optimality.

**Official reviewer-facing reference** (vendored from e-VRO):

- `data/external/frvcpy/frvcpy-instance.json`
- `data/external/frvcpy/testdata.json` — **133** routes with known objectives

Parity: every testdata route vs real Solver, upstream
`assertAlmostEqual(..., places=3)` → abs tolerance **5×10⁻⁴**.
Measured max abs error: **≈ 4.86×10⁻⁷**.

Tiny `routes.json` is smoke only. Montoya XML sequential node-ID tours in
`data/external/frvcpy/benchmark/` are **optional nonlinear sensitivity**,
**not** official published tours.

Native FRVCP code must not import torch. `src/baselines/__init__.py` is lazy.

---

## 4. Directory map

```
src/data/           canonical parser, paths, scan
src/physics/        Demir energy, charging, battery, profiles
src/domain/         load convention
src/routing/        PyVRP generator, corpus, serialize
src/simulation/     FixedRouteSimulator, shield, feasibility
src/rl/             Hybrid PPO, encoder, env, train_loop
src/baselines/      greedy, lookahead, DiscretePPO, DDQN, frvcpy
src/exact/          label-setting (small routes)
src/experiments/    splits, evaluate, isolation, stats, actors

data/raw/           immutable EVRPTW-GR
data/routes/        frozen corpus
data/splits/        frozen 20/6/6
data/external/frvcpy/  native FRVCP + testdata.json

configs/rl/         PPO / DDQN TOML
scripts/            CLIs (see §6)
tests/              pytest
docs/               longer design notes
checkpoints/        gitignored run artifacts
results/pilot/      TRAIN/VAL pilot only
results/tables|figures|summaries/  paper outputs (keep empty until paper runs)
```

Scenario names that **must not be mixed** in one table:

`main_test`, `ablation`, `soc_reserve`, `terrain_analysis`, `frvcpy_native`,
`exact_small`, `nonlinear_sensitivity`, `smoke`, `pilot`.

Paper seeds: `[42, 43, 44, 45, 46]`. Ablations: `[42, 43, 44]`.
Statistics: hierarchical bootstrap **training seed then `base_instance`**.

---

## 5. Hard rules (do not violate)

Do **not**:

- regenerate the EVRPTW-GR corpus or change PyVRP settings
- change the 20/6/6 split
- change physics, shield, Hybrid PPO architecture, or reward without approval
- use TEST for tuning, early stopping, budget choice, or the current pilot
- launch the five-seed paper matrix until the budget is frozen
- write the manuscript yet
- put pilot numbers into `results/tables/` or `results/figures/`
- silently retune lr / entropy / width after seeing VAL

If VAL feasibility stays near zero, losses explode, or the policy collapses:
**stop and report**. Do not “fix it” by unapproved hyperparameter search.

---

## 6. Commands

All commands assume the repo root (`RL code/`) and Python 3.11+.

### 6.1 Install

```bash
python -m pip install --upgrade pip setuptools
pip install -e ".[dev]"
# optional native FRVCP sidecar
pip install -e ".[dev,frvcpy]"
# optional plots
pip install -e ".[plots]"
```

CI uses CPU torch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy "pyvrp==0.14.0" pytest
pip install -e . --no-deps
```

### 6.2 Tests and freeze

```bash
python -m pytest tests -q --tb=short
python -m pytest tests/baselines/test_frvcpy_parity.py -q --tb=short   # needs frvcpy
python scripts/preflight_experiments.py --paper
```

`--paper` fails if git is dirty, pinned hashes drift, pytest fails, corpus
audit fails, or split leakage is detected. It writes
`results/summaries/experiment_freeze.json` (untracked; that file then dirties
the tree **after** a successful freeze).

### 6.3 Dataset / corpus / splits (already done — do not rerun for the paper)

```bash
python scripts/inspect_dataset.py
python scripts/generate_routes.py          # DO NOT regenerate frozen corpus
python scripts/audit_corpus.py --routes data/routes
python scripts/make_splits.py --out data/splits   # DO NOT rerun
```

### 6.4 TRAIN / VALIDATION Hybrid PPO pilot (current stage)

Preflight first (clean tree):

```bash
python scripts/preflight_experiments.py --paper
```

Then:

```bash
python scripts/train_rl.py --method hybrid_ppo --split train --seeds 42,43 --config configs/rl/hybrid_ppo_pilot.toml
```

Do **not** pass `--smoke`. Do **not** cap `--max-train-routes` or
`--max-val-routes`. `--split test` is rejected by the CLI.

Default checkpoint dir: `checkpoints/HybridPPO/seed_{seed}/`

Copy/keep under `results/pilot/` (not paper `results/tables/`):

- `curves.jsonl`
- `manifest.json`
- `best.pt`, `last.pt`
- `normalizer_provenance.json`

`--out-dir` is a **single** directory; both seeds would overwrite. Leave it
unset, or run one seed at a time with a distinct out dir:

```bash
python scripts/train_rl.py --method hybrid_ppo --split train --seeds 42 --config configs/rl/hybrid_ppo_pilot.toml --out-dir results/pilot/seed_42
python scripts/train_rl.py --method hybrid_ppo --split train --seeds 43 --config configs/rl/hybrid_ppo_pilot.toml --out-dir results/pilot/seed_43
```

Python stdout is buffered when not a TTY. For live logs:

```bash
python -u scripts/train_rl.py --method hybrid_ppo --split train --seeds 43 --config configs/rl/hybrid_ppo_pilot.toml
```

### 6.5 Other training (not now)

```bash
python scripts/train_rl.py --method hybrid_ppo --split train --seeds paper
python scripts/train_rl.py --method discrete_ppo --seeds paper
python scripts/train_rl.py --method attention_ppo --seeds paper
python scripts/train_rl.py --method legacy_ddqn --seeds paper
python scripts/run_ablations.py --seeds ablation
```

Smoke only: add `--smoke` (caps VAL at 16 routes). Never use smoke numbers in
the paper.

### 6.6 Evaluation / tables (TEST only after the method is frozen)

```bash
python scripts/run_baselines.py --split test --scenario main_test
python scripts/evaluate.py --split test --scenario main_test --methods hybrid_ppo,discrete_ppo,attention_ppo,legacy_ddqn --seeds paper
python scripts/evaluate.py --split test --scenario ablation --methods A1,A2,A3,A4,A5 --seeds ablation
python scripts/run_soc_reserve.py --levels 0,0.05,0.10,0.15
python scripts/run_exact_small.py --split test --max-customers 5
python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy --scenario frvcpy_native
python scripts/run_frvcpy_benchmark.py --parity-only
python scripts/run_frvcpy_benchmark.py --fixtures-only
python scripts/run_frvcpy_benchmark.py --montoya
python scripts/analyze_results.py --scenario main_test --split test
python scripts/make_tables.py --scenario main_test
python scripts/make_figures.py
```

Every eval CLI requires `--scenario`.

### 6.7 GitHub Actions

Workflow: `.github/workflows/tests.yml`

| Job | Must pass |
| --- | --- |
| `unit-tests` | Python 3.11, CPU torch, `pytest tests` |
| `frvcpy` | Python 3.11, `frvcpy` installed, official testdata parity + tiny smoke |

`continue-on-error` was removed. Both jobs are required.

---

## 7. Current status (17 Sep 2026)

| Item | State |
| --- | --- |
| Frozen SHA | `a40b5853f25308ac13af94f308fe8fcdfb258f21` |
| CI | Both jobs green on this SHA |
| frvcpy parity | 133/133, max abs error ≈ 4.86e−7 |
| Local tests | 174 passed (with frvcpy installed) |
| Corpus / split hashes | Unchanged, pinned in preflight |
| TEST | Not used |
| 5-seed paper run | **Not started** |
| Ablations | **Not started** |
| Manuscript | **Not started** |

### Pilot seed 42 (finished)

- Device: **CPU** (`cuda_name: null`)
- Runtime: **2318 s** (~39 min), ≈ 11.6 s/update
- Status: `early_stop` after **200** updates (patience 20 × interval 10)
- Best checkpoint: **update 0**
- Best VAL feasibility: **0.0** (0/77)
- VAL completion-all: **576.60** (all-routes `H`)
- Loss: large and noisy, finite (no NaN/Inf in `curves.jsonl`)
- Artifacts: `checkpoints/HybridPPO/seed_42/` (`best.pt`, `last.pt`,
  `curves.jsonl`, `manifest.json`, `normalizer_provenance.json`)
- Manifest recorded `git_dirty: true` because preflight had already written
  untracked `results/summaries/experiment_freeze.json`

This is a **no-learning** signal on VAL. Standing instruction: do not silently
change lr, entropy, architecture, shield, or reward.

### Pilot seed 43 (not finished)

Only `checkpoints/HybridPPO/seed_43/best.pt` exists (started ~14:34 on 16 Sep,
then the session was interrupted). No curves, last checkpoint, or manifest.

### Remaining wall-clock (CPU, sequential)

- Seed 43 only: ~35–45 min if it early-stops like 42; ~70–80 min if it hits 400
- Two-seed pilot in total: seed 42 already paid ~39 min

### What is **not** decided yet

The paper Hybrid PPO budget (`200` vs `300` vs `400`) must wait until **both**
pilot seeds exist, unless the user stops on the seed-42 pathology.

---

## 8. Exact next commands (when approved)

Finish seed 43 only, still no TEST:

```bash
python -u scripts/train_rl.py --method hybrid_ppo --split train --seeds 43 --config configs/rl/hybrid_ppo_pilot.toml
```

Then copy `checkpoints/HybridPPO/seed_{42,43}/` into `results/pilot/`, analyze
VAL curves + `best.pt` behavior, and only then freeze a paper
`budget_updates`.

Do **not** run:

```bash
python scripts/train_rl.py --method hybrid_ppo --seeds paper
python scripts/evaluate.py --split test ...
```

until that budget is approved.

---

## 9. Longer design docs

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

If this briefing disagrees with those files, **this file plus the frozen SHA
and pinned hashes** are the operational source of truth for the current stage.
