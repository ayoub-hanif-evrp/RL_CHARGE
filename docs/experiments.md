# Part 3B experiments

Training, evaluation, ablations, and statistics live here. The Part 3A MDP,
shield, Hybrid PPO architecture, corpus, and splits are frozen unless a
verified bug appears.

## What is and is not compared

- Primary objective: **route completion time**. Energy, distance, and times
  stay separate. There is no mixed-unit `total_cost`.
- Every method sees the same predetermined split population. Failures stay in
  the table (`feasible=false`). The all-routes objective uses depot due date
  `H` for infeasible episodes so filtering cannot hide losses.
- Validation is never mixed into test tables.
- **AttentionPPO** is an architectural baseline (node-type embedding on the
  frozen-route actor). It is **not** a reproduction of joint routing+charging
  papers (eTruckRouting, EV-GNN, HeVRPMD, curriculum HetGAT EVRPTW).
- **frvcpy** is exact only on native Montoya/FRVCP instances under
  `data/external/frvcpy/`. Those IDs are never joined to EVRPTW-GR splits.
  `evrptwgr_to_frvcp_surrogate()` remains `not_equivalent`.
- Label-setting on 5-customer frozen routes is exact for the linear-charging
  extreme-point action set if it finishes. Timeouts are `timeout`, not exact.

## Seeds

Paper: `42,43,44,45,46`. Extended: `42…51`. Ablations: `42,43,44`.
Do not pick the best seed. Config: `configs/experiments/seeds.toml`.

## Hardware

CUDA if `torch.cuda.is_available()`, else CPU. Device is recorded in each
run manifest.

## Commands

```bash
python -m pytest tests -q
python scripts/audit_corpus.py --routes data/routes
python scripts/preflight_experiments.py
python scripts/train_rl.py --method hybrid_ppo --split train --seeds paper
python scripts/train_rl.py --method discrete_ppo --seeds paper
python scripts/train_rl.py --method legacy_ddqn --seeds paper
python scripts/train_rl.py --method attention_ppo --seeds paper
python scripts/run_ablations.py --seeds ablation
python scripts/run_baselines.py --split test
python scripts/evaluate.py --split test --methods all --eval-mode
python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy
python scripts/run_exact_small.py --split test --max-customers 5
python scripts/run_soc_reserve.py --levels 0,0.05,0.10,0.15
python scripts/analyze_results.py
python scripts/make_tables.py
python scripts/make_figures.py
```

Smoke (not paper): add `--smoke` to `train_rl.py` / `run_ablations.py`.

Size-generalization is extra (`scripts/run_size_generalization.py`), never the
headline test number.

Paper Hybrid PPO budget is in `configs/rl/hybrid_ppo.toml`
(`budget_updates=200`, `rollout_steps=256`). Smoke stays in
`configs/rl/hybrid_ppo_smoke.toml`.

## Result files

```
results/raw/{run_id}.jsonl
results/runs/{run_id}/manifest.json
results/summaries/*.csv
results/tables/*.csv + *.md
results/figures/*.png
checkpoints/{method}/seed_{k}/
```

Raw records include method, route, parent, terrain, split, seed, feasibility,
separate time/energy/distance metrics, runtime, and provenance hashes.
