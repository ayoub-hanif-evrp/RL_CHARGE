# Reproducibility

Pinned scientific method tree for **V3 claims**: SHA `df35705c012b4ce88674a5b8906176c97838b06c`

For reproducing *V3* numbers, the historical invariant was:

```bash
git diff df35705 HEAD -- src third_party configs/rl configs/physics configs/routing configs/experiments configs/v2
```

**V4-F** intentionally adds a reward module and env wiring under a new stage
(`results/v4_reward/`). Default `RewardConfig` remains `V3_TIME`, so V3 reward
equations stay available. V3 frozen results/checkpoints/TEST locks are not
overwritten; V4 claims require their own protocol and (later) a fresh TEST.

## Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests -q
```

Recorded environment: `results/v3_hppo/ENVIRONMENT.json`.

## Development ablation (TRAIN/VAL only — not TEST)

```bash
python scripts/v3_hppo/run_ablation.py --variant B0 --seed 42
# variants: B0 B1 B2 B3 ; seeds: 42 43 44 45 46
```

## Checkpoint verification (no TEST)

```bash
python scripts/v3_hppo/freeze_checkpoints.py --verify
```

## Publication artifacts from frozen raw rows (preferred)

```bash
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Writes `results_paper/` (PNG figures + CSV/MD/TeX tables). Does **not** train or evaluate TEST.

Official publication builder: `scripts/paper/build_results_paper.py` only.

## Do not casually re-run

- Consumed V3 TEST evaluation (`EVALUATION_CONSUMED.json` present)
- Consumed V2 final evaluation
- Any command that selects final FA-HPPO checkpoints from TEST

Accurate post-TEST claim: **no final FA-HPPO checkpoint was retrained, retuned, or reselected after TEST.**
Predeclared gold development ablations may finish later without contaminating TEST.

If evaluation must be redone for a scientific reason, open a **new versioned** experiment namespace instead of overwriting.

## V4 reward study (new namespace; does not reopen V3 claims)

Design: `docs/V4_REWARD_DESIGN.md`  
Protocol: `results/v4_reward/PAPER_PROTOCOL.json`  
\(C_{\mathrm{train}}=10.0\) (TRAIN median horizon): `results/v4_reward/C_TRAIN.json`

```bash
python scripts/v4_reward/run_reward_ablation.py --variant all
python scripts/paper/build_v4_training_figures.py
```

- Artifacts: `results/v4_reward/`, `checkpoints_v4/`, `results_v4/figures/`
- Default env reward remains `V3_TIME` (frozen V3 equation).
- V4 kinds embed \(C_{\mathrm{train}}\) and use PPO `return_scale=1`.
- **No V4 confirmatory TEST** until reward selection + freeze + dedicated TEST protocol.
- Do not use consumed V3 TEST for reward selection.

## Historical regeneration

- V2 analysis (read-only): `python scripts/v2/analyze_final_v2.py`
- V1 tooling: root `scripts/*.py` (archived)

See also `results/v3_hppo/WORDING_CLARIFICATIONS.md`.
