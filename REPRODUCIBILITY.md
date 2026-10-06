# Reproducibility

## Current V4 package (standalone)

Authoritative reward training SHA is recorded in
`results/v4_reward/FINAL_REWARD_FREEZE.json` (`training_git_sha`).

Regenerate **displays only** (no TEST rerun, no retrain):

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

Outputs: `results_v4_paper/`.

Verify frozen TEST lock / consumption:

```bash
python scripts/v4_test/write_test_lock.py --verify
```

Do **not** reopen `results/v4_test/EVALUATION_CONSUMED.json`.

## Environment / tests

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests -q
```

## Historical V3 (untouched)

Pinned scientific method tree for historical V3 claims:
SHA `df35705c012b4ce88674a5b8906176c97838b06c`

```bash
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Outputs remain under `results_paper/` and must not be mixed into
`results_v4_paper/`.

## Immutable rules

- No post-TEST hyperparameter or checkpoint changes.
- No reward redesign after TEST.
- No deletion of unfavorable rows.
- Certificate timeout ≠ infeasible.
- V4 paper package must remain free of prior-version comparisons.
