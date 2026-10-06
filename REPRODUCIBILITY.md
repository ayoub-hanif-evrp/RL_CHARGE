# Reproducibility

## Current V4 package (standalone)

Authoritative reward training SHA is recorded in
`results/v4/reward_development/FINAL_REWARD_FREEZE.json` (`training_git_sha`).
Checkpoints: `models/v4/fa_hppo/`.

Regenerate **displays only** (no TEST rerun, no retrain):

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

Outputs: `results/v4/paper/`.

Verify frozen TEST lock (resolves legacy path keys via `repo_paths`):

```bash
python scripts/v4/test/write_test_lock.py --verify
```

Do **not** reopen `results/v4/test/EVALUATION_CONSUMED.json`.

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

Outputs remain under `results/v3/paper/` and must not be mixed into
`results/v4/paper/`.

## Immutable rules

- No post-TEST hyperparameter or checkpoint changes.
- No reward redesign after TEST.
- No deletion of unfavorable rows.
- Certificate timeout ≠ infeasible.
- V4 paper package must remain free of prior-version comparisons.

Layout aliases: `src/repo_paths.py`. See `docs/reproducibility/PATH_MIGRATION.md`.
