# Reproducibility

Pinned scientific method tree: SHA `df35705c012b4ce88674a5b8906176c97838b06c`

Invariant for algorithm/physics code:

```bash
git diff df35705 HEAD -- src third_party configs/rl configs/physics configs/routing configs/experiments configs/v2
```

must be empty. New V3 protocol tests under `tests/test_v3_*` are allowed and do not reopen the method freeze.

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

Writes `results_paper/` (PDF + PNG figures + CSV/MD/TeX tables). Does **not** train or evaluate TEST.

`scripts/paper/build_paper_artifacts.py` is deprecated and delegates to `build_results_paper.py`.

## Do not casually re-run

- Consumed V3 TEST evaluation (`EVALUATION_CONSUMED.json` present)
- Consumed V2 final evaluation
- Any command that selects final FA-HPPO checkpoints from TEST

Accurate post-TEST claim: **no final FA-HPPO checkpoint was retrained, retuned, or reselected after TEST.**
Predeclared gold development ablations may finish later without contaminating TEST.

If evaluation must be redone for a scientific reason, open a **new versioned** experiment namespace instead of overwriting.

## Historical regeneration

- V2 analysis (read-only): `python scripts/v2/analyze_final_v2.py`
- V1 tooling: root `scripts/*.py` (archived)

See also `results/v3_hppo/WORDING_CLARIFICATIONS.md`.
