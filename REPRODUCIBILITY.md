# Reproducibility

Pinned scientific method tree: SHA `df35705c012b4ce88674a5b8906176c97838b06c`  
(`src/`, `configs/`, `tests/`, `third_party/` must match this freeze.)

## Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests -q
```

Recorded environment: `results/v3_hppo/ENVIRONMENT.json` (after freeze).

## Development ablation (TRAIN/VAL only — not TEST)

```bash
python scripts/v3_hppo/run_ablation.py --variant B0 --seed 42
# variants: B0 B1 B2 B3 ; seeds: 42 43 44 45 46
```

## Checkpoint verification (no TEST)

```bash
python scripts/v3_hppo/freeze_checkpoints.py --verify
```

## Paper artifacts from frozen raw rows (preferred)

```bash
python scripts/paper/build_paper_artifacts.py
python scripts/paper/build_paper_artifacts.py --verify
```

## Do not casually re-run

- Consumed V3 TEST evaluation (`EVALUATION_CONSUMED.json` present)
- Consumed V2 final evaluation
- Any command that selects checkpoints from TEST

If evaluation must be redone for a scientific reason, open a **new versioned** experiment namespace instead of overwriting.

## Historical regeneration

- V2 analysis (read-only): `python scripts/v2/analyze_final_v2.py`
- V1 tooling: root `scripts/*.py` (archived)
