# Release checklist

Owner-only items are marked TODO where metadata is unavailable.

## V4 package (current, standalone)

- [x] Full test suite green (`python -m pytest tests -q` — 256 passed)
- [x] Authoritative V4 reward freeze (`results/v4_reward/FINAL_REWARD_FREEZE.json`)
- [x] Authoritative five-seed training under `final_authoritative/` with `code_git_dirty=false`
- [x] V4 TEST protocol committed before generation
- [x] `results/v4_test/CHECKPOINT_FREEZE.json` present
- [x] `results/v4_test/TEST_LOCK.json` verified (180 routes; seed start 500000)
- [x] `results/v4_test/EVALUATION_CONSUMED.json` present after one-shot TEST
- [x] Standalone paper package: `python scripts/paper/build_v4_results_paper.py --verify`
- [x] Publication outputs in `results_v4_paper/` (PNG only; no prior-version comparisons)
- [x] Benchmark wording clarification (`docs/V4_BENCHMARK_WORDING.md`)
- [x] Standalone claims (`paper/V4_CLAIMS.md`)
- [x] Historical V1/V2/V3 evidence trees untouched

## Historical V3 package (archived; do not reopen)

- [x] V3 protocol / freeze / lock / consumed TEST under `results/v3_hppo/`
- [x] Historical displays under `results_paper/`

## Owner TODO (do not invent)

Do **not** invent license, author identity, ORCID, DOI, or venue.

- [ ] Confirm intended SPDX license and add `LICENSE` if known
- [ ] Author name(s), email(s), ORCID(s), affiliation(s) for `CITATION.cff`
- [ ] Optional `.zenodo.json` with verified metadata only
- [ ] Choose GitHub release tag name
- [ ] Archive via GitHub/Zenodo and insert DOI into the manuscript **after** release
- [ ] Confirm public distribution rights for any non-vendored external data
- [ ] Write manuscript `.tex` / paper source from `results_v4_paper/` + `paper/V4_CLAIMS.md`

## Forbidden before release

- Inventing DOI / ORCID / license / venue
- Post-TEST hyperparameter or final-checkpoint changes
- Deleting unfavorable ablation or baseline rows
- Claiming external-domain generalization for the SynthCharge held-out TEST
- Mixing prior-version comparisons into the V4 paper-facing package
- Reopening a consumed TEST
