# Release checklist

Owner-only items are marked TODO where metadata is unavailable.

## Automated / agent-completable

- [x] Full test suite green (`python -m pytest tests -q`)
- [x] Clean working tree on the release commit *(verify at tag time)*
- [x] `results/v3_hppo/PAPER_PROTOCOL.json` committed before TEST evaluation
- [x] `results/v3_hppo/CHECKPOINT_FREEZE.json` present with verified SHA256
- [x] `results/v3_hppo/TEST_LOCK.json` verified
- [x] `results/v3_hppo/EVALUATION_CONSUMED.json` present after one-shot TEST
- [x] Paper artifacts regenerated: `python scripts/paper/build_results_paper.py --verify`
- [x] Publication outputs live only in `results_paper/` (PNG figures)
- [x] Rejected V3 candidate instances removed (180 locked members retained)
- [x] `FINALIZATION_REPORT.md` complete
- [x] V2 `results/v2/final/` untouched
- [x] No DiscretePPO in V3 paper-facing TEST matrix
- [x] Table 4 split into development vs TEST amount sensitivity
- [x] Figure 9 regenerated from archived native FRVCP statistics
- [x] Joint seed×route sensitivity JSON written (analysis-only)
- [x] Wording clarifications: fresh SynthCharge TEST (not external generalization)

## Owner TODO (do not invent)

- [ ] Confirm intended SPDX license and add `LICENSE` if known
- [ ] Author name(s), email(s), ORCID(s), affiliation(s) for `CITATION.cff`
- [ ] Optional `.zenodo.json` with verified metadata only
- [ ] Choose GitHub release tag name
- [ ] Archive via GitHub/Zenodo and insert DOI into the manuscript **after** release
- [ ] Confirm public distribution rights for any non-vendored external data
- [ ] Write manuscript `.tex` / paper source from `paper/` artifacts

## Forbidden before release

- Inventing DOI / ORCID / license / venue
- Post-TEST hyperparameter or final-checkpoint changes
- Deleting unfavorable ablation or baseline rows
- Claiming frvcpy exactness for EVRPTW-GR
- Claiming continuous amount learning beats FA-HPPO-Max on V3 without new evidence
- Calling V3 “external domain generalization”
