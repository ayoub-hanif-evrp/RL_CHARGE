# Release checklist

Owner-only items are marked TODO where metadata is unavailable.

## Automated / agent-completable

- [ ] Full test suite green (`python -m pytest tests -q`)
- [ ] Clean working tree on the release commit
- [ ] `results/v3_hppo/PAPER_PROTOCOL.json` committed before TEST evaluation
- [ ] `results/v3_hppo/CHECKPOINT_FREEZE.json` present with verified SHA256
- [ ] `results/v3_hppo/TEST_LOCK.json` verified
- [ ] `results/v3_hppo/EVALUATION_CONSUMED.json` present after one-shot TEST
- [ ] Paper artifacts regenerated: `python scripts/paper/build_paper_artifacts.py --verify`
- [ ] `FINALIZATION_REPORT.md` complete
- [ ] V2 `results/v2/final/` untouched
- [ ] No DiscretePPO in V3 paper-facing TEST matrix

## Owner TODO (do not invent)

- [ ] Confirm intended SPDX license and add `LICENSE` if known
- [ ] Author name(s), email(s), ORCID(s), affiliation(s) for `CITATION.cff`
- [ ] Optional `.zenodo.json` with verified metadata only
- [ ] Choose GitHub release tag name
- [ ] Archive via GitHub/Zenodo and insert DOI into the manuscript **after** release
- [ ] Confirm public distribution rights for any non-vendored external data

## Forbidden before release

- Inventing DOI / ORCID / license / venue
- Post-TEST hyperparameter or checkpoint changes
- Deleting unfavorable ablation or baseline rows
- Claiming frvcpy exactness for EVRPTW-GR
