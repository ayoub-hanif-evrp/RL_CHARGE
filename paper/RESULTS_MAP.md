# Results map (FA-HPPO paper)

Every paper display regenerates from tracked raw/stat files via `scripts/paper/build_paper_artifacts.py`.

| Paper item | Source raw / stats | Generator | Metric | Dataset | Split | Routes | Seeds | Role |
|------------|--------------------|-----------|--------|---------|-------|--------|-------|------|
| Fig 1 method overview | conceptual | `build_paper_artifacts.py` (`figure_method_overview`) | n/a | n/a | n/a | n/a | n/a | schematic |
| Fig 2 main TEST | `results/v3_hppo/raw/synthcharge_test.jsonl` | same | feasibility, completion_all | SynthCharge | fresh TEST | 180 | 5 | confirmatory |
| Fig 3 charging-required | same | same | feasibility, completion_all | SynthCharge | TEST | 144 | 5 | confirmatory |
| Fig 4 five-seed robustness | same | same | per-seed feasibility/completion | SynthCharge | TEST | 180 | 5 | confirmatory |
| Fig 5 B0–B2 ablation | `results/v3_hppo/ablation/` | same | VAL feasibility | gold | VAL | 47 | 5 | development |
| Fig 6 training stability | ablation curves | same | value loss, grad | gold | TRAIN/VAL | — | 5 | development |
| Fig 7 difficulty heatmap | TEST raw | same | charging-required feas | SynthCharge | TEST | 144 | 5 | confirmatory |
| Fig 8 failure analysis | TEST raw | same | failure reasons | SynthCharge | TEST | 180 | 5 | confirmatory |
| Fig 9 native FRVCP | `results/final/statistics/frvcpy_native/` | same | feasibility / gap | native FRVCP archive | V1 | 133 | — | separate reference |
| Table 1 benchmark | corpora metadata | same | composition | all | — | — | — | provenance |
| Table 2 main TEST | TEST stats | same | primary metrics | SynthCharge | TEST | 180 | 5 | confirmatory |
| Table 3 charging-required | TEST stats | same | subset metrics | SynthCharge | TEST | 144 | 5 | confirmatory |
| Table 4A development ablation | ablation stats | same | B0–B2 | gold | VAL | 47 | 5 | development |
| Table 4B amount sensitivity | TEST stats | same | free / Min / Max | SynthCharge | TEST | 180 | 5 | secondary TEST |
| Table 5 failures | TEST stats | same | reasons | SynthCharge | TEST | 180 | 5 | confirmatory |
| Sensitivity JSON | TEST raw | same | joint seed×route bootstrap | SynthCharge | TEST | 180 | 5 | analysis-only |

## Wording

V3 is a **fresh independently generated SynthCharge confirmatory TEST** (new seeds ≥400000), not “external domain generalization.”

## Historical (not paper confirmatory)

| Item | Location | Note |
|------|----------|------|
| V2 HybridPPO vs DiscretePPO | `results/v2/final/` | Preserved; disclose in experimental history |
| V1 five-seed / FRVCP | `results/final/` | Archived |
