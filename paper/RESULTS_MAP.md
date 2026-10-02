# Results map (FA-HPPO paper)

Filled as V3 artifacts are produced. Every paper display must regenerate from tracked raw/stat files.

| Paper item | Source raw / stats | Generator | Metric | Dataset | Split | Routes | Seeds | Role |
|------------|--------------------|-----------|--------|---------|-------|--------|-------|------|
| Fig 1 method overview | conceptual | `scripts/paper/fig_method_overview.py` | n/a | n/a | n/a | n/a | n/a | schematic |
| Fig 2 main TEST | `results/v3_hppo/raw/synthcharge_test.jsonl` | `scripts/paper/build_paper_artifacts.py` | feasibility, completion_all | SynthCharge | TEST | 180 | 5 | confirmatory |
| Fig 3 charging-required | same | same | feasibility, completion_all | SynthCharge | TEST | 144 | 5 | confirmatory |
| Fig 4 five-seed robustness | same | same | per-seed feasibility/completion | SynthCharge | TEST | 180 | 5 | confirmatory |
| Fig 5 B0–B2 ablation | `results/v3_hppo/ablation/` | same | VAL feasibility | gold | VAL | 47 | 5 | development |
| Fig 6 training stability | ablation curves | same | value loss, grad | gold | TRAIN/VAL | — | 5 | development |
| Fig 7 difficulty heatmap | TEST raw | same | charging-required feas | SynthCharge | TEST | 144 | 5 | confirmatory |
| Fig 8 failure analysis | TEST raw | same | failure reasons | SynthCharge | TEST | 180 | 5 | confirmatory |
| Fig 9 native FRVCP | `results/final/figures/frvcpy_native/` + V1 stats | documented export | FRVCP gap | native FRVCP | V1 archive | — | — | separate reference |
| Table 1 benchmark | corpora metadata | same | composition | all | — | — | — | provenance |
| Table 2 main TEST | TEST stats | same | primary metrics | SynthCharge | TEST | 180 | 5 | confirmatory |
| Table 3 charging-required | TEST stats | same | subset metrics | SynthCharge | TEST | 144 | 5 | confirmatory |
| Table 4 ablation | ablation stats | same | B0–B2 (+Min/Max) | gold / TEST | VAL/TEST | — | 5 | mixed; labeled |
| Table 5 failures | TEST stats | same | reasons | SynthCharge | TEST | 180 | 5 | confirmatory |

## Historical (not paper confirmatory)

| Item | Location | Note |
|------|----------|------|
| V2 HybridPPO vs DiscretePPO | `results/v2/final/` | Preserved; not V3 story |
| V1 five-seed TEST | `results/final/` | Archived |
