# Publication outputs (`results/v3/paper/`)

Canonical manuscript outputs. **PNG only** (600 dpi). All figures are flat in `figures/`.

## Figures (20)

| # | File | Question |
|---|------|----------|
| 1 | `fig01_usecase_route.png` | What is the fixed-route charging use case? |
| 2 | `fig02_method_schematic.png` | How does FA-HPPO decide? |
| 3 | `fig03_soc_envelope.png` | What is the SOC envelope? |
| 4 | `fig04_illustrative_soc.png` | How does SOC evolve on the use-case? |
| 5 | `fig05_main_feasibility.png` | Main TEST feasibility? |
| 6 | `fig06_main_completion.png` | Main TEST completion? |
| 7 | `fig07_charging_required.png` | Hard charging-required subset? |
| 8 | `fig08_performance_profiles.png` | Feasibility + speed together? |
| 9 | `fig09_by_layout.png` | Effect of layout? |
| 10 | `fig10_by_length.png` | Effect of route length? |
| 11 | `fig11_difficulty_heatmap.png` | Layout × length feasibility? |
| 12 | `fig12_gain_vs_lookahead.png` | Where are gains largest? |
| 13 | `fig13_ablation_bars.png` | Which methodology factors matter? |
| 14 | `fig14_amount_policies.png` | Learned u vs Max vs Min? |
| 15 | `fig15_charging_effort.png` | How much charging is used? |
| 16 | `fig16_runtime.png` | Evaluation cost? |
| 17 | `fig17_failure_by_regime.png` | Where do failures concentrate? |
| 18 | `fig18_envelope_variants.png` | Envelope design nuance (dev)? |
| 19 | `fig19_charge_decision.png` | What does one charge decision look like? |
| 20 | `fig20_frvcp_reference.png` | Native FRVCP reference? |

Captions: `CAPTIONS.md`. Map: `claims/RESULTS_MAP.md`.

```bash
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```
