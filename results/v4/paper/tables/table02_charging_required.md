# Table 2 — Charging-required / no-charge subsets

`n_routes` is the number of TEST routes in the subset (not seed×route).
FA-HPPO uncertainty is a 95% Student-t interval across five independently trained seeds.

| Method | Subset | n_routes | Feasibility |
|---|---|---:|---:|
| FA-HPPO | charging_required | 144 | 93.19% [90.37, 96.01] |
| FA-HPPO | no_charge_required | 36 | 100.00% [100.00, 100.00] |
| One-step lookahead | charging_required | 144 | 61.81% |
| One-step lookahead | no_charge_required | 36 | 97.22% |
| Greedy full charge | charging_required | 144 | 40.97% |
| Greedy full charge | no_charge_required | 36 | 100.00% |
| Greedy minimum charge | charging_required | 144 | 30.56% |
| Greedy minimum charge | no_charge_required | 36 | 100.00% |
