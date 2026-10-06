# Table 5 — Operational metrics (supplemental)

> Method-feasible rows carry a **survivor-bias warning** and must not be read as
> cross-method efficiency gains. Prefer matched common-feasible deltas for comparisons.

Units of `n`:
- FA-HPPO method-feasible rows: route×seed observations;
- deterministic baselines: route observations;
- matched comparisons: matched route×seed pairs.

| Conditioning | Method | Metric | n | n_unit | Mean | SD |
|---|---|---|---:|---|---:|---:|
| method_feasible_routes_only | FA-HPPO | completion_time | 851 | route×seed | 3.80287 | 0.875856 |
| method_feasible_routes_only | FA-HPPO | charging_time | 851 | route×seed | 0.291332 | 0.169432 |
| method_feasible_routes_only | FA-HPPO | energy_charged | 851 | route×seed | 0.291332 | 0.169432 |
| method_feasible_routes_only | FA-HPPO | n_station_visits | 851 | route×seed | 1.99647 | 1.4328 |
| method_feasible_routes_only | FA-HPPO | terminal_soc | 851 | route×seed | 0.407167 | 0.190939 |
| method_feasible_routes_only | FA-HPPO | total_distance | 851 | route×seed | 2.11386 | 0.61617 |
| method_feasible_routes_only | FA-HPPO | runtime_s | 851 | route×seed | 0.104741 | 0.0787892 |
| method_feasible_routes_only | One-step lookahead | completion_time | 124 | route | 3.39859 | 0.678778 |
| method_feasible_routes_only | One-step lookahead | charging_time | 124 | route | 0.20035 | 0.143141 |
| method_feasible_routes_only | One-step lookahead | energy_charged | 124 | route | 0.20035 | 0.143141 |
| method_feasible_routes_only | One-step lookahead | n_station_visits | 124 | route | 2.33065 | 1.46904 |
| method_feasible_routes_only | One-step lookahead | terminal_soc | 124 | route | 0.306765 | 0.270692 |
| method_feasible_routes_only | One-step lookahead | total_distance | 124 | route | 1.91058 | 0.435356 |
| method_feasible_routes_only | One-step lookahead | runtime_s | 124 | route | 0.115946 | 0.094854 |
| method_feasible_routes_only | Greedy full charge | completion_time | 95 | route | 3.51972 | 0.740655 |
| method_feasible_routes_only | Greedy full charge | charging_time | 95 | route | 0.228922 | 0.180659 |
| method_feasible_routes_only | Greedy full charge | energy_charged | 95 | route | 0.228922 | 0.180659 |
| method_feasible_routes_only | Greedy full charge | n_station_visits | 95 | route | 0.621053 | 0.487699 |
| method_feasible_routes_only | Greedy full charge | terminal_soc | 95 | route | 0.506201 | 0.315819 |
| method_feasible_routes_only | Greedy full charge | total_distance | 95 | route | 1.70577 | 0.431287 |
| method_feasible_routes_only | Greedy full charge | runtime_s | 95 | route | 0.0473214 | 0.0386799 |
| method_feasible_routes_only | Greedy minimum charge | completion_time | 80 | route | 3.18345 | 0.575984 |
| method_feasible_routes_only | Greedy minimum charge | charging_time | 80 | route | 0.026245 | 0.0319296 |
| method_feasible_routes_only | Greedy minimum charge | energy_charged | 80 | route | 0.026245 | 0.0319296 |
| method_feasible_routes_only | Greedy minimum charge | n_station_visits | 80 | route | 0.55 | 0.500633 |
| method_feasible_routes_only | Greedy minimum charge | terminal_soc | 80 | route | 0.0810499 | 0.145574 |
| method_feasible_routes_only | Greedy minimum charge | total_distance | 80 | route | 1.5753 | 0.313265 |
| method_feasible_routes_only | Greedy minimum charge | runtime_s | 80 | route | 0.045965 | 0.0423982 |
| matched_common_feasible | FA-HPPO minus One-step lookahead | completion_time | 617 | matched route×seed pairs | 0.153008 | 0.355721 |
| matched_common_feasible | FA-HPPO minus One-step lookahead | charging_time | 617 | matched route×seed pairs | 0.0491758 | 0.156969 |
| matched_common_feasible | FA-HPPO minus One-step lookahead | energy_charged | 617 | matched route×seed pairs | 0.0491758 | 0.156969 |
| matched_common_feasible | FA-HPPO minus One-step lookahead | n_station_visits | 617 | matched route×seed pairs | -0.507293 | 1.47612 |
| matched_common_feasible | FA-HPPO minus One-step lookahead | terminal_soc | 617 | matched route×seed pairs | 0.107966 | 0.310478 |
| matched_common_feasible | FA-HPPO minus One-step lookahead | total_distance | 617 | matched route×seed pairs | 0.0239566 | 0.292848 |
| matched_common_feasible | FA-HPPO minus One-step lookahead | runtime_s | 617 | matched route×seed pairs | -0.0216726 | 0.0429525 |
| matched_common_feasible | FA-HPPO minus Greedy full charge | completion_time | 474 | matched route×seed pairs | 0.0570488 | 0.344617 |
| matched_common_feasible | FA-HPPO minus Greedy full charge | charging_time | 474 | matched route×seed pairs | 0.0116449 | 0.168451 |
| matched_common_feasible | FA-HPPO minus Greedy full charge | energy_charged | 474 | matched route×seed pairs | 0.0116449 | 0.168451 |
| matched_common_feasible | FA-HPPO minus Greedy full charge | n_station_visits | 474 | matched route×seed pairs | 1.0865 | 1.13698 |
| matched_common_feasible | FA-HPPO minus Greedy full charge | terminal_soc | 474 | matched route×seed pairs | -0.0876649 | 0.365161 |
| matched_common_feasible | FA-HPPO minus Greedy full charge | total_distance | 474 | matched route×seed pairs | 0.186843 | 0.263014 |
| matched_common_feasible | FA-HPPO minus Greedy full charge | runtime_s | 474 | matched route×seed pairs | 0.0520371 | 0.0448112 |
| matched_common_feasible | FA-HPPO minus Greedy minimum charge | completion_time | 400 | matched route×seed pairs | 0.214134 | 0.29391 |
| matched_common_feasible | FA-HPPO minus Greedy minimum charge | charging_time | 400 | matched route×seed pairs | 0.179513 | 0.120981 |
| matched_common_feasible | FA-HPPO minus Greedy minimum charge | energy_charged | 400 | matched route×seed pairs | 0.179513 | 0.120981 |
| matched_common_feasible | FA-HPPO minus Greedy minimum charge | n_station_visits | 400 | matched route×seed pairs | 0.93 | 0.99124 |
| matched_common_feasible | FA-HPPO minus Greedy minimum charge | terminal_soc | 400 | matched route×seed pairs | 0.343026 | 0.232058 |
| matched_common_feasible | FA-HPPO minus Greedy minimum charge | total_distance | 400 | matched route×seed pairs | 0.16921 | 0.246773 |
| matched_common_feasible | FA-HPPO minus Greedy minimum charge | runtime_s | 400 | matched route×seed pairs | 0.0453701 | 0.0379422 |
