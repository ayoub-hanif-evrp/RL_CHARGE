# Table F — native FRVCP (gaps only vs native frvcpy Solver; official upstream e-VRO/frvcpy reference routes/objectives; not EVRPTW-GR)

| block | method | n_routes | n_feasible | n_finite_duration | n_nonfinite_duration | mean_duration_feasible | mean_gap_percent_vs_frvcpy_solver | n_gap_rows | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native_FRVCP | FRVCPGreedyFull | 133 | 29 | 29 | 0 | 7.51964986849468 | 16.16595670502378 | 29 | official upstream e-VRO/frvcpy reference routes/objectives; exact only for native FRVCP; not EVRPTW-GR |
| native_FRVCP | FRVCPGreedyMin | 133 | 23 | 23 | 0 | 6.190848171101118 | 2.3052524293682946 | 23 | official upstream e-VRO/frvcpy reference routes/objectives; exact only for native FRVCP; not EVRPTW-GR |
| native_FRVCP | frvcpy_Solver | 133 | 133 | 133 | 0 | 7.935957547585218 |  | 0 | official upstream e-VRO/frvcpy reference routes/objectives; exact only for native FRVCP; not EVRPTW-GR |
