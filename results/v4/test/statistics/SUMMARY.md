# V4 vs V3 fresh TEST summary

- V4-only feasible: 10
- V3-only feasible: 15
- both feasible: 841
- both infeasible: 34

- mean seed feasibility V4: {'n': 5, 'mean': 0.9455555555555556, 'sd': 0.018172696926440118}
- mean seed feasibility V3: {'n': 5, 'mean': 0.9511111111111111, 'sd': 0.012044157438154892}
- paired completion_all mean(V4-V3): {'n': 900, 'mean': -0.012580597221572018, 'sd': 0.9908318694799221}

Common-feasible operational mean(V4-V3):
- route_completion_time: {'n': 841, 'mean_v4_minus_v3': -0.049010626531108555, 'sd': 0.4275012553341686}
- total_charging_time: {'n': 841, 'mean_v4_minus_v3': -0.00269746609990715, 'sd': 0.12800549301532976}
- total_energy_charged: {'n': 841, 'mean_v4_minus_v3': -0.00269746609990715, 'sd': 0.12800549301532976}
- n_station_visits: {'n': 841, 'mean_v4_minus_v3': -0.039239001189060645, 'sd': 1.510954095045326}
- terminal_soc: {'n': 841, 'mean_v4_minus_v3': 0.007107661865388392, 'sd': 0.17654637431151055}
- total_distance: {'n': 841, 'mean_v4_minus_v3': -0.022162123384250023, 'sd': 0.3810208564949849}
- runtime_s: {'n': 841, 'mean_v4_minus_v3': -0.0017870150958592078, 'sd': 0.026916893657566186}
