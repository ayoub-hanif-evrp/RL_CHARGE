# V4 paired VAL analysis

## Exact-horizon audit (successful VAL rows, all variants×seeds)

- n_success_rows: 1766
- |T-H| < 1e-9: 0
- |T-H| < 1e-6: 0
- min |T-H|: 3.824028127752733
- median |T-H|: 6.116211283832202

No successful VAL evaluation row finishes exactly at the horizon under either tolerance; T=H remains a theoretical boundary where success and failure returns coincide for V4_BASE_NO_L_FAIL.

## Paired comparisons

### V4_BASE_NO_L_FAIL vs V3_TIME

- feasibility: V4_BASE_NO_L_FAIL only=7, V3_TIME only=7, tie=436 (n=450)
- paired completion_all mean(V4_BASE_NO_L_FAIL-V3_TIME)=-0.0366

Common-feasible operational mean deltas (b - a):

- completion_time: n=435, mean=-0.03673383155701749
- total_charging_time: n=435, mean=-0.013836980521669817
- total_energy_charged: n=435, mean=-0.013836980521669817
- n_station_visits: n=435, mean=-0.15862068965517243
- terminal_soc: n=435, mean=-0.003071679047169288
- total_distance: n=435, mean=-0.0504332356112084

Per-seed feasibility:
- seed 42: V3_TIME=98.9%, V4_BASE_NO_L_FAIL=98.9%
- seed 43: V3_TIME=97.8%, V4_BASE_NO_L_FAIL=98.9%
- seed 44: V3_TIME=95.6%, V4_BASE_NO_L_FAIL=98.9%
- seed 45: V3_TIME=98.9%, V4_BASE_NO_L_FAIL=97.8%
- seed 46: V3_TIME=100.0%, V4_BASE_NO_L_FAIL=96.7%

### V4_PBRS vs V4_BASE_NO_L_FAIL

- feasibility: V4_PBRS only=5, V4_BASE_NO_L_FAIL only=7, tie=438 (n=450)
- paired completion_all mean(V4_PBRS-V4_BASE_NO_L_FAIL)=0.0132

Common-feasible operational mean deltas (b - a):

- completion_time: n=435, mean=-0.007221871991266698
- total_charging_time: n=435, mean=-0.0014113462926295306
- total_energy_charged: n=435, mean=-0.0014113462926295306
- n_station_visits: n=435, mean=0.05057471264367816
- terminal_soc: n=435, mean=-0.000201927069686888
- total_distance: n=435, mean=-0.005322301859019121

Per-seed feasibility:
- seed 42: V4_BASE_NO_L_FAIL=98.9%, V4_PBRS=98.9%
- seed 43: V4_BASE_NO_L_FAIL=98.9%, V4_PBRS=96.7%
- seed 44: V4_BASE_NO_L_FAIL=98.9%, V4_PBRS=98.9%
- seed 45: V4_BASE_NO_L_FAIL=97.8%, V4_PBRS=97.8%
- seed 46: V4_BASE_NO_L_FAIL=96.7%, V4_PBRS=96.7%

