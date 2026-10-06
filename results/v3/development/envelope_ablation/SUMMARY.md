# Envelope ablation summary

**POST-HOC / DEVELOPMENT-ONLY MECHANISM STUDY — NO V3 TEST USED.**

| Variant | Mean VAL feas. | SD | Route feas. | Completion | Value loss | Grad norm |
|---|---:|---:|---:|---:|---:|---:|
| Arrival-to-Max | 0.993 | 0.006 | 0.993 | 3.882 | 0.0057 | 0.494 |
| EnergyLower-to-Max | 0.989 | 0.014 | 0.989 | 3.883 | 0.0061 | 0.536 |
| EnergyLower-to-TimeUpper | 0.982 | 0.017 | 0.982 | 3.915 | 0.0056 | 0.497 |

Honest observation: on this SynthCharge VAL set, Arrival-to-Max mean feasibility is slightly highest; EnergyLower-to-TimeUpper is slightly lower. This is development evidence only and must not rewrite V3 confirmatory claims.

## Per-seed feasibility

| Variant | 42 | 43 | 44 | 45 | 46 |
|---|---:|---:|---:|---:|---:|
| Arrival-to-Max | 1.000 | 0.989 | 0.989 | 0.989 | 1.000 |
| EnergyLower-to-Max | 1.000 | 0.967 | 0.989 | 1.000 | 0.989 |
| EnergyLower-to-TimeUpper | 0.989 | 0.978 | 0.956 | 0.989 | 1.000 |
