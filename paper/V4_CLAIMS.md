# V4 claims discipline (standalone)

This document governs manuscript claims for the **V4** experimental package only.
It does not discuss prior experimental versions.

## Supported claims

### Main performance

FA-HPPO with the normalized time-horizon reward achieved approximately **94.6%**
mean feasibility across five seeds on the fresh independently generated
held-out SynthCharge TEST.

### Baseline comparison

On the same locked TEST and feasibility-aware environment, FA-HPPO substantially
exceeded the feasibility of One-step lookahead, Greedy full charge, and
Greedy minimum charge.

### Charging-required routes

FA-HPPO achieved approximately **93.2%** mean feasibility on the 144
charging-required routes. On the 36 no-charge-required routes it reached **100%**.

### Reward finding (development / VALIDATION)

Potential-based reward shaping did not improve parent-balanced validation
feasibility relative to the simpler normalized time-horizon objective. The
simpler reward was therefore selected.

### Reward interpretation

The selected reward represents normalized elapsed time on feasible transitions
and maps terminal failure to the remaining horizon, so successful returns equal
\(-T/C_{\mathrm{train}}\) and failed returns equal \(-H/C_{\mathrm{train}}\)
(with \(C_{\mathrm{train}}=10\) on SynthCharge). Hard constraints are enforced
structurally rather than by arbitrary soft penalty weights.

## Limitations (must appear)

- Fixed customer sequence: the method does not solve routing.
- SynthCharge linear energy model.
- Certificate-filtered stress benchmark; certificates are positive feasibility
  witnesses, not proofs of infeasibility when search times out.
- Independently generated and held out, but **not** a different external domain.
- Deterministic heuristic baselines are not exact global optimization methods.
- Station visits, energy charged, and terminal SOC are secondary reported
  metrics, not directly optimized objectives.
- Continuous charging amount should not be oversold without specific supporting
  V4 evidence.

## Forbidden claims

- External-domain generalization.
- Exact global optimality.
- That potential shaping improved FA-HPPO.
- Energy / station / terminal-SOC “efficiency” without matched common-feasible evidence.
- Calling terminal SOC “waste.”
- Post-TEST tuning, reselection, or reward redesign.
