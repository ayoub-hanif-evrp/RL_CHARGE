# Continuous amount head limitation (V3)

**Not a V3 bug claim.** Technical note for honest interpretation and V4 design.

## Frozen parameterization

In `src/rl/policy.py` (`HybridPolicy.forward`):

```text
alpha = softplus(x) + 1
beta  = softplus(y) + 1
```

hence \(\alpha > 1\), \(\beta > 1\).

Deterministic evaluation (`act`, `eval_mode=True`):

\[
u = \frac{\alpha}{\alpha+\beta}
\]

(the Beta mean). Stochastic samples are clamped to \([10^{-4},\,1-10^{-4}]\).

## What this can / cannot represent

With \(\alpha,\beta>1\), Beta densities are **unimodal in the open interval** \((0,1)\). The free policy cannot place Beta mass that is singular at the endpoints in the classical \(\alpha\le 1\) or \(\beta\le 1\) sense.

Exact endpoints \(u\in\{0,1\}\) are therefore not natural outcomes of the free Beta mean (they are approached only as one parameter grows large relative to the other).

## Relation to FA-HPPO-Max

On consumed V3 TEST, **FA-HPPO-Max** (\(u=1\)) ≈ free FA-HPPO (~94.3% vs ~94.2%; 173/180 routes tied).

**Do not claim** that the Beta parameterization *caused* Max≈free. The result motivates studying endpoint-capable amount policies as a **V4 research question**, and supports the envelope-centric interpretation of V3 performance.

## V4 candidates (document only)

- Beta without forced `+1` where numerically safe;
- explicit endpoint masses / hurdle mixture at \(u=0,1\);
- alternative bounded continuous distributions.

See `docs/V4_PROPOSAL.md` item V4-B.
