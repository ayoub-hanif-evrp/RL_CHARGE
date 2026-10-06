# V4 benchmark wording clarification

This note corrects paper-facing terminology. It does **not** alter the locked
TEST corpus, `TEST_LOCK.json`, or raw evaluation rows.

## Balanced TEST strata (use these in the manuscript)

The locked V4 TEST admits routes by:

\[
\text{layout}\times\text{frozen-route-length bin}
\]

with:

| Factor | Levels |
|--------|--------|
| Layout | `R`, `C`, `RC` |
| Route-length bin | `short` \([3,5]\), `medium` \([6,10]\), `long` \([11,15]\) customers on the frozen route |
| Per cell | 20 routes = 16 charging-required + 4 no-charge-required |

Therefore:

\[
3\times 3\times 20 = 180
\]

Overall counts:

| Subset | Count |
|--------|------:|
| Total routes | 180 |
| Charging-required | 144 |
| No-charge-required | 36 |
| Per layout (R / C / RC) | 60 |
| Per length bin (short / medium / long) | 60 |

**Correct manuscript phrasing:** “20 routes per layout × frozen-route-length cell.”

## Generator settings that are *not* balanced strata

Instance generation also cycles customer-count settings `{15, 30, 50}` (with
associated station counts). These are **generator settings**, not balanced
TEST quotas. Do **not** write “20 routes per layout × customer-scale cell” for
the final V4 paper difficulty tables/figures.

## Preferred TEST naming

Use:

> fresh independently generated held-out SynthCharge TEST

Do **not** use “external TEST” or “external-domain generalization” for this
benchmark. The TEST is independently generated and held out, but it remains
within the SynthCharge linear-energy family.
