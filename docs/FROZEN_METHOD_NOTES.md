# Frozen method notes (V3)

**Method freeze SHA:** `df35705c012b4ce88674a5b8906176c97838b06c`

These notes document representation and parameterization quirks of the
**consumed V3** FA-HPPO method. They do **not** authorize silent fixes in the
frozen scientific tree (`src/`, `third_party/`, listed `configs/`).

Correcting quirks requires a new method version (V4), retraining, new
checkpoints, and a fresh untouched TEST protocol.

---

## 1. Station-feature duplication

### Location

- File: `src/rl/features.py`
- Function: `_station_features`
- Blob SHA at freeze tip (tracked): see `git rev-parse HEAD:src/rl/features.py`
  (must remain identical to the freeze tree under the method-freeze diff check)

### Frozen 12-dimensional station vector

| Index | Intended semantic (as coded) | Source expression |
|------:|------------------------------|-------------------|
| 0 | current → station distance | `arc_cf.distance` |
| 1 | current → station travel time | `arc_cf.travel_time` |
| 2 | current → station energy | `energy_cf.net_energy` |
| 3 | station → next distance | `arc_fn.distance` |
| 4 | station → next travel time | `arc_fn.travel_time` |
| 5 | station → next energy | `energy_fn.net_energy` |
| 6 | `arrival` (assigned from `interval.soc_lower`) | **duplicate of index 9** |
| 7 | station altitude | `station.altitude` |
| 8 | detour time | `t_cf + t_fn - t_cn` |
| 9 | continuation / interval lower SOC | `interval.soc_lower` |
| 10 | interval upper SOC | `interval.soc_upper` |
| 11 | slack to next after travel to station | `_slack(next, t+t_cf)` |

### Why indices 6 and 9 are duplicates

The frozen code does:

```python
interval = soc_interval_for_station(...)
arrival = interval.soc_lower
# ...
# index 6 ← arrival
# index 9 ← interval.soc_lower
```

So index 6 is **not** the physical station-arrival SOC computed by
`_arrival_soc`. It is a second copy of the feasibility-interval lower bound.

Under `soc_interval="continuation_to_max"`, that lower bound is

\[
\max(\text{arrival SOC},\;\text{energy-continuation departure SOC}),
\]

which equals arrival SOC only when continuation does not raise the floor.

### Why this does **not** invalidate V3

Training and evaluation used the **same** representation. The confirmatory
V3 TEST compares policies trained and evaluated under that frozen feature
map. Duplication wastes one feature slot; it does not introduce a train/eval
mismatch.

### Likely intended distinction (unproven)

A natural intended design would have been:

- index 6 = actual predicted SOC upon reaching the station (`_arrival_soc`);
- index 9 = energy-continuation (feasibility) lower departure SOC;
- index 10 = (possibly time-aware) upper SOC.

That intent is **plausible from naming** (`arrival = ...`) but is **not
proven** by historical documentation in this repository. Do not claim the
author meant something else without evidence.

### Correction path

See `docs/V4_METHOD_CANDIDATES.md`. Any fix changes the learned controller and
requires a new protocol—not a silent patch of V3.

Provenance protection: `tests/rl/test_frozen_station_features.py`.

---

## 2. Beta continuous-head parameterization

### Location

- File: `src/rl/policy.py`
- `HybridPolicy.forward`:

```python
alpha = softplus(x) + 1.0
beta  = softplus(y) + 1.0
```

therefore \(\alpha > 1\) and \(\beta > 1\).

### Evaluation / sampling

- Deterministic eval: \(u = \alpha/(\alpha+\beta)\) (Beta mean).
- Stochastic training samples are clamped to \([10^{-4},\,1-10^{-4}]\).

### Careful interpretation (not a “bug”)

The parameterization favors **unimodal interior** Beta densities and cannot
represent boundary-singular Beta densities (mass near \(u=0\) or \(u=1\)
corresponding to \(\alpha\le 1\) or \(\beta\le 1\)).

On the consumed V3 TEST, **FA-HPPO-Max** (\(u=1\)) performs approximately
identically to free FA-HPPO (~94.3% vs ~94.2%; 173/180 routes tied in
route-averaged feasibility). Endpoint-capable action parameterizations are a
**reasonable future research direction**.

Do **not** claim:

> the Beta parameterization caused FA-HPPO-Max to match or beat FA-HPPO.

That causal link is **not established**. The amount-sensitivity result supports
an envelope-centric interpretation of performance, not a causal indictment of
the Beta head.

---

## 3. Central scientific framing (V3)

Preferred contribution statement:

> Feasibility-aware action formulation for fixed-route EV charging that
> combines an energy-continuation lower SOC bound, an optimistic
> time-feasibility upper SOC bound, and PPO-based station/continue control.

Honest amount-control statement:

> The feasibility-aware SOC envelope accounts for much of the observed
> performance. Once a time-aware upper SOC bound is imposed, targeting the
> upper bound performs nearly identically to learned continuous amount control
> on this benchmark.

---

## 4. Related documents

- `docs/V4_METHOD_CANDIDATES.md` — proposed fixes / alternatives
- `docs/STRONG_COMPARATOR_FEASIBILITY_STUDY.md` — planning baselines
- `docs/ENVIRONMENT_REPRODUCIBILITY.md` — known vs unknown package versions
- `paper/CLAIMS.md` — claim discipline
