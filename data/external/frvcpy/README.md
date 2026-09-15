# Native FRVCP fixtures (not EVRPTW-GR)

Vendored from [e-VRO/frvcpy](https://github.com/e-VRO/frvcpy) (Froger et al. 2019 labeling; Montoya / VRP-REP 2016-0020 charging functions).

These instance IDs are **never** joined into EVRPTW-GR train/validation/test splits.

`frvcpy` is exact for FRVCP given the published energy matrix and piecewise charging curves. It is **not** exact for EVRPTW-GR (payload-dependent directed energy + Schneider customer time windows). `evrptwgr_to_frvcp_surrogate()` stays `not_equivalent`.

## Official reference (reviewer-facing exact benchmark)

Copied unchanged from the upstream package:

| File | Upstream path | Role |
| --- | --- | --- |
| `frvcpy-instance.json` | `frvcpy/test/data/frvcpy-instance.json` | Native Solver instance (tc0c40s8cf0) |
| `testdata.json` | `frvcpy/test/data/testdata.json` | 133 fixed routes with known objectives |
| `vrprep-instance.xml` | `frvcpy/test/data/vrprep-instance.xml` | VRP-REP XML used by the upstream translator test |

Parity: for every `testdata.json` route, `frvcpy.solver.Solver` must match the stored objective to upstream `assertAlmostEqual(..., places=3)` (`abs` tolerance `5e-4`). Only after parity succeeds may these routes be used for method optimality-gap experiments.

## Smoke fixtures (not official published tours)

| File | Role |
| --- | --- |
| `tiny-instance.json` | Small native fixture for unit tests |
| `routes.json` | Tiny smoke sequence only |

## Optional Montoya XML (`benchmark/`)

10-customer Montoya / VRP-REP 2016-0020 XML files. Sequential node-ID tours in `benchmark/routes.json` are **not** official published FRVCP tours. They may be used as optional nonlinear sensitivity/generalization data.

Hashes are recorded in `hashes.json` (SHA-256 of the bytes as vendored).
