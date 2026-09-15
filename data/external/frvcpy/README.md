# Native FRVCP fixtures (not EVRPTW-GR)

Vendored from [e-VRO/frvcpy](https://github.com/e-VRO/frvcpy) (Froger et al. 2019 labeling; Montoya / VRP-REP 2016-0020 charging functions).

These instance IDs are **never** joined into EVRPTW-GR train/validation/test splits.

`frvcpy` is exact for FRVCP given the published energy matrix and piecewise charging curves. It is **not** exact for EVRPTW-GR (payload-dependent directed energy + Schneider customer time windows). `evrptwgr_to_frvcp_surrogate()` stays `not_equivalent`.

## Files

| File | Role |
| --- | --- |
| `frvcpy-instance.json` | Native Solver instance from the frvcpy repo |
| `frvcpy-instance.schema.json` | JSON schema |
| `vrprep-instance.xml` | VRP-REP XML with published fast/normal/slow breakpoints |
| `tiny-instance.json` | Small native fixture for tests and smoke gaps |
| `routes.json` | Frozen customer sequences for the native benchmark |

Hashes are recorded in `hashes.json` (SHA-256 of the bytes as vendored).
