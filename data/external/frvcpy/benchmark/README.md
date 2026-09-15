# Official native FRVCP benchmark

Vendored from:

- [e-VRO/frvcpy](https://github.com/e-VRO/frvcpy) (`frvcpy-instance.json`, `vrprep-instance.xml`)
- [rafaelmartinelli/EVRPNLLib.jl](https://github.com/rafaelmartinelli/EVRPNLLib.jl) 10-customer
  Montoya et al. (2017) E-VRP-NL XML files (VRP-REP 2016-0020)

Original piecewise charging functions are copied unchanged. These instance IDs
are **never** joined to EVRPTW-GR train/validation/test splits.

`tiny-instance.json` and the two-route `../routes.json` remain unit-test/smoke
fixtures only.

`frvcpy.solver.Solver` is exact for these native FRVCP instances. The EVRPTW-GR
surrogate stays `not_equivalent`.
