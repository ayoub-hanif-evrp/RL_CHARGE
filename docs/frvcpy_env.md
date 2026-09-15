# frvcpy sidecar environment

`frvcpy` (Froger labeling; Kullman et al., INFORMS JOC) may fail to install on
Python ≥ 3.11. Do **not** rewrite the solver. Use a separate environment:

```bash
python3.10 -m venv .venv-frvcpy
.venv-frvcpy/Scripts/activate   # Windows
# source .venv-frvcpy/bin/activate  # Unix
pip install frvcpy
python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy --scenario frvcpy_native
```

Unit tests keep using `data/external/frvcpy/tiny-instance.json` and the 2-route
`routes.json` fixtures. The official Montoya / VRP-REP 2016-0020 / e-VRO set
lives under `data/external/frvcpy/benchmark/` and is never joined to EVRPTW-GR
splits.

If `import frvcpy` fails, `solve_native` records `frvcpy_not_installed` rather
than substituting another algorithm. GitHub Actions runs an optional
Python 3.10 job for this sidecar.
