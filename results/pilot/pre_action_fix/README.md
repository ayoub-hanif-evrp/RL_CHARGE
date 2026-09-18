# Failed pre-paper Hybrid PPO pilot (do not mix with final results)

This folder is a **labeled archive of the TRAIN/VALIDATION Hybrid PPO
pilot that failed to establish a stable training region**.

It is **not** a paper TEST result. TEST was not used.

| Seed | VAL feasibility | Notes |
| --- | --- | --- |
| 42 | 0/77 throughout | collapsed charger |
| 43 | peak 16/77 at update 200, then 0/77 | spike then collapse |

Large `.pt` checkpoints stay under `checkpoints/HybridPPO/` and are not
part of this archive.

Diagnostics produced by `run_audit.py` also live here.
