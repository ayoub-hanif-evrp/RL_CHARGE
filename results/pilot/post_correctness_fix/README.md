# Third Hybrid PPO TRAIN/VAL pilot

Archive for seeds 42 and 43 after the visited-continuation / encoder /
failure-reward correctness audit (`1f9cdda` plus the logging/terminology
pass in the working tree).

TEST was not used. The frozen corpus and 20/6/6 split were not changed.
PPO hyperparameters were not changed.

| File | Contents |
| --- | --- |
| `pilot_summary.json` | Machine-readable decision and per-seed metrics |
| `seed_{42,43}/` | curves, manifests, hashes, best/last policy behaviour |

Large `.pt` checkpoints are gitignored (`results/pilot/**/*.pt`).

**Decision: freeze the predeclared training protocol.** Different best
updates per seed are expected. Do not start TEST or seeds 44–46 in this
archive step.

