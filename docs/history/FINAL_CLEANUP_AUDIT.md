# Final cleanup audit

Starting HEAD: `5c6a5968707155bf13152ef889be54510bf5d7fb`  
Method freeze check (`src`, `third_party`, scientific configs): **empty**  
V3 locked instances: **180** / tracked instances: **2846** → unlocked candidates: **2666**

## Classification legend

| Class | Meaning |
|-------|---------|
| KEEP | Required for science, tests, licenses, or frozen provenance |
| DELETE | Safe to remove from HEAD; recoverable from git |
| MOVE/ARCHIVE | Relocate if needed (prefer delete + git history) |
| GENERATED-DUPLICATE | Regenerable from frozen raw/stats |
| UNCERTAIN | Keep |

## Candidates

| Path | Class | References | Recoverable? | Affects V1/V2/V3 repro? | Reason |
|------|-------|------------|--------------|-------------------------|--------|
| `data/.../instances/*` unlocked (2666) | **DELETE** | Not in TEST_LOCK, corpus, certificates, or split | yes | no (rejected candidates; audit JSON kept) | Generator wrote candidates before reject |
| Locked 180 instances | KEEP | TEST_LOCK + corpus | — | yes | V3 TEST members |
| `candidate_audit.jsonl` | KEEP | provenance | — | yes | Seeds + rejection reasons |
| `results/v3_hppo/{protocol,locks,raw,stats,ablation,...}` | KEEP | paper builder | — | yes | Frozen V3 evidence |
| `results/v2/final/` | KEEP | historical | — | V2 | Includes DiscretePPO |
| `results/final/` (except junk) | KEEP | FRVCP fig A2 | — | V1 | Historical package |
| `results/final/excluded/` | KEEP | docs | — | V1 archive policy | Prefer preserve |
| `results/final/.../placeholder.png` | KEEP | none active | — | no | Historical V1; leave |
| `results/pilot/correctness_audit/official_route_plan_mapping.json` | KEEP | `tests/v2/test_v2_root_cause.py` | — | tests | Required by tests |
| `results/pilot/**` (other) | **DELETE** | self-docs only | yes | no | Development pilots |
| `results/smoke/` | **DELETE** | README only | yes | no | Smoke tables not paper |
| `configs/rl/*smoke*`, `*pilot*` | KEEP | tests + scripts | — | tests | Loaded by pytest |
| `context.md` | KEEP / UNCERTAIN | lab handoff | — | no | Prefer keep |
| `paper/figures/`, `paper/tables/` | GENERATED-DUPLICATE → **DELETE** after `results_paper/` | docs | yes | no | Replaced by `results_paper/` |
| `paper/CLAIMS.md`, `RESULTS_MAP.md` | KEEP | claims | — | yes | Manuscript discipline |
| `results/figures/`, `results/tables/` | n/a (empty/absent) | — | — | — | Nothing to delete |
| `results/summaries/experiment_freeze.json` | UNCERTAIN → KEEP | context.md | — | historical note | Tiny; keep |
| `.gitkeep` under paper/figures|tables | DELETE with dirs | — | yes | no | Obsolete dirs |

## V3 instance deletion algorithm (executed)

1. Parse locked instance paths from `TEST_LOCK.json` → 180  
2. `git ls-files` instances → 2846  
3. unlocked = tracked − locked → 2666  
4. Verified unlocked ∩ corpus relative_paths = ∅  
5. Delete unlocked from HEAD; keep `candidate_audit.jsonl`  
6. Re-verify TEST_LOCK hashes for 180 files  

## Soft vs aggressive

Soft cleanup previously preferred keep. This pass **aggressively** removes confirmed non-members and development clutter while preserving freeze, locks, raw rows, and test dependencies.
