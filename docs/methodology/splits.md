# Dataset splits

The split unit is the Solomon–Schneider **parent** `base_instance` (32 groups).
Every size and terrain of `c101` stays together.

## Ratios

20 train / 6 validation / 6 test parents. Seed **42**.

## Algorithm (pinned)

1. Stratify by `(customer_distribution, schedule_type)` — six strata.
2. Sort names, `random.Random(42).shuffle`, assign last → test, second-last →
   val, rest → train (one val and one test parent per stratum).
3. **Large_Network repair:** if TEST has zero Large files, swap the TEST parent
   with a same-stratum Large parent from VAL. On this download that swap is
   `c108` ↔ `c101`.

Expected counts:

- Parents: train 20, validation 6, test 6
- Instances: train 73, validation 30, test 21
- Large parents: train `{c102, r107}`, val `{r102}`, test `{c101}`

Validation is for hyperparameters, early stopping, and architecture. The test
split stays untouched until the method is frozen (Part 3B).

Leakage tests forbid any shared `base_instance`, `instance_id`, or `route_id`.

Write files with:

```bash
python scripts/make_splits.py --out data/splits
```
