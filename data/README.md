# Data

## Dataset

**EVRPTW-GR — Dataset for the Electric Vehicle Routing Problem with Time Windows given the Altitude Information**

- **DOI:** [10.17632/srfdbp2twv.1](https://doi.org/10.17632/srfdbp2twv.1)
- **Authors:** Rastani, Keskin, Yüksel, Çatay
- **Companion paper:** Rastani, S., Keskin, M., Yüksel, T., Çatay, B. (2025). *Uphill Struggles, Downhill Gains: How Road Gradients and Load Dynamics Influence Electric Vehicle Routing Decisions?* European Journal of Operational Research.
- **Base instances:** Schneider, M., Stenger, A., Goeke, D. (2014). *The Electric Vehicle-Routing Problem with Time Windows and Recharging Stations.* Transportation Science 48(4), 500–520.

The bundled `EVRPTW-GR_Dataset_Description.pdf` is the authoritative description of the format.

## The raw data is immutable

**Nothing may write to `data/raw/`.** The parser opens files read-only, and the test suite hashes the entire raw tree before and after a full scan to prove it is untouched. If you need a transformed view of the data, write it to `data/processed/` instead.

Raw values are stored exactly as published. No unit is converted and no quantity is derived: distance is never turned into travel time or energy, and the `altitude` column is carried through verbatim.

## Directory structure

```
data/
  raw/
    EVRPTW_GR/                          <- immutable, tracked in Git
      EVRPTW-GR_Dataset_Description.pdf
      Small_Network/
        {5,10,15}_Customers/
          {Level,Nearly_Level,Very_Gentle}/   *.txt
      Medium_Network/
        25_Customers/Nearly_Level/           *.txt
      Large_Network/
        50_Customers/Nearly_Level/           *.txt
  processed/                            <- generated, gitignored
  routes/                               <- empty; reserved for a later part
  splits/                               <- empty; reserved for a later part
```

### Instance file format

A tab-separated header, one tab-separated row per node, a blank line, then a footer of vehicle parameters:

```
StringID	Type		x		y		demand		ReadyTime	DueDate		ServiceTime	altitude
D0		d		40.0		50.0		0.0		0.0		1236.0		0.0		0.0
S0		f		40.0		50.0		0.0		0.0		1236.0		0.0		0.0
C30		c		20.0		55.0		182.5		355.0		407.0		90.0		0.0

Q Vehicle fuel tank capacity /77.75/
C Vehicle load capacity /200.0/
r fuel consumption rate /1.0/
g inverse refueling rate /3.47/
v average Velocity /1.0/
```

`Type` is `d` for the depot, `f` for a charging station, and `c` for a customer.

Filenames follow two conventions. Small and Medium networks use `c101C25_NL.txt` (instance, customer count, terrain variant); the Large network uses `c101_50_21_NL.txt` (instance, customer count, station count, terrain variant). In both, the leading letters give the customer distribution — `c` clustered, `r` random, `rc` mixed — and the following digit gives the Schneider schedule type, where 1 is a short horizon with narrow time windows and 2 a long horizon with wide ones. The trailing `_L`, `_NL`, `_VG` are the Level, Nearly Level, and Very Gentle terrain variants.

## Running the inspection

```bash
python scripts/inspect_dataset.py                    # scan and write metadata
python scripts/inspect_dataset.py --no-write         # report only
python scripts/inspect_dataset.py --fail-on-warning  # treat warnings as failures
python scripts/inspect_dataset.py --root PATH --out-dir PATH
```

The script exits `1` if any validation **error** is found, and `0` when only warnings are present. It works from any working directory.

Tests:

```bash
pip install -e ".[dev]"
pytest
```

## Generated files under `data/processed/`

Both are produced by `scripts/inspect_dataset.py` and are gitignored, since they are reproducible from the raw data at any time.

| File | Contents |
| --- | --- |
| `dataset_manifest.csv` | One row per instance: identity, node counts, per-instance demand / altitude / time-window ranges, and vehicle parameters. |
| `dataset_summary.json` | Dataset-wide counts and ranges, the full list of validation issues, and the terrain-variant coverage gap. |

## Current state of this download

124 instance files parse with **zero validation errors**. The following are reported as warnings rather than corrected, because fixing any of them would mean inventing data.

**The download is incomplete.** The documentation specifies three terrain variants for every subset, i.e. 156 files. Small_Network is complete, but Medium_Network and Large_Network contain only `Nearly_Level`. 32 documented files are absent (24 Medium, 8 Large) and are listed under `coverage.missing_variants` in the summary.

**Small_Network load capacity was never rescaled.** The documentation states demand was scaled up to suit large vehicles. Medium and Large use `C = 3650.0` and are internally consistent, but Small_Network kept the original Schneider capacities (200, 700, 1000) while carrying the scaled demands. In 54 of the 108 Small files a single customer's demand therefore exceeds the stated vehicle load capacity — for example `c101C10_L.txt` has `C = 200.0` against a maximum demand of `730.0`. Reported as `demand_exceeds_capacity`.

**Three files break the documented altitude rule.** The documentation states Very Gentle altitudes are 1.5 times the Nearly Level values. This holds for 514 of 549 node pairs but fails in `r203C10_VG`, `c208C15_VG`, and `c101C5_VG`. Reported as `vg_ratio_mismatch`.

**Vehicle curb weight is absent from Small_Network.** The `M` footer entry appears only in the 16 Medium and Large files. It is left as `None` elsewhere and never given a substitute value. Reported as `missing_curb_weight`.

**The `altitude` column has no documented unit.** The description calls it "elevation of the node" but gives no unit or range, and the published values run from `-0.928571` to `1.3`. No range constraint is enforced and no physical interpretation is applied.

Two further quirks are handled silently by the parser because they are purely cosmetic: `Small_Network/15_Customers/Nearly_level` is spelled with a lowercase `l` unlike every other terrain folder, and line endings are mixed (108 files LF, 16 CRLF, 15 with no trailing newline).

## Scope

Part 1 covers data ingestion, validation, and inspection only.

**Not implemented:** routing, energy consumption modelling, road-gradient or regenerative-braking equations, charging curves and charging behaviour, state of charge, travel-time or distance-to-energy conversions, train/validation/test splits, and reinforcement learning. `data/routes/` and `data/splits/` exist as placeholders for later parts and are intentionally empty.

All dataset access must go through the canonical parser in [`src/data`](../src/data). Do not add a second parser.
