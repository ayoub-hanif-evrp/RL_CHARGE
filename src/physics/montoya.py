"""Montoya / VRP-REP 2016-0020 piecewise charging knots.

These knots are a *sensitivity* conversion onto Schneider/EVRPTW-GR SOC and
time units. They are not original EVRPTW-GR physics.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional

from data.paths import REPO_ROOT

from .charging import GenericPiecewiseLinearChargingModel

MONTOYA_TOML = REPO_ROOT / "configs" / "physics" / "montoya_piecewise.toml"
MONTOYA_BATTERY_WH = 16000.0


def load_montoya_spec(path: Path | None = None) -> dict:
    with Path(path or MONTOYA_TOML).open("rb") as handle:
        return tomllib.load(handle)


def montoya_piecewise_model(
    cs_type: str = "fast",
    *,
    scale_full_charge_time: Optional[float] = None,
    path: Path | None = None,
) -> GenericPiecewiseLinearChargingModel:
    spec = load_montoya_spec(path)
    curves = spec["curves"]
    if cs_type not in curves:
        raise KeyError(f"unknown Montoya CS type {cs_type!r}")
    charge = [float(x) for x in curves[cs_type]["charge_wh"]]
    times = [float(x) for x in curves[cs_type]["time_h"]]
    capacity = float(spec.get("battery_wh", MONTOYA_BATTERY_WH))
    soc = [c / capacity for c in charge]
    if scale_full_charge_time is not None:
        full = times[-1] if times[-1] > 0 else 1.0
        times = [t / full * float(scale_full_charge_time) for t in times]
    model = GenericPiecewiseLinearChargingModel(soc_knots=soc, cumulative_time_knots=times)
    model.name = f"montoya_{cs_type}_sensitivity_not_original_evrptwgr"
    return model
