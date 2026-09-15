"""Simulator actions. The caller, not a heuristic trigger, chooses when to charge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinueAction:
    """Travel to the next frozen customer, or return to the depot if none remain."""


@dataclass(frozen=True)
class ChargeAction:
    """Travel to ``station_id`` and charge continuously to ``target_soc``.

    ``target_soc`` is a fraction of Q. It is not restricted to a discrete grid.
    Visiting a station does not advance the customer index.
    """

    station_id: str
    target_soc: float


Action = ContinueAction | ChargeAction
