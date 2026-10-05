"""V4 reward variants for FA-HPPO (and V3_TIME regression).

Hard constraints stay in the simulator/shield/envelope. This module only
defines the *training* scalar reward. Evaluation metrics are unchanged.

V4 kinds embed ``C_train`` in the reward and expect PPO ``return_scale=1``.
``V3_TIME`` reproduces the frozen V3 env reward (unnormalized ``-Δt`` /
``-(H-t0)-L``); any historical PPO ``return_scale`` is applied outside.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping

from experiments.dataset import parse_route_instance
from routing.fixed_route import FrozenRoute
from simulation.progress import remaining_time_lower_bound
from simulation.simulator import FixedRouteSimulator


class RewardKind(str, Enum):
    V3_TIME = "V3_TIME"
    V4_BASE = "V4_BASE"
    V4_PBRS = "V4_PBRS"
    V4_BASE_NO_L_FAIL = "V4_BASE_NO_L_FAIL"


C_TRAIN_RULE = "median_depot_horizon_over_TRAIN_routes_only"


def train_horizon_list(routes: Iterable[FrozenRoute]) -> list[float]:
    """Depot horizons for the supplied routes (caller must pass TRAIN only)."""
    horizons: list[float] = []
    for route in routes:
        instance = parse_route_instance(route)
        h = float(instance.depot.due_date)
        if h <= 0.0:
            raise ValueError(f"non-positive depot horizon for route {route.route_id}")
        horizons.append(h)
    if not horizons:
        raise ValueError("C_train requires at least one TRAIN route")
    return horizons


def compute_c_train(routes: Iterable[FrozenRoute]) -> float:
    """``C_train = median_{r in TRAIN} H_r``. Never call with VAL/TEST."""
    horizons = train_horizon_list(routes)
    return float(statistics.median(horizons))


@dataclass(frozen=True)
class RewardConfig:
    kind: RewardKind = RewardKind.V3_TIME
    c_train: float = 1.0
    gamma: float = 1.0
    c_train_rule: str = C_TRAIN_RULE

    def __post_init__(self) -> None:
        if float(self.c_train) <= 0.0:
            raise ValueError("c_train must be positive")
        if abs(float(self.gamma) - 1.0) > 1e-12 and self.kind == RewardKind.V4_PBRS:
            raise ValueError("V4_PBRS currently supports gamma=1 only (F=Φ'-Φ)")

    def assert_compatible_with_ppo_gamma(self, ppo_gamma: float, *, tol: float = 1e-12) -> None:
        """PBRS shaping F=Φ'-Φ is valid only when PPO gamma matches reward gamma."""
        if self.kind != RewardKind.V4_PBRS:
            return
        if abs(float(ppo_gamma) - float(self.gamma)) > tol:
            raise ValueError(
                f"V4_PBRS gamma mismatch: reward_config.gamma={self.gamma} vs ppo_gamma={ppo_gamma}"
            )

    @classmethod
    def v3_time(cls) -> "RewardConfig":
        return cls(kind=RewardKind.V3_TIME, c_train=1.0)

    @classmethod
    def v4(cls, kind: RewardKind | str, c_train: float) -> "RewardConfig":
        return cls(kind=RewardKind(kind), c_train=float(c_train), gamma=1.0)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "RewardConfig":
        return cls(
            kind=RewardKind(str(payload.get("kind", RewardKind.V3_TIME.value))),
            c_train=float(payload.get("c_train", 1.0)),
            gamma=float(payload.get("gamma", 1.0)),
            c_train_rule=str(payload.get("c_train_rule", C_TRAIN_RULE)),
        )

    @property
    def embeds_c_train(self) -> bool:
        return self.kind != RewardKind.V3_TIME


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    base_reward: float
    shaping_reward: float
    delta_t: float
    phi_before: float
    phi_after: float
    l_remaining: float
    failed: bool


class RewardComputer:
    """Pure reward math for one env step."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig.v3_time()

    def potential(self, simulator: FixedRouteSimulator) -> float:
        """Φ(s) = -L_remaining(s) / C_train for V4_PBRS; else 0.

        Absorbing / completed states use Φ = 0 (L_remaining already returns 0
        when ``simulator.state.completed``).
        """
        if self.config.kind != RewardKind.V4_PBRS:
            return 0.0
        return -remaining_time_lower_bound(simulator) / float(self.config.c_train)

    def feasible_step(
        self,
        *,
        delta_t: float,
        phi_before: float,
        simulator_after: FixedRouteSimulator,
    ) -> RewardBreakdown:
        dt = float(delta_t)
        if dt < -1e-12:
            raise ValueError("delta_t must be non-negative for feasible steps")
        dt = max(0.0, dt)
        c = float(self.config.c_train)
        phi_after = self.potential(simulator_after)
        if self.config.kind == RewardKind.V3_TIME:
            base = -dt
        else:
            base = -dt / c
        shaping = phi_after - phi_before if self.config.kind == RewardKind.V4_PBRS else 0.0
        return RewardBreakdown(
            reward=base + shaping,
            base_reward=base,
            shaping_reward=shaping,
            delta_t=dt,
            phi_before=float(phi_before),
            phi_after=float(phi_after),
            l_remaining=remaining_time_lower_bound(simulator_after),
            failed=False,
        )

    def failure_step(
        self,
        simulator: FixedRouteSimulator,
        *,
        decision_time: float,
        phi_before: float,
    ) -> RewardBreakdown:
        """Terminal failure reward.

        Conventions (documented in ``docs/V4_REWARD_DESIGN.md``):
        - Absorbing terminal potential Φ_abs = 0.
        - ``decision_time`` is the pre-action clock.
        - ``L_remaining`` is evaluated on the current (failure) simulator state.

        V3_TIME / V4_BASE:
            r = -(H - t0)/scale - L_fail/scale
            (scale=1 for V3_TIME; scale=C_train for V4_BASE)

        V4_BASE_NO_L_FAIL:
            r = -(H - t0)/C_train

        V4_PBRS (no L in the *base* failure term; shaping supplies −Φ):
            r_base = -(H - t0)/C_train
            F = Φ_abs - Φ(s_t) = -Φ(s_t) = L(s_t)/C_train
            r = r_base + F
            Cumulative shaped failure return telescopes to (L_0 - H)/C_train.
        """
        t0 = float(decision_time)
        h = float(simulator.horizon)
        l_fail = remaining_time_lower_bound(simulator)
        c = float(self.config.c_train)
        kind = self.config.kind

        if kind == RewardKind.V3_TIME:
            base = -(h - t0) - l_fail
            shaping = 0.0
            phi_after = 0.0
        elif kind == RewardKind.V4_BASE:
            base = (-(h - t0) - l_fail) / c
            shaping = 0.0
            phi_after = 0.0
        elif kind == RewardKind.V4_BASE_NO_L_FAIL:
            base = -(h - t0) / c
            shaping = 0.0
            phi_after = 0.0
        elif kind == RewardKind.V4_PBRS:
            base = -(h - t0) / c
            phi_after = 0.0  # absorbing terminal
            shaping = phi_after - float(phi_before)
        else:
            raise ValueError(f"unknown reward kind: {kind}")

        return RewardBreakdown(
            reward=base + shaping,
            base_reward=base,
            shaping_reward=shaping,
            delta_t=0.0,
            phi_before=float(phi_before),
            phi_after=float(phi_after),
            l_remaining=float(l_fail),
            failed=True,
        )
