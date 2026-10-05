"""Thin simulator + shield wrapper. Gym is not required."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from domain.load_convention import LoadConvention, require_load_convention
from physics.parameters import PhysicsProfile
from routing.fixed_route import FrozenRoute
from simulation.feasibility import InfeasibilityReason
from simulation.shield import (
    CONTINUE_INDEX,
    action_from_discrete,
    dead_end_diagnostic,
    evaluate_shield,
)
from simulation.simulator import FixedRouteSimulator

from .ablation import AblationConfig
from .features import FeatureBundle, extract_features
from .normalization import Normalizer
from .rewards import RewardComputer, RewardConfig


@dataclass
class StepInfo:
    reward: float
    done: bool
    failed: bool
    features: FeatureBundle
    reason: Optional[InfeasibilityReason] = None
    extra: dict = field(default_factory=dict)
    executed_discrete: int = 0
    executed_u: float = 0.0


class ShieldedRouteEnv:
    def __init__(
        self,
        instance,
        route: FrozenRoute,
        profile: Optional[PhysicsProfile] = None,
        load_convention=LoadConvention.OFFICIAL_REFERENCE_PICKUP,
        normalizer: Optional[Normalizer] = None,
        ablation: Optional[AblationConfig] = None,
        charging_model=None,
        reward_config: Optional[RewardConfig] = None,
    ):
        self.instance = instance
        self.route = route
        self.load_convention = require_load_convention(load_convention)
        self.profile = profile or PhysicsProfile.from_instance(instance)
        self.normalizer = normalizer
        self.ablation = ablation or AblationConfig()
        self.reward_config = reward_config or RewardConfig.v3_time()
        self.reward_computer = RewardComputer(self.reward_config)
        self.simulator = FixedRouteSimulator(
            instance,
            route.customer_ids,
            self.profile,
            self.load_convention,
            charging_model=charging_model,
        )
        self.simulator.time_aware_envelope = bool(self.ablation.time_aware)
        self.return_value = 0.0
        self.base_return_value = 0.0
        self.shaping_return_value = 0.0
        # Scientific objective return: -T on success, -H on failure (no L, no shaping).
        self.objective_return_value = 0.0
        self.normalized_base_return_value = 0.0
        self.failed = False
        self._phi = 0.0

    @property
    def horizon(self) -> float:
        return self.simulator.horizon

    def _features(self) -> FeatureBundle:
        return extract_features(
            self.simulator,
            self.normalizer,
            use_remaining_route=self.ablation.use_remaining_route,
            use_terrain_load_features=self.ablation.use_terrain_load_features,
            soc_interval=self.ablation.soc_interval,
        )

    def reset(self) -> FeatureBundle:
        self.simulator.reset()
        self.return_value = 0.0
        self.base_return_value = 0.0
        self.shaping_return_value = 0.0
        self.objective_return_value = 0.0
        self.normalized_base_return_value = 0.0
        self.failed = False
        self._phi = self.reward_computer.potential(self.simulator)
        return self._features()

    def _finalize_objective(self, *, failed: bool) -> None:
        """Set objective / normalized-base returns at episode end.

        Objective (unnormalized): -T on success, -H on failure.
        Normalized base: objective / C_train (for V3_TIME, c_train defaults to 1).
        """
        if failed:
            self.objective_return_value = -float(self.horizon)
        else:
            self.objective_return_value = -float(self.simulator.state.time.value)
        c = float(self.reward_config.c_train)
        self.normalized_base_return_value = self.objective_return_value / c

    def observe(self) -> FeatureBundle:
        return self._features()

    def _apply_breakdown(self, breakdown) -> dict:
        self.return_value += float(breakdown.reward)
        self.base_return_value += float(breakdown.base_reward)
        self.shaping_return_value += float(breakdown.shaping_reward)
        self._phi = float(breakdown.phi_after)
        return {
            "reward_kind": self.reward_config.kind.value,
            "c_train": float(self.reward_config.c_train),
            "base_reward": float(breakdown.base_reward),
            "shaping_reward": float(breakdown.shaping_reward),
            "phi_before": float(breakdown.phi_before),
            "phi_after": float(breakdown.phi_after),
            "l_remaining": float(breakdown.l_remaining),
            "delta_t": float(breakdown.delta_t),
        }

    def _failure(self, *, t0: float, reason: InfeasibilityReason, executed: dict, extra_update: dict | None = None) -> StepInfo:
        breakdown = self.reward_computer.failure_step(
            self.simulator,
            decision_time=t0,
            phi_before=self._phi,
        )
        reward_extra = self._apply_breakdown(breakdown)
        self.failed = True
        self._finalize_objective(failed=True)
        reward_extra.update(
            {
                "objective_return": float(self.objective_return_value),
                "normalized_base_return": float(self.normalized_base_return_value),
                "training_return": float(self.return_value),
                "shaping_return": float(self.shaping_return_value),
            }
        )
        extra = {**executed["extra"], **reward_extra}
        if extra_update:
            extra.update(extra_update)
        return StepInfo(
            reward=float(breakdown.reward),
            done=True,
            failed=True,
            features=self._features(),
            reason=reason,
            executed_discrete=executed["executed_discrete"],
            executed_u=executed["executed_u"],
            extra=extra,
        )

    def step(self, discrete_index: int, u: float = 0.0) -> StepInfo:
        if discrete_index == CONTINUE_INDEX:
            u = 0.0
        t0 = self.simulator.state.time.value
        shield = evaluate_shield(self.simulator)
        executed = {
            "executed_discrete": int(discrete_index),
            "executed_u": float(u),
            "extra": {
                "discrete_index": discrete_index,
                "u": u,
                "continue": discrete_index == CONTINUE_INDEX,
            },
        }
        if not shield.any_legal:
            return self._failure(
                t0=t0,
                reason=InfeasibilityReason.NO_FEASIBLE_ACTION,
                executed=executed,
                extra_update={"dead_end": dead_end_diagnostic(self.simulator)},
            )
        if discrete_index >= len(shield.mask) or not shield.mask[discrete_index]:
            return self._failure(
                t0=t0,
                reason=InfeasibilityReason.INVALID_STATE,
                executed=executed,
            )
        action = action_from_discrete(
            self.simulator,
            discrete_index,
            u,
            soc_mode=self.ablation.soc_interval,
        )
        result = self.simulator.step(action)
        t1 = self.simulator.state.time.value
        if not result.feasible:
            return self._failure(
                t0=t0,
                reason=result.reason,
                executed=executed,
            )
        breakdown = self.reward_computer.feasible_step(
            delta_t=t1 - t0,
            phi_before=self._phi,
            simulator_after=self.simulator,
        )
        reward_extra = self._apply_breakdown(breakdown)
        done = self.simulator.state.completed
        if done:
            # Absorbing terminal: force Φ = 0 for bookkeeping consistency.
            self._phi = 0.0
            self._finalize_objective(failed=False)
            reward_extra.update(
                {
                    "objective_return": float(self.objective_return_value),
                    "normalized_base_return": float(self.normalized_base_return_value),
                    "training_return": float(self.return_value),
                    "shaping_return": float(self.shaping_return_value),
                }
            )
        return StepInfo(
            reward=float(breakdown.reward),
            done=done,
            failed=False,
            features=self._features(),
            executed_discrete=executed["executed_discrete"],
            executed_u=executed["executed_u"],
            extra={**executed["extra"], **reward_extra},
        )
