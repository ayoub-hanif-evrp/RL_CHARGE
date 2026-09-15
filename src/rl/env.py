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
    evaluate_shield,
)
from simulation.simulator import FixedRouteSimulator

from .ablation import AblationConfig
from .features import FeatureBundle, extract_features
from .normalization import Normalizer


@dataclass
class StepInfo:
    reward: float
    done: bool
    failed: bool
    features: FeatureBundle
    reason: Optional[InfeasibilityReason] = None
    extra: dict = field(default_factory=dict)


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
    ):
        self.instance = instance
        self.route = route
        self.load_convention = require_load_convention(load_convention)
        self.profile = profile or PhysicsProfile.from_instance(instance)
        self.normalizer = normalizer
        self.ablation = ablation or AblationConfig()
        self.simulator = FixedRouteSimulator(
            instance,
            route.customer_ids,
            self.profile,
            self.load_convention,
            charging_model=charging_model,
        )
        self.return_value = 0.0
        self.failed = False

    @property
    def horizon(self) -> float:
        return self.simulator.horizon

    def _features(self) -> FeatureBundle:
        return extract_features(
            self.simulator,
            self.normalizer,
            use_remaining_route=self.ablation.use_remaining_route,
            use_terrain_load_features=self.ablation.use_terrain_load_features,
        )

    def reset(self) -> FeatureBundle:
        self.simulator.reset()
        self.return_value = 0.0
        self.failed = False
        return self._features()

    def observe(self) -> FeatureBundle:
        return self._features()

    def step(self, discrete_index: int, u: float = 0.0) -> StepInfo:
        if self.ablation.discrete_u:
            from baselines.discrete_ppo import snap_u

            u = snap_u(u)
        t0 = self.simulator.state.time.value
        h = self.horizon
        shield = evaluate_shield(self.simulator)
        if not shield.any_legal:
            reward = -(h - t0)
            self.return_value += reward
            self.failed = True
            return StepInfo(
                reward=reward,
                done=True,
                failed=True,
                features=self._features(),
                reason=InfeasibilityReason.NO_FEASIBLE_ACTION,
            )
        if discrete_index >= len(shield.mask) or not shield.mask[discrete_index]:
            reward = -(h - t0)
            self.return_value += reward
            self.failed = True
            return StepInfo(
                reward=reward,
                done=True,
                failed=True,
                features=self._features(),
                reason=InfeasibilityReason.INVALID_STATE,
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
            reward = -(h - t0)
            self.return_value += reward
            self.failed = True
            return StepInfo(
                reward=reward,
                done=True,
                failed=True,
                features=self._features(),
                reason=result.reason,
            )
        reward = -(t1 - t0)
        self.return_value += reward
        done = self.simulator.state.completed
        return StepInfo(
            reward=reward,
            done=done,
            failed=False,
            features=self._features(),
            extra={"discrete_index": discrete_index, "u": u, "continue": discrete_index == CONTINUE_INDEX},
        )
