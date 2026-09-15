"""Deterministic fixed-route charging simulator.

The customer order is immutable. The simulator does not decide *when* to
charge; the caller supplies ``ContinueAction`` or ``ChargeAction``. There is
no rule-based charging trigger and no discrete six-level SOC grid.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from data.models import EVRPTWGRInstance, Node, NodeType
from domain.load_convention import (
    LoadConvention,
    initial_payload_kg,
    payload_after_service_kg,
    require_load_convention,
)
from domain.quantities import BatteryEnergy, Energy, PayloadMass, SocFraction, TravelTime
from physics.battery import BatteryModel
from physics.charging import BenchmarkCompatibleLinearChargingModel, ChargingModel
from physics.energy import EnergyModel
from physics.network import DirectedArcNetwork
from physics.parameters import PhysicsProfile

from .actions import Action, ChargeAction, ContinueAction
from .events import ChargeEvent, CustomerServiceEvent, DepotEvent, TravelEvent
from .feasibility import FeasibilityService, InfeasibilityReason, ZERO_CHARGE_EPS
from .metrics import TrajectoryMetrics
from .state import SimulatorState


@dataclass(frozen=True)
class TransitionResult:
    feasible: bool
    reason: Optional[InfeasibilityReason]
    state: SimulatorState
    metrics: TrajectoryMetrics


class FixedRouteSimulator:
    def __init__(
        self,
        instance: EVRPTWGRInstance,
        customer_ids: Sequence[str],
        profile: PhysicsProfile,
        load_convention,
        charging_model: Optional[ChargingModel] = None,
        network: Optional[DirectedArcNetwork] = None,
        energy_model: Optional[EnergyModel] = None,
        battery_model: Optional[BatteryModel] = None,
        ignore_energy: bool = False,
    ):
        self.instance = instance
        self.customer_ids: Tuple[str, ...] = tuple(customer_ids)
        if len(set(self.customer_ids)) != len(self.customer_ids):
            raise ValueError("frozen customer sequence contains duplicates")
        customers = {node.string_id for node in instance.customers}
        for cid in self.customer_ids:
            if cid not in customers:
                raise ValueError(f"{cid} is not a customer of this instance")
        self.load_convention = require_load_convention(load_convention)
        self.profile = profile
        self.ignore_energy = bool(ignore_energy)
        self.depot_id = instance.depot.string_id
        self.network = network or DirectedArcNetwork(instance, profile.average_velocity)
        self.energy_model = energy_model or EnergyModel(self.network, profile)
        self.battery_model = battery_model or BatteryModel(profile)
        self.charging_model = charging_model or BenchmarkCompatibleLinearChargingModel(
            profile.inverse_refueling_rate
        )
        self.feasibility = FeasibilityService(self.energy_model, self.battery_model)
        self.state = self._initial_state()

    @property
    def horizon(self) -> float:
        return float(self.instance.depot.due_date)

    @property
    def load_convention_name(self) -> str:
        return self.load_convention.value

    def next_frozen_node_id(self) -> str:
        if self.state.next_customer_index < len(self.customer_ids):
            return self.customer_ids[self.state.next_customer_index]
        return self.depot_id

    def remaining_customer_ids(self) -> Tuple[str, ...]:
        return self.customer_ids[self.state.next_customer_index :]

    def clone(self) -> "FixedRouteSimulator":
        cloned = FixedRouteSimulator(
            self.instance,
            self.customer_ids,
            self.profile,
            self.load_convention,
            charging_model=self.charging_model,
            network=self.network,
            energy_model=self.energy_model,
            battery_model=self.battery_model,
            ignore_energy=self.ignore_energy,
        )
        cloned.state = copy.deepcopy(self.state)
        return cloned

    def _initial_state(self) -> SimulatorState:
        battery = self.battery_model.initial_state()
        payload = PayloadMass(
            initial_payload_kg(self.load_convention, self.instance, self.customer_ids)
        )
        if payload.value > self.profile.payload_capacity_kg + 1e-9:
            # Still construct the state; the first service/step records the violation.
            pass
        state = SimulatorState(
            current_node_id=self.depot_id,
            time=TravelTime(0.0),
            battery_energy=battery.energy,
            soc=battery.soc,
            payload=payload,
            next_customer_index=0,
            n_station_visits_since_last_customer=0,
            completed=False,
        )
        state.events.append(
            DepotEvent(
                kind="depot_depart_ready",
                node_id=self.depot_id,
                time=state.time,
                battery=state.battery_energy,
                payload=state.payload,
            )
        )
        return state

    def reset(self) -> SimulatorState:
        self.state = self._initial_state()
        return self.state

    def metrics(
        self, feasible: bool = True, reason: Optional[InfeasibilityReason] = None
    ) -> TrajectoryMetrics:
        return self.state.metrics.snapshot(
            terminal_soc=self.state.soc,
            completion_time=self.state.time,
            feasible=feasible and self.state.completed,
            reason=reason,
            load_convention=self.load_convention_name,
        )

    def step(self, action: Action) -> TransitionResult:
        if self.state.completed:
            return self._fail(InfeasibilityReason.INVALID_STATE)
        if isinstance(action, ContinueAction):
            return self._continue()
        if isinstance(action, ChargeAction):
            return self._charge(action)
        return self._fail(InfeasibilityReason.INVALID_STATE)

    def _continue(self) -> TransitionResult:
        if self.state.next_customer_index < len(self.customer_ids):
            target_id = self.customer_ids[self.state.next_customer_index]
            node = self.network.node(target_id)
            travel = self._travel(target_id)
            if travel is not None:
                return travel
            service = self._serve_customer(node)
            if service is not None:
                return service
            self.state.next_customer_index += 1
            self.state.n_station_visits_since_last_customer = 0
            return self._ok()
        travel = self._travel(self.depot_id)
        if travel is not None:
            return travel
        depot = self.network.node(self.depot_id)
        if self.state.time.value > depot.due_date + 1e-12:
            self.state.metrics.time_window_violations += 1
            return self._fail(InfeasibilityReason.TIME_WINDOW_VIOLATION)
        self.state.completed = True
        self.state.events.append(
            DepotEvent(
                kind="depot_arrival",
                node_id=self.depot_id,
                time=self.state.time,
                battery=self.state.battery_energy,
                payload=self.state.payload,
            )
        )
        return self._ok()

    def _charge(self, action: ChargeAction) -> TransitionResult:
        if self.state.n_station_visits_since_last_customer >= self.profile.loop_guard_station_visits:
            return self._fail(InfeasibilityReason.LOOP_GUARD)
        try:
            node = self.network.node(action.station_id)
        except KeyError:
            return self._fail(InfeasibilityReason.UNKNOWN_STATION)
        if node.node_type is not NodeType.STATION:
            return self._fail(InfeasibilityReason.UNKNOWN_STATION)
        travel = self._travel(action.station_id)
        if travel is not None:
            return travel
        battery = self.battery_model.state_from_soc(self.state.soc.value)
        target_soc = float(action.target_soc)
        if not (battery.min_soc_fraction - 1e-12 <= target_soc <= battery.max_soc_fraction + 1e-12):
            return self._fail(InfeasibilityReason.INVALID_TARGET_SOC)
        if target_soc + 1e-12 < self.state.soc.value:
            return self._fail(InfeasibilityReason.INVALID_TARGET_SOC)
        if (
            self.state.current_node_id == action.station_id
            and target_soc < self.state.soc.value + ZERO_CHARGE_EPS
        ):
            return self._fail(InfeasibilityReason.ZERO_CHARGE_NOOP)
        target_energy = BatteryEnergy(target_soc * battery.capacity.value)
        try:
            charge = self.charging_model.charging_time(
                energy_before=self.state.battery_energy,
                energy_target=target_energy,
                battery=battery,
                station_id=action.station_id,
                current_time=self.state.time,
            )
        except Exception:
            return self._fail(InfeasibilityReason.INVALID_TARGET_SOC)
        applied = self.battery_model.apply_charge(battery, charge.energy_added)
        if not applied.feasible:
            return self._fail(InfeasibilityReason.INVALID_TARGET_SOC)
        arrival = self.state.time
        departure = TravelTime(arrival.value + charge.charging_duration.value)
        self.state.time = departure
        self.state.battery_energy = applied.after.energy
        self.state.soc = applied.after.soc
        self.state.n_station_visits_since_last_customer += 1
        self.state.metrics.number_of_station_visits += 1
        self.state.metrics.total_charging_time += charge.charging_duration.value
        self.state.metrics.total_energy_charged += charge.energy_added.value
        self.state.events.append(
            ChargeEvent(
                kind="charge",
                station_id=action.station_id,
                soc_before=charge.soc_before,
                soc_target=charge.soc_target,
                soc_after=applied.after.soc,
                energy_added=charge.energy_added,
                charging_duration=charge.charging_duration,
                arrival_time=arrival,
                departure_time=departure,
                payload=self.state.payload,
            )
        )
        return self._ok()

    def _travel(self, to_id: str) -> Optional[TransitionResult]:
        check = self.feasibility.can_reach(
            self.state.current_node_id, to_id, self.state.payload, self._battery()
        )
        if not self.ignore_energy and not check.battery_feasible:
            return self._fail(InfeasibilityReason.INSUFFICIENT_ENERGY)
        if self.ignore_energy:
            applied_energy = Energy(0.0)
            battery_after = self.state.battery_energy
            soc_after = self.state.soc
            ceiling_hit = False
            consumed = Energy(0.0)
            recovered = Energy(0.0)
            net = Energy(0.0)
        else:
            applied = self.battery_model.apply_arc_energy(self._battery(), check.energy.net_energy)
            applied_energy = check.energy.net_energy
            battery_after = applied.after.energy
            soc_after = applied.after.soc
            ceiling_hit = applied.ceiling_hit
            consumed = check.energy.consumed_energy
            recovered = check.energy.recovered_energy
            net = check.energy.net_energy
        arc = self.network.arc(self.state.current_node_id, to_id)
        departure = self.state.time
        arrival = TravelTime(departure.value + arc.travel_time.value)
        self.state.metrics.total_distance += arc.distance.value
        self.state.metrics.total_travel_time += arc.travel_time.value
        self.state.metrics.total_energy_consumed += consumed.value
        self.state.metrics.total_energy_regenerated += recovered.value
        self.state.metrics.total_net_energy += net.value
        self.state.events.append(
            TravelEvent(
                kind="travel",
                from_node=self.state.current_node_id,
                to_node=to_id,
                distance=arc.distance,
                travel_time=arc.travel_time,
                altitude_difference=arc.delta_altitude,
                gradient_percent=arc.gradient_percent,
                payload_before=self.state.payload,
                energy_net=net,
                energy_consumed=consumed,
                energy_recovered=recovered,
                battery_before=self.state.battery_energy,
                battery_after=battery_after,
                departure_time=departure,
                arrival_time=arrival,
                ceiling_hit=ceiling_hit,
            )
        )
        self.state.current_node_id = to_id
        self.state.time = arrival
        self.state.battery_energy = battery_after
        self.state.soc = soc_after
        return None

    def _serve_customer(self, node: Node) -> Optional[TransitionResult]:
        arrival = self.state.time
        waiting = TravelTime(max(0.0, node.ready_time - arrival.value))
        service_start = TravelTime(arrival.value + waiting.value)
        if service_start.value > node.due_date + 1e-12:
            self.state.metrics.time_window_violations += 1
            return self._fail(InfeasibilityReason.TIME_WINDOW_VIOLATION)
        new_payload_value = payload_after_service_kg(
            self.load_convention, self.state.payload.value, node.demand
        )
        if new_payload_value < -1e-9:
            return self._fail(InfeasibilityReason.CAPACITY_VIOLATION)
        if new_payload_value > self.profile.payload_capacity_kg + 1e-9:
            return self._fail(InfeasibilityReason.CAPACITY_VIOLATION)
        service_time = TravelTime(node.service_time)
        departure = TravelTime(service_start.value + service_time.value)
        payload_before = self.state.payload
        payload_after = PayloadMass(new_payload_value)
        self.state.metrics.total_waiting_at_customers += waiting.value
        self.state.metrics.total_service_time += service_time.value
        self.state.events.append(
            CustomerServiceEvent(
                kind="customer_service",
                customer_id=node.string_id,
                ready_time=node.ready_time,
                due_date=node.due_date,
                waiting_time=waiting,
                service_start=service_start,
                service_end=departure,
                service_time=service_time,
                demand=node.demand,
                payload_before=payload_before,
                payload_after=payload_after,
            )
        )
        self.state.time = departure
        self.state.payload = payload_after
        self.state.served_customers.append(node.string_id)
        return None

    def _battery(self):
        return self.battery_model.state_from_soc(self.state.soc.value)

    def _snapshot(
        self, feasible: bool, reason: Optional[InfeasibilityReason]
    ) -> TrajectoryMetrics:
        return self.state.metrics.snapshot(
            terminal_soc=self.state.soc,
            completion_time=self.state.time,
            feasible=feasible,
            reason=reason,
            load_convention=self.load_convention_name,
        )

    def _ok(self) -> TransitionResult:
        return TransitionResult(
            feasible=True,
            reason=None,
            state=self.state,
            metrics=self._snapshot(self.state.completed, None),
        )

    def _fail(self, reason: InfeasibilityReason) -> TransitionResult:
        return TransitionResult(
            feasible=False,
            reason=reason,
            state=self.state,
            metrics=self._snapshot(False, reason),
        )


def run_continue_only(simulator: FixedRouteSimulator) -> TransitionResult:
    """Replay a frozen route with Continue actions only. Not a route filter."""
    result = TransitionResult(
        feasible=True,
        reason=None,
        state=simulator.state,
        metrics=simulator.metrics(feasible=False),
    )
    while not simulator.state.completed:
        result = simulator.step(ContinueAction())
        if not result.feasible:
            return result
    return result
