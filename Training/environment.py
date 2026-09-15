import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from data_manager import DataManager
from matrices_creator import MatricesCreator
from helper_functions import HelperFunctions

class Environment:
    """Environment for EV charging station selection with RL"""
    
    def __init__(self, data_manager: DataManager):
        """
        Initialize Environment
        
        Args:
            data_manager: DataManager instance
        """
        self.data_manager = data_manager
        self.matrices_creator = MatricesCreator(data_manager)
        self.helper = HelperFunctions()
        
        # Get normalization constants
        self.norm_constants = data_manager.get_normalization_constants()
        
        # Episode state
        self.current_instance_id = None
        self.current_vehicle_id = None
        self.current_vehicle = None
        self.current_instance = None
        self.route = None
        self.route_index = None
        self.current_position = None
        self.current_battery = None
        self.current_time = None
        
        # Reward weights
        self.cost_weights = {
            'w1': 0.2,  # Distance to station
            'w2': 0.2,  # Distance station to next
            'w3': 0.2,  # Waiting time
            'w4': 0.2,  # Charging time
            'w5': 0.2   # Price
        }
        self.alpha = 0.75  # Late margins weight
        self.beta = 0.25   # Battery waste weight
    
    # ==================== EPISODE MANAGEMENT ====================
    
    def reset(self, instance_id: str, vehicle_id: str) -> Dict[str, Any]:
        """
        Reset environment for new episode
        
        Args:
            instance_id: Instance identifier
            vehicle_id: Vehicle identifier
            
        Returns:
            Initial state or None if vehicle doesn't need charging
        """
        # Load instance and vehicle data
        self.current_instance_id = instance_id
        self.current_vehicle_id = vehicle_id
        self.current_vehicle, self.current_instance = self.data_manager.get_vehicle_data(
            instance_id, vehicle_id
        )
        
        # Initialize route
        self.route = self.current_vehicle['route']
        self.route_index = 0  # Start at depot
        
        # Initialize battery
        self.current_battery = self.data_manager.get_initial_battery(vehicle_id)
        self.max_battery = self.data_manager.get_max_battery_capacity(vehicle_id)
        
        # Initialize time
        self.current_time = 0.0
        
        # Initialize position (depot)
        depot = self.data_manager.get_depot(instance_id)
        if depot:
            self.current_position = (depot['x'], depot['y'])
        else:
            self.current_position = (self.current_vehicle['x'], self.current_vehicle['y'])
        
        # Move through route until charging is needed
        return self._advance_to_next_charging_point(just_charged=False)
    
    def _advance_to_next_charging_point(self, just_charged: bool = False) -> Optional[Dict[str, Any]]:
        """
        Advance through route until charging is needed
        
        Args:
            just_charged: If True, move to next customer first before checking
        
        Returns:
            State dict if charging needed, None if episode complete
        """
        customers = self.data_manager.get_customers(self.current_instance_id)
        stations = self.data_manager.get_stations(self.current_instance_id)
        
        # Move through route
        while self.route_index < len(self.route) - 1:
            next_node_id = self.route[self.route_index + 1]
            
            # Skip if not a customer (shouldn't happen but safety check)
            if next_node_id not in customers:
                self.route_index += 1
                continue
            
            next_node = customers[next_node_id]
            
            # Calculate distance to next node
            dist = self.helper.euclidean_distance(
                self.current_position[0], self.current_position[1],
                next_node['x'], next_node['y']
            )
            
            # Check if we can reach next node
            if self.current_battery < dist:
                # Can't reach next node - need charging
                if just_charged:
                    # Just charged but still can't reach - infeasible
                    return None
                
                state = self.matrices_creator.create_state(
                    self.current_instance_id,
                    self.current_vehicle_id,
                    self.current_position,
                    self.current_battery,
                    self.current_time,
                    self.route,
                    self.route_index
                )
                
                if state['num_stations'] == 0:
                    return None
                
                return state
            
            # Can reach next node - move there
            self.current_battery -= dist
            self.current_time += dist
            self.current_time += next_node.get('service_time', 0.0)
            self.current_position = (next_node['x'], next_node['y'])
            self.route_index += 1
            just_charged = False  # Reset flag after moving
            
            # Now check if we need charging for the NEXT node
            if self.route_index < len(self.route) - 1:
                next_next_id = self.route[self.route_index + 1]
                if next_next_id in customers:
                    next_next_node = customers[next_next_id]
                    if self.helper.check_charging_needed(
                        self.current_battery, self.current_position, next_next_node, stations
                    ):
                        state = self.matrices_creator.create_state(
                            self.current_instance_id,
                            self.current_vehicle_id,
                            self.current_position,
                            self.current_battery,
                            self.current_time,
                            self.route,
                            self.route_index
                        )
                        
                        if state['num_stations'] == 0:
                            return None
                        
                        return state
        
        # Route completed
        return None
    
    # ==================== STEP FUNCTION ====================
    
    def step(self, station_id: str, charge_portion: float) -> Tuple[Optional[Dict], float, bool]:
        """
        Execute charging action and advance route
        
        Args:
            station_id: Selected station ID
            charge_portion: Charge portion (0.2 to 1.0)
            
        Returns:
            Tuple of (next_state, reward, done)
            - next_state: Next state dict or None
            - reward: Reward for this action
            - done: True if episode finished
        """
        # Get station
        station = self.data_manager.get_station(self.current_instance_id, station_id)
        if not station:
            # Invalid station
            return None, -1000.0, True
        
        # Calculate reward BEFORE executing action
        reward = self._calculate_reward(station_id, charge_portion)
        
        # Execute charging action
        # Travel to station
        station_pos = (station['x'], station['y'])
        dist_to_station = self.helper.euclidean_distance(
            self.current_position[0], self.current_position[1],
            station_pos[0], station_pos[1]
        )
        # print("dist_to_station",dist_to_station)
        
        self.current_battery -= dist_to_station
        travel_time = dist_to_station
        self.current_time += travel_time
        
        # Wait at station
        waiting_time = station.get('waiting_time', 0.0)
        self.current_time += waiting_time
        
        # Charge battery
        energy_needed = (self.max_battery - self.current_battery) * charge_portion
        charging_power = station.get('charging_power', 10.0)
        charging_time = self.helper.calculate_charging_time(energy_needed, charging_power)
        self.current_time += charging_time
        self.current_battery += energy_needed
        
        # Update position
        self.current_position = station_pos
        
        # Check if route is still feasible
        remaining_nodes = []
        customers = self.data_manager.get_customers(self.current_instance_id)
        for i in range(self.route_index + 1, len(self.route)):
            node_id = self.route[i]
            if node_id in customers:
                remaining_nodes.append(customers[node_id])
        
        stations = self.data_manager.get_stations(self.current_instance_id)
        is_feasible = self.helper.check_route_feasibility(
            self.current_position, self.current_battery, remaining_nodes, stations
        )
        
        if not is_feasible:
            # Route became infeasible after charging
            return None, reward, True
        
        # Advance to next charging point (just_charged=True to move first)
        next_state = self._advance_to_next_charging_point(just_charged=True)
        
        if next_state is None:
            # Episode complete (reached end of route)
            return None, reward, True
        
        # Episode continues
        return next_state, reward, False
    
    # ==================== REWARD CALCULATION ====================
    
    def _calculate_reward(self, station_id: str, charge_portion: float) -> float:
        """
        Calculate reward for charging action
        
        Args:
            station_id: Selected station ID
            charge_portion: Charge portion (0.2 to 1.0)
            
        Returns:
            Reward (negative cost)
        """
        station = self.data_manager.get_station(self.current_instance_id, station_id)
        customers = self.data_manager.get_customers(self.current_instance_id)
        
        # Get next customer
        next_customer = None
        if self.route_index + 1 < len(self.route):
            next_node_id = self.route[self.route_index + 1]
            if next_node_id in customers:
                next_customer = customers[next_node_id]
        
        # Calculate immediate cost
        immediate_cost = self._calculate_immediate_cost(
            station, next_customer, charge_portion
        )
        
        # Calculate future impact
        future_impact = self._calculate_future_impact(
            station, charge_portion
        )
        
        # Total reward (negative because we minimize cost)
        reward = -(immediate_cost + future_impact)
        # print("reward =>", reward)
        # print("immediate_cost",immediate_cost)
        # print("future_impact", future_impact)
        
        return reward
    
    def _calculate_immediate_cost(self, station: Dict, next_customer: Optional[Dict],
                                  charge_portion: float) -> float:
        """
        Calculate immediate cost (normalized)
        
        Args:
            station: Selected station
            next_customer: Next customer to visit
            charge_portion: Charge portion
            
        Returns:
            Normalized immediate cost
        """
        # Distance to station
        dist_to_station = self.helper.euclidean_distance(
            self.current_position[0], self.current_position[1],
            station['x'], station['y']
        )
        
        # Distance from station to next customer
        if next_customer:
            dist_to_next = self.helper.euclidean_distance(
                station['x'], station['y'],
                next_customer['x'], next_customer['y']
            )
        else:
            dist_to_next = 0.0
        
        # Waiting time
        waiting_time = station.get('waiting_time', 0.0)
        # print("waiting_time", waiting_time)
        
        # Charging time
        energy_needed = (self.max_battery - self.current_battery) * charge_portion
        charging_power = station.get('charging_power', 10.0)
        charging_time = self.helper.calculate_charging_time(energy_needed, charging_power)
        
        # Price
        price = station.get('price', 1.0)
        # print("price", price)
        total_price = price * energy_needed
        # print("total_price", total_price)
        # Normalize components
        # norm_dist_station = self.helper.normalize_value(
        #     dist_to_station, self.norm_constants['max_distance']
        # )
        norm_dist_station = self.helper.normalize_value(
            dist_to_station, 120
        )
        norm_dist_next = self.helper.normalize_value(
            dist_to_next, self.norm_constants['max_distance']
        )
        norm_waiting = self.helper.normalize_value(
            waiting_time, self.norm_constants['max_waiting_time']
        )
        norm_charging_time = self.helper.normalize_value(
            charging_time, self.norm_constants['max_charging_time']
        )
        norm_price = self.helper.normalize_value(
            price, self.norm_constants['max_price']
        )
        
        # Weighted sum
        immediate_cost = (
            self.cost_weights['w1'] * norm_dist_station +
            self.cost_weights['w2'] * norm_dist_next +
            self.cost_weights['w3'] * norm_waiting +
            self.cost_weights['w4'] * norm_charging_time +
            self.cost_weights['w5'] * norm_price
        )

        # print("immediate :",norm_dist_station,norm_dist_next,norm_waiting,norm_charging_time,norm_price)
        
        return immediate_cost
    
    def _calculate_future_impact(self, station: Dict, charge_portion: float) -> float:
        """
        Calculate future impact (normalized)
        
        Args:
            station: Selected station
            charge_portion: Charge portion
            
        Returns:
            Normalized future impact
        """
        customers = self.data_manager.get_customers(self.current_instance_id)
        
        # Simulate future from station position after charging
        simulated_pos = (station['x'], station['y'])
        simulated_time = self.current_time
        simulated_battery = self.current_battery
        
        # Add travel to station
        dist_to_station = self.helper.euclidean_distance(
            self.current_position[0], self.current_position[1],
            station['x'], station['y']
        )
        simulated_time += dist_to_station
        simulated_battery -= dist_to_station
        
        # Add waiting and charging
        simulated_time += station.get('waiting_time', 0.0)
        energy_needed = (self.max_battery - simulated_battery) * charge_portion
        charging_power = station.get('charging_power', 10.0)
        charging_time = self.helper.calculate_charging_time(energy_needed, charging_power)
        simulated_time += charging_time
        simulated_battery += energy_needed
        
        # Calculate late margins for remaining customers
        late_margins = 0.0
        
        for i in range(self.route_index + 1, len(self.route)):
            node_id = self.route[i]
            if node_id not in customers:
                continue
            
            customer = customers[node_id]
            
            # Travel to customer
            dist = self.helper.euclidean_distance(
                simulated_pos[0], simulated_pos[1],
                customer['x'], customer['y']
            )
            simulated_time += dist
            simulated_battery -= dist
            
            # Check lateness
            lateness = self.helper.calculate_lateness(simulated_time, customer)
            late_margins += lateness
            
            # Service time
            service_time = customer.get('service_time', 0.0)
            simulated_time += service_time
            
            # Update position
            simulated_pos = (customer['x'], customer['y'])
        
        # Battery at depot (last position)
        battery_at_depot = max(0.0, simulated_battery)
        
        # Normalize components
        norm_late_margins = self.helper.normalize_value(
            late_margins, self.norm_constants['max_lateness']
        )
        # norm_battery_waste = self.helper.normalize_value(
        #     battery_at_depot, self.norm_constants['max_battery']
        # )
        norm_battery_waste = self.helper.normalize_value(
            battery_at_depot, 250
        )
        
        # Future impact
        future_impact = self.alpha * norm_late_margins + self.beta * norm_battery_waste
        # print("future impact",norm_battery_waste, late_margins)
        
        return future_impact
    
    # ==================== UTILITY METHODS ====================
    
    def set_reward_weights(self, w1: float = None, w2: float = None, w3: float = None,
                          w4: float = None, w5: float = None, alpha: float = None,
                          beta: float = None):
        """
        Set reward function weights
        
        Args:
            w1: Weight for distance to station
            w2: Weight for distance station to next
            w3: Weight for waiting time
            w4: Weight for charging time
            w5: Weight for price
            alpha: Weight for late margins
            beta: Weight for battery waste
        """
        if w1 is not None:
            self.cost_weights['w1'] = w1
        if w2 is not None:
            self.cost_weights['w2'] = w2
        if w3 is not None:
            self.cost_weights['w3'] = w3
        if w4 is not None:
            self.cost_weights['w4'] = w4
        if w5 is not None:
            self.cost_weights['w5'] = w5
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
    
    def get_current_state_info(self) -> Dict[str, Any]:
        """
        Get current state information for debugging
        
        Returns:
            Dictionary with current state info
        """
        return {
            'instance_id': self.current_instance_id,
            'vehicle_id': self.current_vehicle_id,
            'route_index': self.route_index,
            'route_length': len(self.route) if self.route else 0,
            'current_position': self.current_position,
            'current_battery': self.current_battery,
            'max_battery': self.max_battery,
            'current_time': self.current_time
        }