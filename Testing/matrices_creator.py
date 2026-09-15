import numpy as np
from typing import Dict, List, Tuple, Any
from helper_functions import HelperFunctions
from data_manager import DataManager

class MatricesCreator:
    """Creates state representation matrices for the RL agent"""
    
    def __init__(self, data_manager: DataManager):
        """
        Initialize MatricesCreator
        
        Args:
            data_manager: DataManager instance for data access
        """
        self.data_manager = data_manager
        self.helper = HelperFunctions()
        
        # Get normalization constants
        self.norm_constants = data_manager.get_normalization_constants()
    
    # ==================== STATION GRAPH CREATION ====================
    
    def create_station_graph(self, instance_id: str, vehicle_id: str, 
                            current_position: Tuple[float, float],
                            current_battery: float,
                            current_time: float = 0.0,
                            route: List[str] = None,
                            current_route_index: int = 0,
                            k_neighbors: int = 3) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Create station graph with k-NN adjacency matrix and feature matrix
        
        Args:
            instance_id: Instance identifier
            vehicle_id: Vehicle identifier
            current_position: Current vehicle position (x, y)
            current_battery: Current battery range
            current_time: Current time
            route: Vehicle route
            current_route_index: Current position in route
            k_neighbors: Number of neighbors for k-NN graph
            
        Returns:
            Tuple of:
                - station_ids: List of feasible station IDs
                - adjacency_matrix: (num_stations, num_stations)
                - feature_matrix: (num_stations, 5) normalized features
        """
        # Get stations
        all_stations = self.data_manager.get_stations(instance_id)
        
        # Get customers
        all_customers = self.data_manager.get_customers(instance_id)
        
        # Get next node in route for time window check
        next_node = None
        if route and current_route_index + 1 < len(route):
            next_node_id = route[current_route_index + 1]
            if next_node_id in all_customers:
                next_node = all_customers[next_node_id]

        # print("from matrix ", next_node)
        
        # Get feasible stations (reachable + can reach next node on time)
        feasible_station_ids = self.helper.get_feasible_stations(
            current_position, current_battery, all_stations, current_time, next_node
        )
        
        if not feasible_station_ids:
            # No feasible stations
            return [], np.array([]), np.array([])
        
        # Get station data
        feasible_stations = [all_stations[sid] for sid in feasible_station_ids]
        num_stations = len(feasible_stations)
        
        # Create k-NN adjacency matrix
        adjacency_matrix = self.helper.create_knn_adjacency_matrix(
            feasible_stations, k=k_neighbors
        )
        
        # Create feature matrix: [x, y, price, waiting_time, charging_power]
        feature_matrix = np.zeros((num_stations, 5), dtype=np.float32)
        
        max_dist = self.norm_constants['max_distance']
        max_wait = self.norm_constants['max_waiting_time']
        max_price = self.norm_constants['max_price']
        
        for i, (station_id, station) in enumerate(zip(feasible_station_ids, feasible_stations)):
            # Normalize coordinates
            x_norm = station['x'] / max_dist
            y_norm = station['y'] / max_dist
            
            # Normalize price
            price = station.get('price', 1.0)
            price_norm = self.helper.normalize_value(price, max_price)
            
            # Normalize waiting time
            waiting_time = station.get('waiting_time', 0.0)
            waiting_norm = self.helper.normalize_value(waiting_time, max_wait)
            
            # Normalize charging power (inverse - lower is worse)
            charging_power = station.get('charging_power', 10.0)
            charging_power_norm = min(charging_power / 100.0, 1.0)  # Assume max 100 kW
            
            feature_matrix[i] = [x_norm, y_norm, price_norm, waiting_norm, charging_power_norm]
        
        return feasible_station_ids, adjacency_matrix, feature_matrix
    
    # ==================== CUSTOMER MATRIX CREATION ====================
    
    def create_customer_matrix(self, instance_id: str, vehicle_id: str,
                               current_position: Tuple[float, float],
                               route: List[str], current_route_index: int) -> Tuple[List[str], np.ndarray]:
        """
        Create customer feature matrix for remaining customers in route
        
        Args:
            instance_id: Instance identifier
            vehicle_id: Vehicle identifier
            current_position: Current vehicle position
            route: Complete vehicle route
            current_route_index: Current index in route
            
        Returns:
            Tuple of:
                - customer_ids: List of remaining customer IDs
                - feature_matrix: (num_customers, 4) normalized features
        """
        # Get all customers
        all_customers = self.data_manager.get_customers(instance_id)
        
        # Get remaining customers in route (after current position)
        remaining_customer_ids = []
        remaining_customers = []
        
        for i in range(current_route_index + 1, len(route)):
            node_id = route[i]
            if node_id in all_customers:
                remaining_customer_ids.append(node_id)
                remaining_customers.append(all_customers[node_id])
        
        if not remaining_customers:
            # No remaining customers
            return [], np.array([])
        
        num_customers = len(remaining_customers)
        
        # Create feature matrix: [ready_time, due_time, service_time, distance_from_current]
        feature_matrix = np.zeros((num_customers, 4), dtype=np.float32)
        
        max_time = self.norm_constants['max_time']
        max_service = self.norm_constants['max_service_time']
        max_dist = self.norm_constants['max_distance']
        
        for i, customer in enumerate(remaining_customers):
            # Normalize time windows
            ready_time = customer.get('ready_time', 0.0)
            due_time = customer.get('close_time', 1000.0)
            ready_norm = self.helper.normalize_value(ready_time, max_time)
            due_norm = self.helper.normalize_value(due_time, max_time)
            
            # Normalize service time
            service_time = customer.get('service_time', 0.0)
            service_norm = self.helper.normalize_value(service_time, max_service)
            
            # Distance from current position
            distance = self.helper.euclidean_distance(
                current_position[0], current_position[1],
                customer['x'], customer['y']
            )
            distance_norm = self.helper.normalize_value(distance, max_dist)
            
            feature_matrix[i] = [ready_norm, due_norm, service_norm, distance_norm]
        
        return remaining_customer_ids, feature_matrix
    
    # ==================== VEHICLE FEATURES CREATION ====================
    
    def create_vehicle_features(self, vehicle_id: str, current_battery: float,
                            current_time: float, next_customer: Dict,
                            current_position: Tuple[float, float],
                            remaining_route_distance: float) -> np.ndarray:
        """
        Create vehicle feature vector
        
        Args:
            vehicle_id: Vehicle identifier
            current_battery: Current battery range
            current_time: Current time in route
            next_customer: Next customer to visit (or None)
            current_position: Current position
            remaining_route_distance: Distance of remaining route
            
        Returns:
            Vehicle feature vector (7,) normalized  
        """
        # Get fleet specs
        fleet_specs = self.data_manager.get_fleet_specs(vehicle_id)
        max_battery = self.data_manager.get_max_battery_capacity(vehicle_id)
        
        # Distance to next customer
        if next_customer is not None:
            dist_to_next = self.helper.euclidean_distance(
                current_position[0], current_position[1],
                next_customer['x'], next_customer['y']
            )
        else:
            dist_to_next = 0.0
        
        # Normalize features
        max_batt = self.norm_constants['max_battery']
        max_time = self.norm_constants['max_time']
        max_dist = self.norm_constants['max_distance']
        
        battery_norm = self.helper.normalize_value(current_battery, max_batt)
        max_battery_norm = self.helper.normalize_value(max_battery, max_batt)
        time_norm = self.helper.normalize_value(current_time, max_time)
        dist_next_norm = self.helper.normalize_value(dist_to_next, max_dist)
        remaining_dist_norm = self.helper.normalize_value(remaining_route_distance, max_dist)
        
        # ADD: Normalize current position
        x_norm = current_position[0] / max_dist
        y_norm = current_position[1] / max_dist
        
        # Create feature vector 
        vehicle_features = np.array([
            x_norm,
            y_norm,
            battery_norm,
            max_battery_norm,
            time_norm,
            dist_next_norm,
            remaining_dist_norm,
        ], dtype=np.float32)
        
        return vehicle_features
    
    # ==================== COMPLETE STATE CREATION ====================
    
    def create_state(self, instance_id: str, vehicle_id: str,
                    current_position: Tuple[float, float],
                    current_battery: float, current_time: float,
                    route: List[str], current_route_index: int,
                    k_neighbors: int = 3) -> Dict[str, Any]:
        """
        Create complete state representation
        
        Args:
            instance_id: Instance identifier
            vehicle_id: Vehicle identifier
            current_position: Current vehicle position
            current_battery: Current battery range
            current_time: Current time
            route: Complete vehicle route
            current_route_index: Current position in route
            k_neighbors: k for k-NN graph
            
        Returns:
            State dictionary with:
                - station_ids: List of feasible station IDs
                - station_features: (num_stations, 5)
                - adj_matrix: (num_stations, num_stations)
                - customer_features: (num_customers, 4)
                - vehicle_features: (7,)
                - num_stations: Number of feasible stations
        """
        # Create station graph
        station_ids, adj_matrix, station_features = self.create_station_graph(
            instance_id, vehicle_id, current_position, current_battery, current_time, route, current_route_index, k_neighbors
        )
        
        # Create customer matrix
        customer_ids, customer_features = self.create_customer_matrix(
            instance_id, vehicle_id, current_position, route, current_route_index
        )
        
        # Get next customer
        all_customers = self.data_manager.get_customers(instance_id)
        next_customer = None
        if current_route_index + 1 < len(route):
            next_node_id = route[current_route_index + 1]
            if next_node_id in all_customers:
                next_customer = all_customers[next_node_id]
        
        # Calculate remaining route distance
        remaining_nodes = []
        for i in range(current_route_index + 1, len(route)):
            node_id = route[i]
            if node_id in all_customers:
                remaining_nodes.append(all_customers[node_id])
        
        remaining_route_distance = self.helper.calculate_remaining_route_distance(
            current_position, remaining_nodes
        )
        
        # Create vehicle features
        vehicle_features = self.create_vehicle_features(
            vehicle_id, current_battery, current_time, next_customer,
            current_position, remaining_route_distance
        )
        
        # Build state dictionary
        state = {
            'station_ids': station_ids,
            'station_features': station_features,
            'adj_matrix': adj_matrix,
            'customer_features': customer_features,
            'vehicle_features': vehicle_features,
            'num_stations': len(station_ids)
        }
        
        return state
    
    # ==================== UTILITY METHODS ====================
    
    def get_station_position(self, instance_id: str, station_id: str) -> Tuple[float, float]:
        """
        Get station position
        
        Args:
            instance_id: Instance identifier
            station_id: Station identifier
            
        Returns:
            (x, y) position
        """
        station = self.data_manager.get_station(instance_id, station_id)
        if station:
            return (station['x'], station['y'])
        return (0.0, 0.0)
    
    def get_customer_position(self, instance_id: str, customer_id: str) -> Tuple[float, float]:
        """
        Get customer position
        
        Args:
            instance_id: Instance identifier
            customer_id: Customer identifier
            
        Returns:
            (x, y) position
        """
        customer = self.data_manager.get_customer(instance_id, customer_id)
        if customer:
            return (customer['x'], customer['y'])
        return (0.0, 0.0)
    
    def add_depot_as_station(self, instance_id: str, station_ids: List[str],
                            station_features: np.ndarray, 
                            adj_matrix: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Add depot as a zero-cost station option
        
        Args:
            instance_id: Instance identifier
            station_ids: Current station IDs
            station_features: Current station features
            adj_matrix: Current adjacency matrix
            
        Returns:
            Updated (station_ids, station_features, adj_matrix) with depot added
        """
        depot = self.data_manager.get_depot(instance_id)
        if depot is None:
            return station_ids, station_features, adj_matrix
        
        # Create depot station features (all zeros except position)
        max_dist = self.norm_constants['max_distance']
        depot_features = np.array([[
            depot['x'] / max_dist,  # x normalized
            depot['y'] / max_dist,  # y normalized
            0.0,  # price = 0
            0.0,  # waiting_time = 0
            1.0   # charging_power = max (instant)
        ]], dtype=np.float32)
        
        # Add to station features
        if len(station_features) > 0:
            station_features = np.vstack([station_features, depot_features])
        else:
            station_features = depot_features
        
        # Expand adjacency matrix
        n = len(station_ids)
        new_adj = np.zeros((n + 1, n + 1), dtype=np.float32)
        if n > 0:
            new_adj[:n, :n] = adj_matrix
            # Connect depot to all stations (fully connected)
            new_adj[n, :n] = 1.0
            new_adj[:n, n] = 1.0
        
        # Add depot ID
        station_ids.append('D0')
        
        return station_ids, station_features, new_adj