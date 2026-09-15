import numpy as np
import math
from typing import Dict, List, Tuple, Any

class HelperFunctions:
    """Utility functions for the EV charging station selection system"""
    
    # ==================== DISTANCE CALCULATIONS ====================
    
    @staticmethod
    def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate Euclidean distance between two points
        
        Args:
            x1, y1: Coordinates of first point
            x2, y2: Coordinates of second point
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @staticmethod
    def distance_between_nodes(node1: Dict, node2: Dict) -> float:
        """
        Calculate distance between two nodes (customers/stations)
        
        Args:
            node1: First node with 'x' and 'y' keys
            node2: Second node with 'x' and 'y' keys
            
        Returns:
            Distance between nodes
        """
        return HelperFunctions.euclidean_distance(
            node1['x'], node1['y'], node2['x'], node2['y']
        )
    
    @staticmethod
    def get_nearest_station_distance(position: Tuple[float, float], 
                                     stations: Dict) -> float:
        """
        Find distance to nearest station from given position
        
        Args:
            position: (x, y) coordinates
            stations: Dictionary of stations
            
        Returns:
            Distance to nearest station
        """
        min_dist = float('inf')
        for station in stations.values():
            dist = HelperFunctions.euclidean_distance(
                position[0], position[1], station['x'], station['y']
            )
            min_dist = min(min_dist, dist)
        return min_dist if min_dist != float('inf') else 0.0
    
    # ==================== BATTERY & CHARGING CHECKS ====================
    
    @staticmethod
    def check_charging_needed(vehicle_range: float, 
                             vehicle_position: Tuple[float, float],
                             next_node: Dict, 
                             stations: Dict) -> bool:
        """
        Check if vehicle needs to charge based on your logic:
        - Can't reach next node, OR
        - After reaching next node, can't reach any station
        
        Args:
            vehicle_range: Current vehicle range
            vehicle_position: Current (x, y) position
            next_node: Next node to visit
            stations: Available stations
            
        Returns:
            True if charging needed, False otherwise
        """
        # Distance to next node
        dist_to_next = HelperFunctions.euclidean_distance(
            vehicle_position[0], vehicle_position[1],
            next_node['x'], next_node['y']
        )
        
        # Can't reach next node
        if vehicle_range < dist_to_next:
            return True
        
        # Can reach next node, but check if safe to continue
        next_position = (next_node['x'], next_node['y'])
        dist_to_nearest_station = HelperFunctions.get_nearest_station_distance(
            next_position, stations
        )
        
        # After reaching next node, can't reach any station
        if vehicle_range - dist_to_next < dist_to_nearest_station:
            return True
        
        return False
    
    @staticmethod
    def get_feasible_stations(vehicle_position: Tuple[float, float],
                             vehicle_range: float, 
                             stations: Dict,
                             current_time: float = 0.0,
                             next_node: Dict = None) -> List[str]:
        """
        Get list of feasible stations:
        1. Reachable with current battery
        2. From station, can reach next node before its due_date
        
        Args:
            vehicle_position: Current (x, y) position
            vehicle_range: Current battery range
            stations: Dictionary of all stations
            current_time: Current time
            next_node: Next node to visit (with x, y, due_time)
            
        Returns:
            List of feasible station IDs
        """
        feasible = []
        
        for station_id, station in stations.items():
            dist_to_station = HelperFunctions.euclidean_distance(
                vehicle_position[0], vehicle_position[1],
                station['x'], station['y']
            )
            
            # Check battery - can reach station?
            if dist_to_station > vehicle_range:
                continue
            
            # If no next_node provided, skip time check
            if next_node is None:
                feasible.append(station_id)
                continue
            
            # Check if from station we can reach next node before its due_date
            arrival_at_station = current_time + dist_to_station
            
            dist_to_next = HelperFunctions.euclidean_distance(
                station['x'], station['y'],
                next_node['x'], next_node['y']
            )
            arrival_at_next = arrival_at_station + dist_to_next
            due_date = next_node.get('close_time', float('inf'))
            
            if arrival_at_next <= due_date:
                feasible.append(station_id)

        # print("from helper", next_node)
        
        return feasible
    
    @staticmethod
    def calculate_charging_time(energy_needed: float, charging_power: float) -> float:
        """
        Calculate time needed to charge
        
        Args:
            energy_needed: Energy to charge (kWh or km)
            charging_power: Charging power of station (kW or km/unit)
            
        Returns:
            Charging time in time units
        """
        if charging_power <= 0:
            return 0.0
        return energy_needed / charging_power
    
    # ==================== k-NN GRAPH CONSTRUCTION ====================
    
    @staticmethod
    def create_knn_adjacency_matrix(stations: List[Dict], k: int = 3) -> np.ndarray:
        """
        Create k-NN adjacency matrix for stations based on geographic distance
        
        Args:
            stations: List of station dictionaries with 'x' and 'y'
            k: Number of nearest neighbors
            
        Returns:
            Adjacency matrix (num_stations, num_stations)
        """
        n = len(stations)
        adj_matrix = np.zeros((n, n), dtype=np.float32)
        
        if n <= 1:
            return adj_matrix
        
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = HelperFunctions.euclidean_distance(
                        stations[i]['x'], stations[i]['y'],
                        stations[j]['x'], stations[j]['y']
                    )
                    distances.append((dist, j))
            
            # Sort by distance and connect to k nearest
            distances.sort()
            k_actual = min(k, len(distances))
            for _, j in distances[:k_actual]:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0  # Symmetric
        
        return adj_matrix
    
    # ==================== ROUTE FEASIBILITY ====================
    
    @staticmethod
    def check_route_feasibility(vehicle_position: Tuple[float, float],
                                vehicle_range: float, 
                                remaining_route: List[Dict],
                                stations: Dict,
                                current_time: float = 0.0) -> bool:
        """
        Check if remaining route is feasible with current battery
        
        Args:
            vehicle_position: Current position
            vehicle_range: Current battery range
            remaining_route: List of remaining nodes to visit
            stations: Available stations
            current_time: Current time
            
        Returns:
            True if route is feasible, False otherwise
        """
        if not remaining_route:
            return True
        
        pos = vehicle_position
        battery = vehicle_range
        time = current_time
        
        for i, node in enumerate(remaining_route):
            dist = HelperFunctions.euclidean_distance(
                pos[0], pos[1], node['x'], node['y']
            )
            
            # Can't reach this node
            if battery < dist:
                # Check if any station is reachable (next_node = current node we're trying to reach)
                feasible_stations = HelperFunctions.get_feasible_stations(
                    pos, battery, stations, time, node
                )
                if not feasible_stations:
                    return False  # No reachable stations, route infeasible
                else:
                    return True  # Can charge, route still feasible
            
            # Update for next iteration
            battery -= dist
            time += dist
            pos = (node['x'], node['y'])
        
        return True
    
    # ==================== TIME CALCULATIONS ====================
    
    @staticmethod
    def calculate_travel_time(from_pos: Tuple[float, float],
                             to_pos: Tuple[float, float],
                             speed: float = 1.0) -> float:
        """
        Calculate travel time between two positions
        
        Args:
            from_pos: Starting position (x, y)
            to_pos: Destination position (x, y)
            speed: Travel speed (default: 1.0)
            
        Returns:
            Travel time
        """
        distance = HelperFunctions.euclidean_distance(
            from_pos[0], from_pos[1], to_pos[0], to_pos[1]
        )
        return distance / speed
    
    @staticmethod
    def calculate_arrival_time(current_time: float, 
                              from_pos: Tuple[float, float],
                              to_pos: Tuple[float, float],
                              speed: float = 1.0) -> float:
        """
        Calculate arrival time at destination
        
        Args:
            current_time: Current time
            from_pos: Starting position
            to_pos: Destination position
            speed: Travel speed
            
        Returns:
            Arrival time
        """
        travel_time = HelperFunctions.calculate_travel_time(from_pos, to_pos, speed)
        return current_time + travel_time
    
    # ==================== TIME WINDOW UTILITIES ====================
    
    @staticmethod
    def is_time_window_violated(arrival_time: float, customer: Dict) -> bool:
        """
        Check if arrival violates customer time window
        
        Args:
            arrival_time: Arrival time at customer
            customer: Customer dictionary with 'ready_time' and 'due_time'
            
        Returns:
            True if violated (arrive after due_time), False otherwise
        """
        due_time = customer.get('due_time', float('inf'))
        return arrival_time > due_time
    
    @staticmethod
    def get_waiting_time(arrival_time: float, customer: Dict) -> float:
        """
        Calculate waiting time if arriving before ready time
        
        Args:
            arrival_time: Arrival time at customer
            customer: Customer dictionary with 'ready_time'
            
        Returns:
            Waiting time (0 if arrive after ready time)
        """
        ready_time = customer.get('ready_time', 0.0)
        if arrival_time < ready_time:
            return ready_time - arrival_time
        return 0.0
    
    @staticmethod
    def calculate_lateness(arrival_time: float, customer: Dict) -> float:
        """
        Calculate lateness penalty
        
        Args:
            arrival_time: Arrival time at customer
            customer: Customer dictionary with 'due_time'
            
        Returns:
            Lateness (0 if on time)
        """
        due_time = customer.get('due_time', float('inf'))
        if arrival_time > due_time:
            return arrival_time - due_time
        return 0.0
    
    # ==================== NORMALIZATION UTILITIES ====================
    
    @staticmethod
    def normalize_value(value: float, max_value: float) -> float:
        """
        Normalize value to [0, 1] range
        
        Args:
            value: Value to normalize
            max_value: Maximum value for normalization
            
        Returns:
            Normalized value in [0, 1]
        """
        if max_value <= 0:
            return 0.0
        return min(value / max_value, 1.0)
    
    @staticmethod
    def normalize_array(array: np.ndarray, max_value: float) -> np.ndarray:
        """
        Normalize numpy array to [0, 1] range
        
        Args:
            array: Array to normalize
            max_value: Maximum value for normalization
            
        Returns:
            Normalized array
        """
        if max_value <= 0:
            return np.zeros_like(array)
        return np.minimum(array / max_value, 1.0)
    
    # ==================== VECTOR OPERATIONS ====================
    
    @staticmethod
    def get_position_tuple(node: Dict) -> Tuple[float, float]:
        """
        Extract position tuple from node dictionary
        
        Args:
            node: Node with 'x' and 'y' keys
            
        Returns:
            (x, y) tuple
        """
        return (node['x'], node['y'])
    
    @staticmethod
    def calculate_route_distance(route: List[Dict]) -> float:
        """
        Calculate total distance of a route
        
        Args:
            route: List of nodes with 'x' and 'y' coordinates
            
        Returns:
            Total route distance
        """
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += HelperFunctions.distance_between_nodes(
                route[i], route[i + 1]
            )
        return total_distance
    
    @staticmethod
    def calculate_remaining_route_distance(current_pos: Tuple[float, float],
                                          remaining_nodes: List[Dict]) -> float:
        """
        Calculate distance of remaining route from current position
        
        Args:
            current_pos: Current (x, y) position
            remaining_nodes: List of remaining nodes to visit
            
        Returns:
            Total remaining distance
        """
        if not remaining_nodes:
            return 0.0
        
        total_distance = 0.0
        
        # Distance to first node
        total_distance += HelperFunctions.euclidean_distance(
            current_pos[0], current_pos[1],
            remaining_nodes[0]['x'], remaining_nodes[0]['y']
        )
        
        # Distance between remaining nodes
        for i in range(len(remaining_nodes) - 1):
            total_distance += HelperFunctions.distance_between_nodes(
                remaining_nodes[i], remaining_nodes[i + 1]
            )
        
        return total_distance
    
    # ==================== DEPOT UTILITIES ====================
    
    @staticmethod
    def is_depot(node_id: str) -> bool:
        """
        Check if node is depot
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if depot, False otherwise
        """
        return node_id == 'D0' or node_id.lower() == 'depot'
    
    @staticmethod
    def create_depot_as_station(depot: Dict) -> Dict:
        """
        Create depot as a station with zero cost
        
        Args:
            depot: Depot node with 'x' and 'y'
            
        Returns:
            Station dictionary for depot
        """
        return {
            'x': depot['x'],
            'y': depot['y'],
            'price': 0.0,
            'waiting_time': 0.0,
            'charging_power': float('inf')  # Instant charging at depot
        }