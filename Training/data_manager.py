import json
from pathlib import Path
from typing import Dict, Any, Optional, List

class DataManager:
    """Manages and caches all data loading operations"""
    
    def __init__(self, 
                 dataset_path: str = "train_dataset.json",
                 fleet_path: str = "fleet.json"):
        """
        Initialize DataManager
        
        Args:
            dataset_path: Path to training dataset JSON file
            fleet_path: Path to fleet specifications JSON file
        """
        self.dataset_path = Path(dataset_path)
        self.fleet_path = Path(fleet_path)
        
        # Cache for loaded data
        self._dataset_cache = None
        self._fleet_cache = None
        
        # Normalization constants (computed from dataset)
        self.norm_constants = {}
        
        # Load all data
        self._load_all_data()
        self._compute_normalization_constants()
    
    def _load_all_data(self):
        """Load all data files into cache"""
        try:
            # Load training dataset
            with open(self.dataset_path, 'r') as f:
                self._dataset_cache = json.load(f)
            print(f"✓ Loaded dataset with {len(self._dataset_cache)} instances")
            
            # Load fleet specifications
            with open(self.fleet_path, 'r') as f:
                self._fleet_cache = json.load(f)
            print(f"✓ Loaded fleet data with {len(self._fleet_cache)} vehicles")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required data file not found: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in data file: {e}")
    
    def _compute_normalization_constants(self):
        """Compute normalization constants from dataset statistics"""
        max_distance = 0.0
        max_waiting_time = 0.0
        max_price = 0.0
        max_battery = 0.0
        max_service_time = 0.0
        max_time_window = 0.0
        
        # Iterate through all instances to find maximums
        for instance in self._dataset_cache.values():
            
            # Check stations
            for station in instance.get('stations', {}).values():
                max_waiting_time = max(max_waiting_time, station.get('waiting_time', 0.0))
                max_price = max(max_price, station.get('price', 0.0))
                
                # Calculate distances
                for other_station in instance.get('stations', {}).values():
                    dist = ((station['x'] - other_station['x'])**2 + 
                           (station['y'] - other_station['y'])**2)**0.5
                    max_distance = max(max_distance, dist)
            
            # Check customers
            for customer in instance.get('customers', {}).values():
                max_service_time = max(max_service_time, customer.get('service_time', 0.0))
                ready_time = customer.get('ready_time', 0.0)
                due_time = customer.get('close_time', 0.0)
                max_time_window = max(max_time_window, due_time)
                
                # Calculate distances
                for other_customer in instance.get('customers', {}).values():
                    dist = ((customer['x'] - other_customer['x'])**2 + 
                           (customer['y'] - other_customer['y'])**2)**0.5
                    max_distance = max(max_distance, dist)
        
        # Check fleet for max battery
        for vehicle_specs in self._fleet_cache.values():
            max_battery = max(max_battery, vehicle_specs.get('driving_range_km', 250.0))
        
        # Store normalization constants
        self.norm_constants = {
            'max_distance': max_distance if max_distance > 0 else 1000.0,
            'max_waiting_time': max_waiting_time if max_waiting_time > 0 else 100.0,
            'max_charging_time': 100.0,  # Estimated
            'max_price': max_price if max_price > 0 else 50.0,
            'max_battery': max_battery if max_battery > 0 else 200.0,
            'max_lateness': max_time_window * 2 if max_time_window > 0 else 500.0,
            'max_service_time': max_service_time if max_service_time > 0 else 50.0,
            'max_time': max_time_window if max_time_window > 0 else 1000.0
        }
        
        print(f"✓ Computed normalization constants")
    
    # ==================== INSTANCE OPERATIONS ====================
    
    def get_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get instance by ID
        
        Args:
            instance_id: Instance identifier (key in dataset)
            
        Returns:
            Instance dictionary or None if not found
        """
        instance_id = str(instance_id)
        return self._dataset_cache.get(instance_id)
    
    def get_all_instances(self) -> Dict[str, Any]:
        """
        Get all instances
        
        Returns:
            Dictionary of all instances
        """
        return self._dataset_cache
    
    def get_num_instances(self) -> int:
        """Get total number of instances"""
        return len(self._dataset_cache)
    
    # ==================== VEHICLE OPERATIONS ====================
    
    def get_vehicle_data(self, instance_id: str, vehicle_id: str) -> tuple:
        """
        Get vehicle data from instance
        
        Args:
            instance_id: Instance identifier
            vehicle_id: Vehicle identifier (e.g., "V20")
            
        Returns:
            Tuple of (vehicle_dict, instance_dict)
            
        Raises:
            ValueError: If instance or vehicle not found
        """
        instance = self.get_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance {instance_id} not found")
        
        # Search for vehicle by ID field
        for v_key, vehicle in instance.get("vehicles", {}).items():
            if vehicle.get("id") == vehicle_id:
                return vehicle, instance
        
        raise ValueError(f"Vehicle {vehicle_id} not found in instance {instance_id}")
    
    def get_all_vehicles_in_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Get all vehicles in an instance
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Dictionary of vehicles
        """
        instance = self.get_instance(instance_id)
        if not instance:
            return {}
        return instance.get('vehicles', {})
    
    # ==================== FLEET OPERATIONS ====================
    
    def get_fleet_specs(self, vehicle_id: str) -> Dict[str, Any]:
        """
        Get fleet specifications for a vehicle
        
        Args:
            vehicle_id: Vehicle identifier (e.g., "V20")
            
        Returns:
            Fleet specifications dictionary
            
        Raises:
            ValueError: If vehicle not found in fleet data
        """
        if vehicle_id not in self._fleet_cache:
            raise ValueError(f"Vehicle {vehicle_id} not found in fleet data")
        return self._fleet_cache[vehicle_id]
    
    def get_all_fleet_data(self) -> Dict[str, Any]:
        """Get all fleet data"""
        return self._fleet_cache
    
    def get_max_battery_capacity(self, vehicle_id: str) -> float:
        """
        Get maximum battery capacity for vehicle
        
        Args:
            vehicle_id: Vehicle identifier
            
        Returns:
            Maximum driving range (battery capacity)
        """
        fleet_specs = self.get_fleet_specs(vehicle_id)
        return float(fleet_specs.get('driving_range_km', 200.0))
    
    def get_initial_battery(self, vehicle_id: str) -> float:
        """
        Get initial battery level for vehicle
        
        Args:
            vehicle_id: Vehicle identifier
            
        Returns:
            Initial remaining range
        """
        fleet_specs = self.get_fleet_specs(vehicle_id)
        return float(fleet_specs.get('remaining_driving_range_km', 
                                    fleet_specs.get('driving_range_km', 200.0)))
    
    # ==================== CUSTOMER OPERATIONS ====================
    
    def get_customers(self, instance_id: str) -> Dict[str, Any]:
        """
        Get all customers in an instance
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Dictionary of customers
        """
        instance = self.get_instance(instance_id)
        if not instance:
            return {}
        return instance.get('customers', {})
    
    def get_customer(self, instance_id: str, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific customer
        
        Args:
            instance_id: Instance identifier
            customer_id: Customer identifier
            
        Returns:
            Customer dictionary or None
        """
        customers = self.get_customers(instance_id)
        return customers.get(customer_id)
    
    # ==================== STATION OPERATIONS ====================
    
    def get_stations(self, instance_id: str) -> Dict[str, Any]:
        """
        Get all stations in an instance
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Dictionary of stations
        """
        instance = self.get_instance(instance_id)
        if not instance:
            return {}
        return instance.get('stations', {})
    
    def get_station(self, instance_id: str, station_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific station
        
        Args:
            instance_id: Instance identifier
            station_id: Station identifier
            
        Returns:
            Station dictionary or None
        """
        stations = self.get_stations(instance_id)
        return stations.get(station_id)
    
    def get_depot(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get depot node (D0)
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Depot dictionary or None
        """
        customers = self.get_customers(instance_id)
        return customers.get('D0')
    
    # ==================== ROUTE OPERATIONS ====================
    
    def get_vehicle_route(self, instance_id: str, vehicle_id: str) -> List[str]:
        """
        Get route for a vehicle
        
        Args:
            instance_id: Instance identifier
            vehicle_id: Vehicle identifier
            
        Returns:
            List of node IDs in route
        """
        vehicle, _ = self.get_vehicle_data(instance_id, vehicle_id)
        return vehicle.get('route', [])
    
    # ==================== NORMALIZATION CONSTANTS ====================
    
    def get_normalization_constants(self) -> Dict[str, float]:
        """
        Get all normalization constants
        
        Returns:
            Dictionary of normalization constants
        """
        return self.norm_constants.copy()
    
    def get_max_distance(self) -> float:
        """Get maximum distance for normalization"""
        return self.norm_constants['max_distance']
    
    def get_max_waiting_time(self) -> float:
        """Get maximum waiting time for normalization"""
        return self.norm_constants['max_waiting_time']
    
    def get_max_charging_time(self) -> float:
        """Get maximum charging time for normalization"""
        return self.norm_constants['max_charging_time']
    
    def get_max_price(self) -> float:
        """Get maximum price for normalization"""
        return self.norm_constants['max_price']
    
    def get_max_battery(self) -> float:
        """Get maximum battery for normalization"""
        return self.norm_constants['max_battery']
    
    def get_max_lateness(self) -> float:
        """Get maximum lateness for normalization"""
        return self.norm_constants['max_lateness']
    
    # ==================== STATISTICS ====================
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        total_vehicles = 0
        total_customers = 0
        total_stations = 0
        
        for instance in self._dataset_cache.values():
            total_vehicles += len(instance.get('vehicles', {}))
            total_customers += len(instance.get('customers', {}))
            total_stations += len(instance.get('stations', {}))
        
        return {
            'num_instances': len(self._dataset_cache),
            'total_vehicles': total_vehicles,
            'total_customers': total_customers,
            'total_stations': total_stations,
            'avg_vehicles_per_instance': total_vehicles / len(self._dataset_cache) if self._dataset_cache else 0,
            'avg_customers_per_instance': total_customers / len(self._dataset_cache) if self._dataset_cache else 0,
            'avg_stations_per_instance': total_stations / len(self._dataset_cache) if self._dataset_cache else 0,
            'num_vehicle_types': len(self._fleet_cache)
        }
    
    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_dataset_statistics()
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Instances: {stats['num_instances']}")
        print(f"Total Vehicles: {stats['total_vehicles']}")
        print(f"Total Customers: {stats['total_customers']}")
        print(f"Total Stations: {stats['total_stations']}")
        print(f"Avg Vehicles/Instance: {stats['avg_vehicles_per_instance']:.1f}")
        print(f"Avg Customers/Instance: {stats['avg_customers_per_instance']:.1f}")
        print(f"Avg Stations/Instance: {stats['avg_stations_per_instance']:.1f}")
        print(f"Vehicle Types: {stats['num_vehicle_types']}")
        print("="*60 + "\n")