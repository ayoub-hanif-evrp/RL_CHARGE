# Evaluation Helpers - Simulation and utility functions

from helper_functions import HelperFunctions
from baseline_methods import (
    select_nearest_station,
    select_cheapest_station,
    select_earliest_service,
    select_rulebased_station
)

helper = HelperFunctions()


def calculate_base_travel(route, customers):
    """Calculate base route travel distance (without detours)."""
    total = 0.0
    for i in range(len(route) - 1):
        n1, n2 = route[i], route[i + 1]
        if n1 in customers and n2 in customers:
            c1, c2 = customers[n1], customers[n2]
            total += helper.euclidean_distance(c1['x'], c1['y'], c2['x'], c2['y'])
    return total


def select_station(method, feasible_stations, current_pos, current_battery, max_battery, stations, agent, state):
    """Select station and portion based on method."""
    
    if method == 'RL':
        idx, p_idx = agent.select_action(state, training=False)
        return feasible_stations[idx], agent.charge_portions[p_idx]
    
    if method == 'NS':
        return select_nearest_station(feasible_stations, current_pos, stations), 1.0
    
    if method == 'CS':
        return select_cheapest_station(feasible_stations, stations), 1.0
    
    if method == 'ES':
        return select_earliest_service(feasible_stations, current_pos, current_battery, max_battery, stations), 1.0
    
    if method == 'RB':
        return select_rulebased_station(feasible_stations, current_pos, current_battery, max_battery, stations), 1.0
    
    raise ValueError(f"Unknown method: {method}")


def check_route_validity(env, agent, inst_key, vehicle_id, max_stops, max_feasible_stations):
    """
    Check if route is valid for evaluation.
    Returns (is_valid, rl_stops) or (False, -1) if invalid.
    
    A route is valid if:
    - RL needs between 1 and max_stops charging stops
    - Number of feasible stations never exceeds max_feasible_stations
    """
    state = env.reset(inst_key, vehicle_id)
    
    # No charging needed
    if state is None:
        return False, 0
    
    stops = 0
    done = False
    
    while not done:
        feasible_stations = state['station_ids']
        
        # Check feasible stations limit
        if len(feasible_stations) > max_feasible_stations:
            return False, -1
        
        if len(feasible_stations) == 0:
            return False, -1
        
        # RL action
        idx, p_idx = agent.select_action(state, training=False)
        station_id = feasible_stations[idx]
        portion = agent.charge_portions[p_idx]
        
        stops += 1
        
        # Too many stops
        if stops > max_stops:
            return False, -1
        
        state, _, done = env.step(station_id, portion)
    
    return True, stops


def simulate_route(env, agent, method, inst_key, vehicle_id, dm):
    """
    Simulate a route with given method and return metrics.
    
    Returns dict with: travel_cost, charging_cost, waiting_cost, total_cost, stops, final_battery
    """
    # Get data
    customers = dm.get_customers(inst_key)
    stations = dm.get_stations(inst_key)
    vehicle, _ = dm.get_vehicle_data(inst_key, vehicle_id)
    route = vehicle.get('route', [])
    fleet_specs = dm.get_fleet_specs(vehicle_id)
    max_battery = fleet_specs.get('driving_range_km', 200.0)
    
    # Reset
    state = env.reset(inst_key, vehicle_id)
    base_travel = calculate_base_travel(route, customers)
    
    # No charging needed
    if state is None:
        return {
            'travel_cost': base_travel,
            'charging_cost': 0.0,
            'waiting_cost': 0.0,
            'total_cost': base_travel,
            'stops': 0,
            'final_battery': env.current_battery
        }
    
    # Simulate
    detour = 0.0
    charge_cost = 0.0
    wait_cost = 0.0
    stops = 0
    done = False
    
    while not done:
        feasible = state['station_ids']
        if len(feasible) == 0:
            break
        
        pos = env.current_position
        battery = env.current_battery
        
        station_id, portion = select_station(method, feasible, pos, battery, max_battery, stations, agent, state)
        station = stations[station_id]
        
        # Costs
        dist = helper.euclidean_distance(pos[0], pos[1], station['x'], station['y'])
        detour += dist
        
        battery_at = battery - dist
        energy = portion * (max_battery - battery_at)
        
        charge_cost += energy * station.get('price', 1.0)
        wait_cost += station.get('waiting_time', 0)
        stops += 1
        
        state, _, done = env.step(station_id, portion)
    
    total_travel = base_travel + detour

    # print("from simulate_route", stops, method)
    
    return {
        'travel_cost': total_travel,
        'charging_cost': charge_cost,
        'waiting_cost': wait_cost,
        'total_cost': total_travel + charge_cost + wait_cost,
        'stops': stops,
        'final_battery': env.current_battery
    }