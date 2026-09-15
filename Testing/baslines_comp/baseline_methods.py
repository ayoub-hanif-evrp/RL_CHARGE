# Baseline Methods for Charging Station Selection
# All baselines charge to 100% (portion = 1.0)

from helper_functions import HelperFunctions

helper = HelperFunctions()


def select_nearest_station(feasible_stations, current_pos, stations_data):
    """Select station closest to current position."""
    best_id = None
    best_dist = float('inf')

    #print("feasible_stations",stations_data)
    
    for sid in feasible_stations:
        s = stations_data[sid]
        dist = helper.euclidean_distance(current_pos[0], current_pos[1], s['x'], s['y'])
        if dist < best_dist:
            best_dist = dist
            best_id = sid
    
    return best_id


def select_cheapest_station(feasible_stations, stations_data):
    """Select station with lowest price."""
    best_id = None
    best_price = float('inf')
    
    for sid in feasible_stations:
        price = stations_data[sid].get('price', 1.0)
        #print("from scs",price)
        if price < best_price:
            best_price = price
            best_id = sid
    
    return best_id


def select_earliest_service(feasible_stations, current_pos, current_battery, max_battery, stations_data):
    """Select station with minimum service time (travel + wait + charge)."""
    best_id = None
    best_time = float('inf')
    
    for sid in feasible_stations:
        s = stations_data[sid]
        
        # Travel time = distance
        dist = helper.euclidean_distance(current_pos[0], current_pos[1], s['x'], s['y'])
        
        # Charging time
        battery_at_station = current_battery - dist
        energy_needed = max_battery - battery_at_station
        charging_power = s.get('charging_power', 10.0)
        charge_time = energy_needed / charging_power if charging_power > 0 else 0
        
        # Total service time
        total_time = dist + s.get('waiting_time', 0) + charge_time
        
        if total_time < best_time:
            best_time = total_time
            best_id = sid
    
    return best_id


def select_rulebased_station(feasible_stations, current_pos, current_battery, max_battery, stations_data):
    """Select station with minimum weighted score (normalized)."""
    if not feasible_stations:
        return None
    
    # Weights
    w_dist, w_wait, w_price, w_charge = 0.25, 0.25, 0.25, 0.25
    
    # Collect values
    values = []
    for sid in feasible_stations:
        s = stations_data[sid]
        dist = helper.euclidean_distance(current_pos[0], current_pos[1], s['x'], s['y'])
        battery_at = current_battery - dist
        energy = max_battery - battery_at
        power = s.get('charging_power', 10.0)
        charge_time = energy / power if power > 0 else 0
        
        values.append({
            'id': sid,
            'dist': dist,
            'wait': s.get('waiting_time', 0),
            'price': s.get('price', 1.0),
            'charge': charge_time
        })
    
    # Get max for normalization
    max_dist = max(v['dist'] for v in values) or 1
    max_wait = max(v['wait'] for v in values) or 1
    max_price = max(v['price'] for v in values) or 1
    max_charge = max(v['charge'] for v in values) or 1
    
    # Find best
    best_id = None
    best_score = float('inf')
    
    for v in values:
        score = (w_dist * v['dist']/max_dist + 
                 w_wait * v['wait']/max_wait + 
                 w_price * v['price']/max_price + 
                 w_charge * v['charge']/max_charge)
        if score < best_score:
            best_score = score
            best_id = v['id']
    
    return best_id