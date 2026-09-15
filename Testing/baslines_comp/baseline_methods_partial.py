# Baseline Methods for Charging Station Selection
# Now includes HYBRID partial charging (4 factors)

from helper_functions import HelperFunctions

helper = HelperFunctions()


# =============================================================================
# HYBRID PARTIAL CHARGING (4 Factors) - Simplified Version
# =============================================================================

def calculate_hybrid_portion(current_battery, max_battery, station, all_stations, route_index, total_customers):
    """
    Calculate charging portion using 4-factor hybrid approach.
    
    Factors:
        1. Battery Level - lower battery → charge more
        2. Station Price - cheaper → charge more
        3. Route Progress - earlier → charge more
        4. Station Quality - better station → charge more
    
    Returns: portion in [0.2, 0.4, 0.6, 0.8, 1.0]
    """
    
    # === BASE PORTION (higher base = more conservative) ===
    base = 0.60
    
    # === FACTOR 1: BATTERY LEVEL (reduced impact) ===
    battery_ratio = current_battery / max_battery
    
    if battery_ratio < 0.15:
        battery_adj = +0.20      # Very low → charge more
    elif battery_ratio < 0.30:
        battery_adj = +0.10      # Low → charge bit more
    elif battery_ratio > 0.60:
        battery_adj = -0.05      # High → slightly less
    else:
        battery_adj = 0.0        # Medium → no adjustment
    
    # === FACTOR 2: STATION PRICE (reduced impact) ===
    prices = [s.get('price', 1.0) for s in all_stations.values()]
    avg_price = sum(prices) / len(prices) if prices else 1.0
    station_price = station.get('price', 1.0)
    price_ratio = station_price / avg_price if avg_price > 0 else 1.0
    
    if price_ratio < 0.6:
        price_adj = +0.15        # Very cheap → charge more
    elif price_ratio < 0.85:
        price_adj = +0.05        # Below average → charge bit more
    elif price_ratio > 1.4:
        price_adj = -0.10        # Very expensive → charge less
    else:
        price_adj = 0.0          # Average → no adjustment
    
    # === FACTOR 3: ROUTE PROGRESS (reduced impact) ===
    progress = route_index / total_customers if total_customers > 0 else 0.5
    
    if progress < 0.20:
        progress_adj = +0.10     # Early → bit more
    elif progress > 0.90:
        progress_adj = -0.10     # Almost done → bit less
    else:
        progress_adj = 0.0       # Most of route → no adjustment
    
    # === FACTOR 4: STATION QUALITY (reduced impact) ===
    powers = [s.get('charging_power', 10.0) for s in all_stations.values()]
    waits = [s.get('waiting_time', 0.0) for s in all_stations.values()]
    max_power = max(powers) if powers else 10.0
    max_wait = max(waits) if waits else 1.0
    
    power_score = station.get('charging_power', 10.0) / max_power if max_power > 0 else 0.5
    wait_score = 1 - (station.get('waiting_time', 0.0) / max_wait) if max_wait > 0 else 0.5
    quality = 0.5 * power_score + 0.5 * wait_score
    
    if quality > 0.85:
        quality_adj = +0.05      # Excellent station → slightly more
    elif quality < 0.25:
        quality_adj = -0.05      # Bad station → slightly less
    else:
        quality_adj = 0.0        # Average station
    
    # === COMBINE ALL FACTORS ===
    portion = base + battery_adj + price_adj + progress_adj + quality_adj
    
    # === CLIP TO VALID RANGE ===
    portion = max(0.20, min(1.00, portion))
    
    # === ROUND TO NEAREST VALID PORTION ===
    valid_portions = [0.20, 0.40, 0.60, 0.80, 1.00]
    portion = min(valid_portions, key=lambda x: abs(x - portion))
    
    return portion


# =============================================================================
# STATION SELECTION METHODS
# =============================================================================

def select_nearest_station(feasible_stations, current_pos, stations_data):
    """Select station closest to current position."""
    best_id = None
    best_dist = float('inf')
    
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
        if price < best_price:
            best_price = price
            best_id = sid
    
    return best_id


def select_earliest_service(feasible_stations, current_pos, current_battery, max_battery, stations_data, portion=1.0):
    """
    Select station with minimum total service time.
    Total time = travel time + waiting time + charging time
    
    This implicitly considers:
    - Distance (travel time)
    - Waiting time
    - Charging power (affects charging time)
    """
    best_id = None
    best_time = float('inf')
    
    for sid in feasible_stations:
        s = stations_data[sid]
        
        # Travel time (assuming speed = 1)
        dist = helper.euclidean_distance(current_pos[0], current_pos[1], s['x'], s['y'])
        
        # Charging time
        battery_at_station = current_battery - dist
        energy_needed = portion * (max_battery - battery_at_station)
        charging_power = s.get('charging_power', 10.0)
        charge_time = energy_needed / charging_power if charging_power > 0 else 0
        
        # Total service time
        total_time = dist + s.get('waiting_time', 0) + charge_time
        
        if total_time < best_time:
            best_time = total_time
            best_id = sid
    
    return best_id


def select_rulebased_station(feasible_stations, current_pos, current_battery, max_battery, stations_data, portion=1.0):
    """
    Select station with minimum weighted normalized score.
    
    Considers all factors with equal weights:
    - Distance (25%)
    - Waiting time (25%)
    - Price (25%)
    - Charging time (25%)
    """
    if not feasible_stations:
        return None
    
    # Weights (equal importance)
    w_dist, w_wait, w_price, w_charge = 0.25, 0.25, 0.25, 0.25
    
    # Collect values
    values = []
    for sid in feasible_stations:
        s = stations_data[sid]
        dist = helper.euclidean_distance(current_pos[0], current_pos[1], s['x'], s['y'])
        battery_at = current_battery - dist
        energy = portion * (max_battery - battery_at)
        power = s.get('charging_power', 10.0)
        charge_time = energy / power if power > 0 else 0
        
        values.append({
            'id': sid,
            'dist': dist,
            'wait': s.get('waiting_time', 0),
            'price': s.get('price', 1.0),
            'charge': charge_time
        })
    
    # Get max for normalization (avoid division by zero)
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