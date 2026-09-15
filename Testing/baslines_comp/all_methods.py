# =============================================================================
# ALL METHODS: Baselines + Metaheuristics for Charging Station Selection
# All methods use SAME hybrid portion - only optimize station selection
# =============================================================================

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import math
import copy

from helper_functions import HelperFunctions

helper = HelperFunctions()

# Valid charging portions
VALID_PORTIONS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# =============================================================================
# HYBRID PORTION CALCULATION (Used by ALL methods)
# =============================================================================

def calculate_hybrid_portion(current_battery, max_battery, station, stations, route_index, total_customers):
    """Calculate charging portion - SAME for all methods."""
    battery_ratio = current_battery / max_battery
    if battery_ratio < 0.2:
        battery_factor = 1.0
    elif battery_ratio < 0.4:
        battery_factor = 0.9
    elif battery_ratio < 0.6:
        battery_factor = 0.8
    else:
        battery_factor = 0.7
    
    prices = [s.get('price', 1.0) for s in stations.values()]
    min_price, max_price = min(prices), max(prices)
    station_price = station.get('price', 1.0)
    
    if max_price > min_price:
        price_percentile = (station_price - min_price) / (max_price - min_price)
    else:
        price_percentile = 0.5
    
    if price_percentile < 0.3:
        price_factor = 1.0
    elif price_percentile < 0.6:
        price_factor = 0.8
    else:
        price_factor = 0.7
    
    progress = route_index / total_customers if total_customers > 0 else 0.5
    if progress > 0.8:
        progress_factor = 0.7
    else:
        progress_factor = 0.9
    
    portion = 0.4 * battery_factor + 0.4 * price_factor + 0.2 * progress_factor
    portion = max(0.5, min(1.0, portion))
    return min(VALID_PORTIONS, key=lambda x: abs(x - portion))


# =============================================================================
# STATION SCORING (4 factors, equal weights)
# =============================================================================

def score_station(station, current_pos, current_battery, max_battery, all_stations):
    """Score station - lower = better."""
    dist = helper.euclidean_distance(current_pos[0], current_pos[1], station['x'], station['y'])
    wait = station.get('waiting_time', 0)
    price = station.get('price', 1.0)
    
    battery_at = current_battery - dist
    energy = 0.7 * (max_battery - battery_at)
    power = station.get('charging_power', 10.0)
    charge_time = energy / power if power > 0 else 0
    
    max_dist = max(helper.euclidean_distance(current_pos[0], current_pos[1], s['x'], s['y']) 
                   for s in all_stations.values()) or 1
    max_wait = max(s.get('waiting_time', 0) for s in all_stations.values()) or 1
    max_price = max(s.get('price', 1.0) for s in all_stations.values()) or 1
    max_charge = max_dist or 1
    
    return (0.25 * dist / max_dist + 0.25 * wait / max_wait + 
            0.25 * price / max_price + 0.25 * charge_time / max_charge)


def rank_stations_by_score(feasible_ids, stations, current_pos, current_battery, max_battery):
    """Rank stations by score (best first)."""
    scored = [(sid, score_station(stations[sid], current_pos, current_battery, max_battery, stations)) 
              for sid in feasible_ids]
    scored.sort(key=lambda x: x[1])
    return [sid for sid, _ in scored]


# =============================================================================
# ROUTE SIMULATION (Depot to Depot)
# =============================================================================

def simulate_route(env, dm, inst_key, vehicle_id, selector_func):
    """Simulate route from depot to depot."""
    customers = dm.get_customers(inst_key)
    stations = dm.get_stations(inst_key)
    depot = dm.get_depot(inst_key)
    vehicle, _ = dm.get_vehicle_data(inst_key, vehicle_id)
    route = vehicle.get('route', [])
    fleet_specs = dm.get_fleet_specs(vehicle_id)
    max_battery = fleet_specs.get('driving_range_km', 200.0)
    
    # Base travel
    base_travel = 0.0
    for i in range(len(route) - 1):
        n1, n2 = route[i], route[i + 1]
        if n1 in customers:
            pos1 = (customers[n1]['x'], customers[n1]['y'])
        elif n1 == 'D0' and depot:
            pos1 = (depot['x'], depot['y'])
        else:
            continue
        if n2 in customers:
            pos2 = (customers[n2]['x'], customers[n2]['y'])
        elif n2 == 'D0' and depot:
            pos2 = (depot['x'], depot['y'])
        else:
            continue
        base_travel += helper.euclidean_distance(pos1[0], pos1[1], pos2[0], pos2[1])
    
    state = env.reset(inst_key, vehicle_id)
    
    if state is None:
        final_battery = max_battery - base_travel
        if final_battery < 0:
            return {'total_cost': float('inf'), 'feasible': False}
        return {
            'travel_cost': base_travel, 'charging_cost': 0, 'waiting_cost': 0,
            'total_cost': base_travel, 'stops': 0, 'feasible': True,
            'battery_at_depot': final_battery
        }
    
    detour = 0.0
    charge_cost = 0.0
    wait_cost = 0.0
    stops = 0
    done = False
    total_customers = sum(1 for n in route if n in customers)
    
    while not done:
        feasible = state.get('station_ids', [])
        if len(feasible) == 0:
            return {'total_cost': float('inf'), 'feasible': False}
        
        context = {
            'current_pos': env.current_position,
            'current_battery': env.current_battery,
            'max_battery': max_battery,
            'stations': stations,
            'route_index': env.route_index,
            'total_customers': total_customers
        }
        
        station_id, portion = selector_func(feasible, state, context)
        
        if station_id not in feasible:
            station_id = feasible[0]
        if portion not in VALID_PORTIONS:
            portion = 0.8
        
        station = stations[station_id]
        pos = env.current_position
        
        dist = helper.euclidean_distance(pos[0], pos[1], station['x'], station['y'])
        detour += dist
        
        battery_at_station = env.current_battery - dist
        energy = portion * (max_battery - battery_at_station)
        
        charge_cost += energy * station.get('price', 1.0)
        wait_cost += station.get('waiting_time', 0)
        stops += 1
        
        state, _, done = env.step(station_id, portion)
    
    final_pos = env.current_position
    final_battery = env.current_battery
    
    if depot:
        dist_to_depot = helper.euclidean_distance(final_pos[0], final_pos[1], depot['x'], depot['y'])
        battery_at_depot = final_battery - dist_to_depot
    else:
        battery_at_depot = final_battery
    
    if battery_at_depot < 0:
        return {'total_cost': float('inf'), 'feasible': False, 'battery_at_depot': battery_at_depot}
    
    # Battery penalty - penalize high battery at depot (wasted energy)
    battery_penalty = battery_at_depot * 30
    
    return {
        'travel_cost': base_travel + detour,
        'charging_cost': charge_cost,
        'waiting_cost': wait_cost,
        'total_cost': base_travel + detour + charge_cost + wait_cost + battery_penalty,
        'stops': stops,
        'feasible': True,
        'battery_at_depot': battery_at_depot
    }


# =============================================================================
# BASELINE SELECTORS
# =============================================================================

def make_ns_selector():
    """Nearest Station selector."""
    def select(feasible, state, context):
        pos = context['current_pos']
        stations = context['stations']
        best_id = min(feasible, key=lambda sid: 
            helper.euclidean_distance(pos[0], pos[1], stations[sid]['x'], stations[sid]['y']))
        station = stations[best_id]
        portion = calculate_hybrid_portion(
            context['current_battery'], context['max_battery'], station,
            stations, context['route_index'], context['total_customers'])
        return best_id, portion
    return select


def make_cs_selector():
    """Cheapest Station selector."""
    def select(feasible, state, context):
        stations = context['stations']
        best_id = min(feasible, key=lambda sid: stations[sid].get('price', 1.0))
        station = stations[best_id]
        portion = calculate_hybrid_portion(
            context['current_battery'], context['max_battery'], station,
            stations, context['route_index'], context['total_customers'])
        return best_id, portion
    return select


def make_es_selector():
    """Earliest Service selector."""
    def select(feasible, state, context):
        stations = context['stations']
        best_id = min(feasible, key=lambda sid: stations[sid].get('waiting_time', 0))
        station = stations[best_id]
        portion = calculate_hybrid_portion(
            context['current_battery'], context['max_battery'], station,
            stations, context['route_index'], context['total_customers'])
        return best_id, portion
    return select


def make_rb_selector():
    """Rule-Based (4-factor) selector."""
    def select(feasible, state, context):
        stations = context['stations']
        ranked = rank_stations_by_score(feasible, stations, context['current_pos'],
                                        context['current_battery'], context['max_battery'])
        best_id = ranked[0]
        station = stations[best_id]
        portion = calculate_hybrid_portion(
            context['current_battery'], context['max_battery'], station,
            stations, context['route_index'], context['total_customers'])
        return best_id, portion
    return select


def make_rl_selector(agent):
    """RL Agent selector."""
    def select(feasible, state, context):
        idx, p_idx = agent.select_action(state, training=False)
        station_id = feasible[min(idx, len(feasible) - 1)]
        portion = agent.charge_portions[min(p_idx, len(agent.charge_portions) - 1)]
        return station_id, portion
    return select


# =============================================================================
# METAHEURISTIC SELECTOR (Uses SAME hybrid portion as baselines)
# =============================================================================

def make_metaheuristic_selector(station_ranks):
    """
    station_ranks = list of integers (0=best, 1=2nd best, etc.)
    Uses SAME hybrid portion as baselines.
    """
    decision_idx = [0]
    
    def select(feasible, state, context):
        if decision_idx[0] >= len(station_ranks):
            decision_idx[0] = 0
        
        rank = station_ranks[decision_idx[0]]
        decision_idx[0] += 1
        
        ranked = rank_stations_by_score(feasible, context['stations'], context['current_pos'],
                                        context['current_battery'], context['max_battery'])
        
        station_id = ranked[min(rank, len(ranked) - 1)]
        station = context['stations'][station_id]
        
        # SAME hybrid portion as baselines
        portion = calculate_hybrid_portion(
            context['current_battery'], context['max_battery'], station,
            context['stations'], context['route_index'], context['total_customers'])
        
        return station_id, portion
    
    return select


# =============================================================================
# ALNS - Adaptive Large Neighborhood Search
# =============================================================================

def run_alns(env, dm, inst_key, vehicle_id, iterations=50):
    """ALNS - optimizes station ranks only, uses hybrid portion."""
    max_decisions = 10
    
    current = [0] * max_decisions
    
    def evaluate(ranks):
        selector = make_metaheuristic_selector(ranks)
        return simulate_route(env, dm, inst_key, vehicle_id, selector)
    
    current_result = evaluate(current)
    current_cost = current_result['total_cost'] if current_result['feasible'] else float('inf')
    
    best = current[:]
    best_result = current_result
    best_cost = current_cost
    
    weights = {'random': 1.0, 'worst': 1.0}
    
    for _ in range(iterations):
        neighbor = current[:]
        
        if random.random() < weights['random'] / (weights['random'] + weights['worst']):
            idx = random.randint(0, max_decisions - 1)
            op = 'random'
        else:
            idx = max(range(len(neighbor)), key=lambda i: neighbor[i])
            op = 'worst'
        
        neighbor[idx] = random.randint(0, 4)
        
        result = evaluate(neighbor)
        cost = result['total_cost'] if result['feasible'] else float('inf')
        
        if cost < current_cost:
            current = neighbor
            current_cost = cost
            current_result = result
            weights[op] = min(weights[op] * 1.1, 3.0)
            
            if cost < best_cost:
                best = neighbor[:]
                best_cost = cost
                best_result = result
    
    return best_result if best_result['feasible'] else {'total_cost': float('inf'), 'feasible': False}


# =============================================================================
# VNS - Variable Neighborhood Search
# =============================================================================

def run_vns(env, dm, inst_key, vehicle_id, max_iterations=50):
    """VNS - optimizes station ranks only, uses hybrid portion."""
    max_decisions = 10
    current = [0] * max_decisions
    
    def evaluate(ranks):
        selector = make_metaheuristic_selector(ranks)
        return simulate_route(env, dm, inst_key, vehicle_id, selector)
    
    current_result = evaluate(current)
    current_cost = current_result['total_cost'] if current_result['feasible'] else float('inf')
    
    best = current[:]
    best_result = current_result
    best_cost = current_cost
    
    k = 1
    iteration = 0
    
    while iteration < max_iterations and k <= 3:
        neighbor = current[:]
        
        if k == 1:
            idx = random.randint(0, max_decisions - 1)
            neighbor[idx] = random.randint(0, 4)
        elif k == 2:
            for _ in range(2):
                idx = random.randint(0, max_decisions - 1)
                neighbor[idx] = random.randint(0, 4)
        else:
            i, j = random.sample(range(max_decisions), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        result = evaluate(neighbor)
        cost = result['total_cost'] if result['feasible'] else float('inf')
        
        if cost < current_cost:
            current = neighbor
            current_cost = cost
            current_result = result
            k = 1
            
            if cost < best_cost:
                best = neighbor[:]
                best_cost = cost
                best_result = result
        else:
            k += 1
        
        iteration += 1
    
    return best_result if best_result['feasible'] else {'total_cost': float('inf'), 'feasible': False}


# =============================================================================
# ILS - Iterated Local Search
# =============================================================================

def run_ils(env, dm, inst_key, vehicle_id, iterations=50):
    """ILS - optimizes station ranks only, uses hybrid portion."""
    max_decisions = 10
    
    def evaluate(ranks):
        selector = make_metaheuristic_selector(ranks)
        return simulate_route(env, dm, inst_key, vehicle_id, selector)
    
    def local_search(solution, max_steps=5):
        current = solution[:]
        result = evaluate(current)
        cost = result['total_cost'] if result['feasible'] else float('inf')
        
        for _ in range(max_steps):
            improved = False
            idx = random.randint(0, max_decisions - 1)
            
            for new_rank in range(5):
                if new_rank == current[idx]:
                    continue
                neighbor = current[:]
                neighbor[idx] = new_rank
                
                r = evaluate(neighbor)
                c = r['total_cost'] if r['feasible'] else float('inf')
                
                if c < cost:
                    current = neighbor
                    cost = c
                    result = r
                    improved = True
                    break
            
            if not improved:
                break
        
        return current, result, cost
    
    current = [0] * max_decisions
    current, current_result, current_cost = local_search(current)
    
    best = current[:]
    best_result = current_result
    best_cost = current_cost
    
    for _ in range(iterations):
        perturbed = current[:]
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, max_decisions - 1)
            perturbed[idx] = random.randint(0, 4)
        
        improved, result, cost = local_search(perturbed)
        
        if cost < current_cost:
            current = improved
            current_cost = cost
            current_result = result
            
            if cost < best_cost:
                best = improved[:]
                best_cost = cost
                best_result = result
    
    return best_result if best_result['feasible'] else {'total_cost': float('inf'), 'feasible': False}


# =============================================================================
# GA - Genetic Algorithm
# =============================================================================

def run_ga(env, dm, inst_key, vehicle_id, population_size=50, generations=10):
    """GA - optimizes station ranks only, uses hybrid portion."""
    max_decisions = 10
    
    def evaluate(ranks):
        selector = make_metaheuristic_selector(ranks)
        return simulate_route(env, dm, inst_key, vehicle_id, selector)
    
    # Initialize population
    population = [[random.randint(0, 4) for _ in range(max_decisions)] for _ in range(population_size)]
    
    best = None
    best_result = None
    best_cost = float('inf')
    
    for gen in range(generations):
        # Evaluate all
        results = []
        for individual in population:
            result = evaluate(individual)
            cost = result['total_cost'] if result['feasible'] else float('inf')
            results.append((individual, result, cost))
            
            if cost < best_cost:
                best = individual[:]
                best_result = result
                best_cost = cost
        
        # Selection (tournament) + elitism
        new_population = [best[:]] if best else [[0] * max_decisions]
        
        while len(new_population) < population_size:
            # Tournament selection
            t1 = random.sample(results, min(3, len(results)))
            p1 = min(t1, key=lambda x: x[2])[0]
            t2 = random.sample(results, min(3, len(results)))
            p2 = min(t2, key=lambda x: x[2])[0]
            
            # Crossover
            point = random.randint(1, max_decisions - 1)
            child = p1[:point] + p2[point:]
            
            # Mutation
            if random.random() < 0.2:
                idx = random.randint(0, max_decisions - 1)
                child[idx] = random.randint(0, 4)
            
            new_population.append(child)
        
        population = new_population
    
    return best_result if best_result and best_result['feasible'] else {'total_cost': float('inf'), 'feasible': False}


# =============================================================================
# SA - Simulated Annealing
# =============================================================================

def run_sa(env, dm, inst_key, vehicle_id, initial_temp=100, cooling_rate=0.95, iterations=100):
    """SA - optimizes station ranks only, uses hybrid portion."""
    max_decisions = 10
    
    def evaluate(ranks):
        selector = make_metaheuristic_selector(ranks)
        return simulate_route(env, dm, inst_key, vehicle_id, selector)
    
    current = [0] * max_decisions
    current_result = evaluate(current)
    current_cost = current_result['total_cost'] if current_result['feasible'] else float('inf')
    
    best = current[:]
    best_result = current_result
    best_cost = current_cost
    
    temp = initial_temp
    
    for _ in range(iterations):
        # Generate neighbor
        neighbor = current[:]
        idx = random.randint(0, max_decisions - 1)
        neighbor[idx] = random.randint(0, 4)
        
        result = evaluate(neighbor)
        cost = result['total_cost'] if result['feasible'] else float('inf')
        
        # Accept or reject
        delta = cost - current_cost
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
            current = neighbor
            current_cost = cost
            current_result = result
            
            if cost < best_cost:
                best = neighbor[:]
                best_cost = cost
                best_result = result
        
        temp *= cooling_rate
    
    return best_result if best_result and best_result['feasible'] else {'total_cost': float('inf'), 'feasible': False}


# =============================================================================
# PSO - Particle Swarm Optimization
# =============================================================================

def run_pso(env, dm, inst_key, vehicle_id, num_particles=30, iterations=50):
    """PSO - optimizes station ranks only, uses hybrid portion."""
    max_decisions = 10
    
    def evaluate(ranks):
        discrete = [max(0, min(4, int(r))) for r in ranks]
        selector = make_metaheuristic_selector(discrete)
        return simulate_route(env, dm, inst_key, vehicle_id, selector)
    
    # Initialize particles (continuous values 0-4)
    particles = [[random.uniform(0, 4) for _ in range(max_decisions)] for _ in range(num_particles)]
    velocities = [[random.uniform(-1, 1) for _ in range(max_decisions)] for _ in range(num_particles)]
    
    # Personal best
    p_best = [p[:] for p in particles]
    p_best_cost = [float('inf')] * num_particles
    
    # Global best
    g_best = None
    g_best_result = None
    g_best_cost = float('inf')
    
    w = 0.7  # Inertia
    c1 = 1.5  # Cognitive
    c2 = 1.5  # Social
    
    for _ in range(iterations):
        for i in range(num_particles):
            result = evaluate(particles[i])
            cost = result['total_cost'] if result['feasible'] else float('inf')
            
            # Update personal best
            if cost < p_best_cost[i]:
                p_best[i] = particles[i][:]
                p_best_cost[i] = cost
            
            # Update global best
            if cost < g_best_cost:
                g_best = particles[i][:]
                g_best_cost = cost
                g_best_result = result
        
        # Update velocities and positions
        for i in range(num_particles):
            for d in range(max_decisions):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (w * velocities[i][d] + 
                                    c1 * r1 * (p_best[i][d] - particles[i][d]) +
                                    c2 * r2 * (g_best[d] - particles[i][d]) if g_best else 0)
                velocities[i][d] = max(-2, min(2, velocities[i][d]))
                particles[i][d] += velocities[i][d]
                particles[i][d] = max(0, min(4, particles[i][d]))
    
    return g_best_result if g_best_result and g_best_result['feasible'] else {'total_cost': float('inf'), 'feasible': False}