import math
import random

# --- Step 1: Define the Problem (Cities and Distances) ---
cities = {
    0: (0, 0),
    1: (1, 5),
    2: (5, 2),
    3: (6, 6),
    4: (8, 3)
}

NUM_CITIES = len(cities)

# Distance between two cities
def distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Create a distance matrix for fast lookup
dist_matrix = [[distance(i, j) for j in range(NUM_CITIES)] for i in range(NUM_CITIES)]

# --- Step 2: Initialize Parameters ---
NUM_ANTS = 10
ALPHA = 1.0          # importance of pheromone
BETA = 5.0           # importance of heuristic (1/distance)
RHO = 0.5            # pheromone evaporation rate
Q = 100              # pheromone deposit factor
ITERATIONS = 50

# Initialize pheromone matrix (small constant value)
pheromone = [[1.0 for _ in range(NUM_CITIES)] for _ in range(NUM_CITIES)]

# --- Step 3: Construct Solutions ---
def select_next_city(ant_path, current_city):
    unvisited = [city for city in range(NUM_CITIES) if city not in ant_path]
    if not unvisited:
        return None

    # Calculate probability for each unvisited city
    pheromone_values = [pheromone[current_city][j] ** ALPHA for j in unvisited]
    heuristic_values = [(1 / dist_matrix[current_city][j]) ** BETA for j in unvisited]
    combined = [pheromone_values[i] * heuristic_values[i] for i in range(len(unvisited))]

    total = sum(combined)
    probabilities = [c / total for c in combined]

    # Roulette wheel selection
    r = random.random()
    cumulative = 0
    for i, prob in enumerate(probabilities):
        cumulative += prob
        if r <= cumulative:
            return unvisited[i]
    return unvisited[-1]

def construct_solution():
    path = []
    start_city = random.randint(0, NUM_CITIES - 1)
    path.append(start_city)
    current_city = start_city

    while len(path) < NUM_CITIES:
        next_city = select_next_city(path, current_city)
        path.append(next_city)
        current_city = next_city

    path.append(start_city)  # return to start
    return path

# Calculate total distance of a tour
def tour_length(path):
    return sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path) - 1))

# --- Step 4: Update Pheromones ---
def update_pheromones(all_paths):
    global pheromone

    # Evaporate existing pheromone
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            pheromone[i][j] *= (1 - RHO)
            if pheromone[i][j] < 0.0001:  # prevent pheromone vanishing
                pheromone[i][j] = 0.0001

    # Add new pheromone based on ants' solutions
    for path, length in all_paths:
        contribution = Q / length
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            pheromone[a][b] += contribution
            pheromone[b][a] += contribution  # symmetric

# --- Step 5: Iterate the Process ---
def ant_colony_optimization():
    best_path = None
    best_length = float('inf')

    for iteration in range(ITERATIONS):
        all_paths = []
        for _ in range(NUM_ANTS):
            path = construct_solution()
            length = tour_length(path)
            all_paths.append((path, length))

            if length < best_length:
                best_length = length
                best_path = path

        update_pheromones(all_paths)

        print(f"Iteration {iteration+1}: Best Length = {best_length:.4f}")

    # --- Step 6: Output the Best Solution ---
    print("\n=== Final Best Route Found ===")
    print(" -> ".join(map(str, best_path)))
    print(f"Shortest Distance: {best_length:.4f}")

# --- Run the Algorithm ---
if __name__ == "__main__":
    ant_colony_optimization()
