import numpy as np
import math
import random

# --- Step 1: Define the Problem ---
def fitness_function(x):
    # Objective function to maximize
    return x * math.sin(10 * math.pi * x) + 1

# --- Step 2: Initialize Parameters ---
NUM_NESTS = 20        # number of nests
PA = 0.25              # probability of discovery (abandon fraction)
MAX_ITER = 50          # number of iterations
X_BOUND = [0, 1]       # search space

# --- Step 3: Initialize Population ---
def initialize_nests():
    return np.random.uniform(X_BOUND[0], X_BOUND[1], NUM_NESTS)

# --- Step 4: Evaluate Fitness ---
def evaluate_fitness(nests):
    return np.array([fitness_function(x) for x in nests])

# --- Step 5: Generate New Solutions using Lévy flights ---
def levy_flight(Lambda=1.5):
    # Lévy flight step calculation
    sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.randn() * sigma
    v = np.random.randn()
    step = u / abs(v) ** (1 / Lambda)
    return step

def get_cuckoo(nest_best):
    step_size = levy_flight() * (np.random.uniform(-1, 1))
    new_solution = nest_best + step_size * np.random.uniform(-1, 1)
    new_solution = np.clip(new_solution, X_BOUND[0], X_BOUND[1])
    return new_solution

# --- Step 6: Abandon Worst Nests ---
def abandon_nests(nests, fitness):
    # Replace a fraction of worst nests with new random positions
    num_abandon = int(PA * NUM_NESTS)
    worst_indices = np.argsort(fitness)[:num_abandon]
    for idx in worst_indices:
        nests[idx] = np.random.uniform(X_BOUND[0], X_BOUND[1])
    return nests

# --- Step 7: Iterate ---
def cuckoo_search():
    nests = initialize_nests()
    fitness = evaluate_fitness(nests)
    
    best_index = np.argmax(fitness)
    best_nest = nests[best_index]
    best_fitness = fitness[best_index]
    
    for iteration in range(MAX_ITER):
        for i in range(NUM_NESTS):
            new_nest = get_cuckoo(best_nest)
            new_fitness = fitness_function(new_nest)
            
            # If the new solution is better, replace it
            if new_fitness > fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness
        
        # Abandon some nests
        nests = abandon_nests(nests, fitness)
        fitness = evaluate_fitness(nests)
        
        # Update global best
        current_best_index = np.argmax(fitness)
        if fitness[current_best_index] > best_fitness:
            best_fitness = fitness[current_best_index]
            best_nest = nests[current_best_index]
        
        print(f"Iteration {iteration+1}: Best Fitness = {best_fitness:.5f}")
    
    # --- Step 8: Output the Best Solution ---
    print("\n=== Final Result ===")
    print(f"Best solution x = {best_nest:.5f}")
    print(f"Best fitness = {best_fitness:.5f}")

# --- Run the Algorithm ---
if __name__ == "__main__":
    cuckoo_search()
