import numpy as np
import random

# Step 1: Define the problem (maximize this function)
def objective_function(x):
    # Example: f(x) = x * sin(10πx) + 1
    return x * np.sin(10 * np.pi * x) + 1

# Step 2: Initialize Parameters
POP_SIZE = 50
NUM_GENES = 10
GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
LOWER_BOUND, UPPER_BOUND = -1, 2

# Step 3: Initialize Population
def init_population():
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (POP_SIZE, NUM_GENES))

# Step 4: Evaluate Fitness (Gene Expression)
def evaluate_fitness(pop):
    expressed_values = np.mean(pop, axis=1)  # Gene expression: average
    fitness = objective_function(expressed_values)
    return fitness, expressed_values

# ✅ Fixed Step 5: Selection (non-negative probabilities)
def selection(pop, fitness):
    fitness_shifted = fitness - np.min(fitness) + 1e-10
    probs = fitness_shifted / np.sum(fitness_shifted)
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, p=probs)
    return pop[idx]

# Step 6: Crossover
def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, NUM_GENES)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()

# Step 7: Mutation
def mutate(chromosome):
    for i in range(NUM_GENES):
        if np.random.rand() < MUTATION_RATE:
            chromosome[i] += np.random.uniform(-0.2, 0.2)
            chromosome[i] = np.clip(chromosome[i], LOWER_BOUND, UPPER_BOUND)
    return chromosome

# Step 8–10: Main Evolution Loop
population = init_population()
best_solution, best_fitness = None, -np.inf

for generation in range(GENERATIONS):
    fitness, expressed = evaluate_fitness(population)
    
    # Track best
    gen_best_idx = np.argmax(fitness)
    if fitness[gen_best_idx] > best_fitness:
        best_fitness = fitness[gen_best_idx]
        best_solution = expressed[gen_best_idx]
    
    selected = selection(population, fitness)
    
    new_population = []
    for i in range(0, POP_SIZE, 2):
        parent1, parent2 = selected[i], selected[(i+1) % POP_SIZE]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))
    
    population = np.array(new_population)
    
    if generation % 10 == 0:
        print(f"Generation {generation} → Best Fitness: {best_fitness:.5f}")

print("\n=== Final Result ===")
print(f"Best Solution (x): {best_solution}")
print(f"Best Fitness: {best_fitness:.5f}")
