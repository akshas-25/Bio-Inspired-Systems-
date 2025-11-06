import random
import math

# --- Step 1: Define the problem ---
# Function to optimize (maximize)
def fitness_function(x):
    return x * math.sin(10 * math.pi * x) + 1

# --- Step 2: Initialize parameters ---
POP_SIZE = 20         # number of individuals in population
GENS = 50             # number of generations
CROSS_RATE = 0.7      # probability of crossover
MUT_RATE = 0.1        # probability of mutation
X_BOUND = [0, 1]      # search space bounds

# --- Step 3: Create Initial Population ---
def create_population():
    return [random.uniform(*X_BOUND) for _ in range(POP_SIZE)]

# --- Step 4: Evaluate Fitness ---
def evaluate_population(population):
    return [fitness_function(x) for x in population]

# --- Step 5: Selection (Roulette Wheel Selection) ---
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    probs = [f / total_fitness for f in fitnesses]
    return random.choices(population, weights=probs, k=POP_SIZE)

# --- Step 6: Crossover (Single Point) ---
def crossover(parent1, parent2):
    if random.random() < CROSS_RATE:
        alpha = random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child
    else:
        return parent1

# --- Step 7: Mutation ---
def mutate(x):
    if random.random() < MUT_RATE:
        x += random.uniform(-0.1, 0.1)
        x = min(max(x, X_BOUND[0]), X_BOUND[1])  # keep within bounds
    return x

# --- Step 8: Run the Genetic Algorithm ---
def genetic_algorithm():
    population = create_population()
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(GENS):
        fitnesses = evaluate_population(population)

        # Track best individual
        for i in range(POP_SIZE):
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_solution = population[i]

        # Selection
        selected = select(population, fitnesses)

        # Create new generation
        children = []
        for i in range(0, POP_SIZE, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % POP_SIZE]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            children.extend([mutate(child1), mutate(child2)])

        population = children

        print(f"Generation {generation+1}: Best Fitness = {best_fitness:.5f}")

    # --- Step 9: Output the Best Solution ---
    print("\n=== Final Result ===")
    print(f"Best solution x = {best_solution:.5f}")
    print(f"Best fitness = {best_fitness:.5f}")

# --- Run the GA ---
if __name__ == "__main__":
    genetic_algorithm()
