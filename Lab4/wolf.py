import numpy as np
import math
import random

# --- Step 1: Define the Problem ---
def fitness_function(x):
    # Objective function to maximize
    return x * math.sin(10 * math.pi * x) + 1

# --- Step 2: Initialize Parameters ---
NUM_WOLVES = 20        # population size
MAX_ITER = 50           # number of iterations
X_BOUND = [0, 1]        # search space bounds

# --- Step 3: Initialize Population ---
wolves = np.random.uniform(X_BOUND[0], X_BOUND[1], NUM_WOLVES)

# --- Step 4: Evaluate Fitness Function ---
def evaluate_fitness(wolves):
    return np.array([fitness_function(x) for x in wolves])

# --- Step 5: GWO Algorithm ---
def grey_wolf_optimizer():
    global wolves

    # Initialize alpha, beta, delta wolves
    fitness = evaluate_fitness(wolves)
    alpha_pos = wolves[np.argmax(fitness)]
    alpha_score = np.max(fitness)

    beta_pos = alpha_pos
    beta_score = alpha_score

    delta_pos = alpha_pos
    delta_score = alpha_score

    # Main loop
    for iteration in range(MAX_ITER):
        a = 2 - iteration * (2 / MAX_ITER)  # linearly decreases from 2 to 0

        for i in range(NUM_WOLVES):
            r1, r2 = random.random(), random.random()

            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = abs(C1 * alpha_pos - wolves[i])
            X1 = alpha_pos - A1 * D_alpha

            r1, r2 = random.random(), random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = abs(C2 * beta_pos - wolves[i])
            X2 = beta_pos - A2 * D_beta

            r1, r2 = random.random(), random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = abs(C3 * delta_pos - wolves[i])
            X3 = delta_pos - A3 * D_delta

            # Update wolf position (average influence of alpha, beta, delta)
            new_position = (X1 + X2 + X3) / 3

            # Keep within bounds
            new_position = np.clip(new_position, X_BOUND[0], X_BOUND[1])
            wolves[i] = new_position

        # Evaluate new fitness
        fitness = evaluate_fitness(wolves)

        # Update alpha, beta, delta
        sorted_indices = np.argsort(-fitness)  # descending order (maximization)
        alpha_pos, beta_pos, delta_pos = wolves[sorted_indices[:3]]
        alpha_score, beta_score, delta_score = fitness[sorted_indices[:3]]

        print(f"Iteration {iteration+1}: Best Fitness = {alpha_score:.5f}")

    # --- Step 7: Output the Best Solution ---
    print("\n=== Final Result ===")
    print(f"Best position (x): {alpha_pos:.5f}")
    print(f"Best fitness: {alpha_score:.5f}")

# --- Run GWO ---
if __name__ == "__main__":
    grey_wolf_optimizer()
