import numpy as np

# Step 1: Define the optimization function
def objective_function(x):
    # Example: Sphere function -> minimize sum(x_i^2)
    return np.sum(x ** 2)

# Step 2: Initialize parameters
grid_size = (10, 10)           # 10x10 grid
num_cells = grid_size[0] * grid_size[1]
dimensions = 2                 # number of variables per solution
iterations = 100               # number of iterations
lower_bound, upper_bound = -5.12, 5.12
neighborhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # von Neumann neighborhood

# Step 3: Initialize population (random positions)
population = np.random.uniform(lower_bound, upper_bound, (grid_size[0], grid_size[1], dimensions))
fitness = np.zeros((grid_size[0], grid_size[1]))

# Step 4: Evaluate fitness
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        fitness[i, j] = objective_function(population[i, j])

# Helper: Get neighbors for each cell
def get_neighbors(i, j):
    neighbors = []
    for dx, dy in neighborhood:
        ni, nj = (i + dx) % grid_size[0], (j + dy) % grid_size[1]
        neighbors.append((ni, nj))
    return neighbors

# Step 5: Update rule (local interaction)
def update_cell(i, j):
    current_pos = population[i, j]
    current_fit = fitness[i, j]
    
    # Get best neighbor
    best_neighbor = current_pos
    best_fit = current_fit
    for ni, nj in get_neighbors(i, j):
        if fitness[ni, nj] < best_fit:
            best_neighbor = population[ni, nj]
            best_fit = fitness[ni, nj]
    
    # Move slightly toward the best neighbor (diffusion-like update)
    new_pos = current_pos + np.random.rand() * (best_neighbor - current_pos)
    
    # Ensure boundaries
    new_pos = np.clip(new_pos, lower_bound, upper_bound)
    new_fit = objective_function(new_pos)
    
    # Update if better
    if new_fit < current_fit:
        population[i, j] = new_pos
        fitness[i, j] = new_fit

# Step 6: Iterate and update cells
for t in range(iterations):
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            update_cell(i, j)
    
    # Track the best cell
    best_index = np.unravel_index(np.argmin(fitness), fitness.shape)
    best_value = fitness[best_index]
    
    if t % 10 == 0:
        print(f"Iteration {t} -> Best Fitness: {best_value:.6f}")

# Step 7: Output best solution
best_index = np.unravel_index(np.argmin(fitness), fitness.shape)
best_solution = population[best_index]
print("\n=== Final Result ===")
print("Best Solution:", best_solution)
print("Best Fitness:", objective_function(best_solution))
