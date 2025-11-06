import random
import math

# --- Step 1: Define the problem (objective function) ---
def fitness_function(x):
    return x * math.sin(10 * math.pi * x) + 1

# --- Step 2: Initialize PSO parameters ---
NUM_PARTICLES = 30         # swarm size
MAX_ITER = 50              # number of iterations
W = 0.7                    # inertia weight (controls exploration/exploitation)
C1 = 1.5                   # cognitive coefficient (particle’s own best)
C2 = 1.5                   # social coefficient (global best)
X_BOUND = [0, 1]           # position limits
V_MAX = 0.1                # max velocity

# --- Step 3: Initialize Particles ---
class Particle:
    def __init__(self):
        self.position = random.uniform(*X_BOUND)
        self.velocity = random.uniform(-V_MAX, V_MAX)
        self.best_position = self.position
        self.best_fitness = fitness_function(self.position)

# --- Step 4–6: PSO Algorithm ---
def particle_swarm_optimization():
    # Initialize the swarm
    swarm = [Particle() for _ in range(NUM_PARTICLES)]
    
    # Initialize global best
    global_best_position = max(swarm, key=lambda p: p.best_fitness).best_position
    global_best_fitness = fitness_function(global_best_position)
    
    # Iterations
    for iteration in range(MAX_ITER):
        for particle in swarm:
            # Evaluate current fitness
            fitness = fitness_function(particle.position)
            
            # Update personal best
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position
                
            # Update global best
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position
        
        # Update velocities and positions
        for particle in swarm:
            r1, r2 = random.random(), random.random()
            
            # Velocity update (PSO equation)
            particle.velocity = (
                W * particle.velocity
                + C1 * r1 * (particle.best_position - particle.position)
                + C2 * r2 * (global_best_position - particle.position)
            )
            
            # Limit velocity
            if particle.velocity > V_MAX:
                particle.velocity = V_MAX
            elif particle.velocity < -V_MAX:
                particle.velocity = -V_MAX
            
            # Position update
            particle.position += particle.velocity
            
            # Keep position within bounds
            if particle.position < X_BOUND[0]:
                particle.position = X_BOUND[0]
            elif particle.position > X_BOUND[1]:
                particle.position = X_BOUND[1]
        
        print(f"Iteration {iteration+1}: Global Best Fitness = {global_best_fitness:.5f}")
    
    # --- Step 7: Output the best solution ---
    print("\n=== Final Result ===")
    print(f"Best Position (x): {global_best_position:.5f}")
    print(f"Best Fitness: {global_best_fitness:.5f}")

# --- Run PSO ---
if __name__ == "__main__":
    particle_swarm_optimization()
