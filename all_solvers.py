##==============================================================================
##
## Title:		Travelling Salesman solutions for inspection optimization
## Purpose:		Found optimal inspection path
## Usage:       python solver_all.py <cost_matrix_file.csv>
## Dependency:  pip install numpy pandas gurobipy scipy
## Created on:	12.07.2025 at 08:00:00 by Andrey Dmitriev.
## Python 3.14  (gurobipy is available in Python 3.13 only)
##==============================================================================
import sys
# Get major and minor version
major, minor = sys.version_info[:2]
print(f"Python version: {major}.{minor}")
import time
import numpy as np
import pandas as pd
import random
import math
import itertools
from itertools import permutations
from scipy.optimize import LinearConstraint, Bounds, milp, linprog

if minor <= 13:
    import gurobipy as gp
    from gurobipy import GRB

# Check for command-line argument
if len(sys.argv) != 2:
    print("Usage: python solver_all.py <cost_matrix_file.csv>")
    sys.exit(1)

# Get the file name from the command line
file_name = sys.argv[1]

# Read the CSV into a DataFrame
try:
    df = pd.read_csv(file_name, header=None, delimiter=";")
    distance_matrix = df.to_numpy(dtype=float)
except Exception as e:
    print(f"Error reading file '{file_name}': {e}")
    sys.exit(1)

# Helper function to calculate total distance
def total_distance(route, matrix):
    return sum(matrix[route[i]][route[i+1]] for i in range(len(route) - 1))


##==============================================================================
## Greedy Heuristic Inspection optimizer
##
def solve_tsp_greedy(matrix):
    n = len(matrix)  # Number of cities
    inspected = [False] * n  # Track which cities have been visited
    path = [0]  # Start the inspection from home position  0
    inspected[0] = True  # Mark the starting position as visited
    total_cost = 0  # Initialize total inspection cost

    # Loop to build the inspection by visiting the nearest uninspected
    for _ in range(n - 1):
        last = path[-1]  # Get the last inspected position
        # Find the nearest uninspected position
        next_pos = np.argmin([
            matrix[last][j] if not inspected[j] else np.inf
            for j in range(n)
        ])
        path.append(next_pos)  # Add the next position to the inspection
        inspected[next_pos] = True  # Mark it as inspected
        total_cost += matrix[last][next_pos]  # Add inspection cost

    # Return to the starting home position to complete the inspection
    total_cost += matrix[path[-1]][path[0]]
    path.append(path[0])  # Add the starting position to the end of the inspection

    # Convert inspection to integers and return the result
    route = [int(x) for x in path]
    return route, total_cost


##==============================================================================
## Simulated Annealing solver
## 
def simulated_annealing(distance_matrix, initial_temp=1000, cooling_rate=0.995, stop_temp=1e-8):
    n = len(distance_matrix)  # Number of positions

    # Start with an initial inspection visiting positions in order and returning to the start
    current = list(range(n)) + [0]
    best = current[:]  # Best inspection found so far
    current_cost = total_distance(current, distance_matrix)  # Cost of current inspection
    best_cost = current_cost  # Initialize best cost
    temp = initial_temp  # Set initial temperature

    # Main loop: continue until the system cools below the stopping temperature
    while temp > stop_temp:
        # Randomly select two indices to define a segment to reverse (2-opt move)
        i, j = sorted(random.sample(range(1, n), 2))

        # Create a new inspection by reversing the segment between i and j
        new = current[:i] + current[i:j][::-1] + current[j:]
        new_cost = total_distance(new, distance_matrix)  # Cost of the new inspection
        # Accept the new inspection if it's better, or with a probability based on temperature
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temp):
            current = new  # Update current inspection
            current_cost = new_cost  # Update current cost

            # Update best inspection if the new one is better
            if new_cost < best_cost:
                best = new
                best_cost = new_cost

        # Cool down the system
        temp *= cooling_rate

    # Return the best inspection and its cost
    return best, best_cost


##==============================================================================
## Ant Colony Optimization
##
def ant_colony_optimization(matrix, n_ants=20, n_best=5, n_iterations=100, decay=0.1, alpha=1, beta=2):
    n = len(matrix)  # Number of positions
    pheromone = np.ones((n, n))  # Initialize pheromone levels
    all_inds = range(n)  # Index list for positions

    # Calculate the total distance of an inspection
    def inspection_distance(inspection):
        return sum(matrix[inspection[i]][inspection[i + 1]] for i in range(len(inspection) - 1))

    # Generate a single inspection path starting from a given position
    def gen_inspection(start):
        inspection = [start]
        visited = set(inspection)
        for _ in range(n - 1):
            i = inspection[-1]
            probs = []
            for j in all_inds:
                if j not in visited:
                    tau = pheromone[i][j] ** alpha  # Influence of pheromone
                    eta = (1 / matrix[i][j]) ** beta  # Influence of distance (heuristic)
                    probs.append((j, tau * eta))
            total = sum(p for _, p in probs)
            probs = [(j, p / total) for j, p in probs]  # Normalize probabilities
            r = random.random()
            cumulative = 0
            for j, p in probs:
                cumulative += p
                if r <= cumulative:
                    inspection.append(j)
                    visited.add(j)
                    break
        inspection.append(start)  # Return to starting position
        return inspection

    best_inspection = None
    best_distance = float("inf")

    # Main optimization loop
    for _ in range(n_iterations):
        # Generate inspections for all ants
        all_inspections = [gen_inspection(0) for _ in range(n_ants)]
        # Sort inspections by total distance
        all_inspections.sort(key=lambda r: inspection_distance(r))
        # Update pheromones based on the best inspections
        for inspection in all_inspections[:n_best]:
            for i in range(n):
                pheromone[inspection[i]][inspection[i + 1]] += 1.0 / inspection_distance(inspection)
        pheromone *= (1 - decay)  # Apply pheromone evaporation
        # Update best inspection if a better one is found
        if inspection_distance(all_inspections[0]) < best_distance:
            best_distance = inspection_distance(all_inspections[0])
            best_inspection = all_inspections[0]

    return best_inspection, best_distance


##==============================================================================
## Tabu Search
##
def tabu_search(matrix, iterations=500, tabu_size=50):
    n = len(matrix)  # Number of positions

    # Start with an initial inspection visiting positions in order and returning to the start
    current = list(range(n)) + [0]
    best = current[:]  # Best inspection found so far
    best_cost = total_distance(best, matrix)  # Cost of the best inspection
    tabu_list = []  # List to store recently visited inspections (tabu)

    # Main optimization loop
    for _ in range(iterations):
        neighbors = []

        # Generate neighbors by swapping two positions in the current inspection
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                neighbor = current[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap two positions
                neighbor[-1] = neighbor[0]  # Ensure the inspection returns to the start

                # Only consider neighbors not in the tabu list
                if neighbor not in tabu_list:
                    neighbors.append((neighbor, total_distance(neighbor, matrix)))

        # Sort neighbors by their inspection cost (ascending)
        neighbors.sort(key=lambda x: x[1])

        # Choose the best neighbor as the new current inspection
        current = neighbors[0][0]

        # Update the best inspection if the new one is better
        if neighbors[0][1] < best_cost:
            best = current
            best_cost = neighbors[0][1]

        # Add the current inspection to the tabu list
        tabu_list.append(current)

        # Maintain the tabu list size
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    # Return the best inspection and its cost
    return best, best_cost


##==============================================================================
## Lin-Kernighan Heuristic (simplified 2-opt)
##
def lin_kernighan(matrix):
    # Local optimization using 2-opt swaps
    def two_opt(inspection):
        best = inspection  # Best inspection found so far
        improved = True

        # Repeat until no further improvement is found
        while improved:
            improved = False
            # Try all pairs of non-adjacent positions to reverse
            for i in range(1, len(inspection) - 2):
                for j in range(i + 1, len(inspection) - 1):
                    if j - i == 1:
                        continue  # Skip adjacent swaps (no effect)
                    # Create a new inspection by reversing the segment between i and j
                    new_inspection = best[:i] + best[i:j][::-1] + best[j:]
                    # Accept the new inspection if it's better
                    if total_distance(new_inspection, matrix) < total_distance(best, matrix):
                        best = new_inspection
                        improved = True
            inspection = best  # Update current inspection

        return best  # Return the best inspection found

    # Start with a simple inspection visiting positions in order and returning to the start
    initial = list(range(len(matrix))) + [0]
    return two_opt(initial)  # Apply 2-opt optimization


##==============================================================================
## Genetic Algorithm solver
##
def genetic_algorithm(distance_matrix, population_size=100, generations=500, mutation_rate=0.01):
    import random

    # Create a random inspection (a sequence of positions starting and ending at position 0)
    def create_inspection():
        inspection = list(range(1, len(distance_matrix)))  # Exclude starting position
        random.shuffle(inspection)  # Shuffle the middle positions
        return [0] + inspection + [0]  # Start and end at position 0

    # Crossover two parent inspections to produce a child
    def crossover(parent1, parent2):
        start, end = sorted(random.sample(range(1, len(parent1) - 1), 2))
        child = [-1] * len(parent1)

        # Copy a slice from parent1
        child[start:end] = parent1[start:end]

        # Fill remaining positions from parent2 in order, skipping duplicates
        p2_index = 1
        for i in range(1, len(child) - 1):
            if child[i] == -1:
                while parent2[p2_index] in child:
                    p2_index += 1
                child[i] = parent2[p2_index]

        # Ensure the inspection starts and ends at position 0
        child[0] = child[-1] = 0
        return child

    # Mutate an inspection by swapping two positions (with a small probability)
    def mutate(inspection):
        if random.random() < mutation_rate:
            i, j = sorted(random.sample(range(1, len(inspection) - 1), 2))
            inspection[i], inspection[j] = inspection[j], inspection[i]

    # Initialize the population with random inspections
    population = [create_inspection() for _ in range(population_size)]

    # Evolve the population over multiple generations
    for _ in range(generations):
        # Sort inspections by total distance (fitness)
        population.sort(key=lambda r: total_distance(r, distance_matrix))

        # Select the top 10 inspections to carry over to the next generation
        next_gen = population[:10]

        # Generate new inspections through crossover and mutation
        while len(next_gen) < population_size:
            parents = random.sample(population[:50], 2)  # Select parents from top 50
            child = crossover(parents[0], parents[1])
            mutate(child)
            next_gen.append(child)

        population = next_gen  # Update population

    # Return the best inspection and its total distance
    best = min(population, key=lambda r: total_distance(r, distance_matrix))
    return best, total_distance(best, distance_matrix)


##==============================================================================
## This is a Gurobi-based MILP (Mixed-Integer Linear Programming) solver 
## using the Miller Tucker Zemlin (MTZ) formulation to eliminate subtours.
##
if minor <= 13:
    def solve_tsp_gurobi(matrix):
        n = len(matrix)  # Number of positions

        # Create Gurobi model
        model = gp.Model()
        model.setParam('OutputFlag', 0)  # Suppress solver output

        # Decision variables: x[i, j] = 1 if inspection goes from position i to j
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        
        # MTZ variables for subtour elimination
        u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
        
        # Objective: minimize total inspection cost
        model.setObjective(
            gp.quicksum(matrix[i][j] * x[i, j] for i in range(n) for j in range(n)),
            GRB.MINIMIZE
        )
        
        # Constraints: each position has exactly one outgoing and one incoming inspection
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)
            model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)
        
        # Subtour elimination constraints (Miller-Tucker-Zemlin formulation)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)
        
        # Solve the model
        model.optimize()
        
        # Extract the inspection path from the solution
        if model.status == GRB.OPTIMAL:
            inspection = []
            current = 0
            inspected = set([current])  # Track inspected positions
            inspection.append(current)
        
            # Reconstruct the inspection path from decision variables
            while len(inspection) < n:
                for j in range(n):
                    if x[current, j].X > 0.5 and j not in inspected:
                        inspection.append(j)
                        inspected.add(j)
                        current = j
                        break
        
            inspection.append(inspection[0])  # Return to starting position
        
            # Calculate total inspection cost
            total_cost = sum(matrix[inspection[i]][inspection[i + 1]] for i in range(n))
            return inspection, total_cost
        else:
            raise Exception("No optimal solution found.")
        
        

##==============================================================================
## Mixed-Integer Linear Programming (MILP) - milp() from scipy.optimize
##
def solve_tsp_milp(matrix):
    n = matrix.shape[0]
    num_vars = n * n + (n - 1)

    # Objective function
    c = np.concatenate([matrix.flatten(), np.zeros(n - 1)])

    # Bounds
    bounds = []
    for i in range(n):
        for j in range(n):
            bounds.append((0, 0) if i == j else (0, 1))
    for _ in range(n - 1):
        bounds.append((1, n - 1))

    # Equality constraints: one incoming and one outgoing edge per node
    A_eq = []
    b_eq = []

    for i in range(n):  # Outgoing
        row = [0] * num_vars
        for j in range(n):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(1)

    for j in range(n):  # Incoming
        row = [0] * num_vars
        for i in range(n):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(1)

    # Inequality constraints: MTZ subtour elimination
    A_ub = []
    b_ub = []

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                row = [0] * num_vars
                row[i * n + j] = n
                row[n * n + i - 1] = 1
                row[n * n + j - 1] = -1
                A_ub.append(row)
                b_ub.append(n - 1)

    # Convert to numpy arrays
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Integrality: binary for x[i][j], continuous for u[i]
    integrality = np.array([1] * (n * n) + [0] * (n - 1))

    # Solve MILP
    res = milp(c=c,
               constraints=[LinearConstraint(A_eq, b_eq, b_eq),
                            LinearConstraint(A_ub, -np.inf, b_ub)],
               integrality=integrality,
               bounds=Bounds(*zip(*bounds)),
               options={"disp": False})

    if res.success:
        x = res.x[:n * n].reshape((n, n))

        # Extract tour
        tour = []
        current = 0
        visited = set()
        while len(tour) < n:
            tour.append(current)
            visited.add(current)
            for j in range(n):
                if x[current, j] > 0.5 and j not in visited:
                    current = j
                    break
            else:
                break
        tour.append(tour[0])  # return to start

        total_cost = sum(matrix[tour[i]][tour[i+1]] for i in range(n))
        return tour, total_cost
    else:
        raise Exception(f"SciPy MILP solver failed: {res.message}")


##==============================================================================
## Linear Programming (Simplex method) - linprog() from scipy.optimize  
##
def solve_tsp_simplex(matrix):
    n = len(matrix)  # Number of positions
    c = matrix.flatten()  # Flatten the cost matrix for the objective function

    # Constraints: each position must be departed from and arrived at exactly once
    A_eq = []
    b_eq = []

    # Row constraints: each position must be departed from exactly once
    for i in range(n):
        row = [0] * (n * n)
        for j in range(n):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(1)

    # Column constraints: each position must be arrived at exactly once
    for j in range(n):
        col = [0] * (n * n)
        for i in range(n):
            col[i * n + j] = 1
        A_eq.append(col)
        b_eq.append(1)

    # Bounds: all decision variables between 0 and 1
    bounds = [(0, 1) for _ in range(n * n)]

    # Prevent self-loops (no inspection from a position to itself)
    for i in range(n):
        bounds[i * n + i] = (0, 0)

    # Solve the LP relaxation using the Simplex method
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        x = res.x.reshape((n, n))  # Reshape solution to matrix form
        inspected = [False] * n  # Track which positions have been inspected
        inspection = [0]  # Start inspection from position 0
        inspected[0] = True
        current = 0

        # Construct the inspection path using the LP solution
        for _ in range(n - 1):
            next_pos = np.argmax(x[current])
            if inspected[next_pos]:
                # Fallback: pick the next uninspected position with highest value
                candidates = [(j, x[current][j]) for j in range(n) if not inspected[j]]
                if not candidates:
                    break
                next_pos = max(candidates, key=lambda item: item[1])[0]
            inspection.append(next_pos)
            inspected[next_pos] = True
            current = next_pos

        inspection.append(inspection[0])  # Return to starting position

        # Calculate total inspection cost
        total_cost = sum(matrix[inspection[i]][inspection[i + 1]] for i in range(n))
        path = [int(x) for x in inspection]
        return path, round(total_cost, 3)
    else:
        return [], float('inf')


##==============================================================================
## Brute Force solver (will take few minutes for 12 positions)
##
def solve_brute_force(distance_matrix):
    n = len(distance_matrix)
    cities = list(range(1, n))
    min_cost = float('inf')
    optimal_route = []

    for perm in itertools.permutations(cities):
        route = [0] + list(perm) + [0]  # start and end at city 0
        cost = sum(distance_matrix[route[i]][route[i+1]] for i in range(n))
        if cost < min_cost:
            min_cost = cost
            optimal_route = route

    return optimal_route, min_cost


##==============================================================================
## Held Karp Brute Force solver (can take few minutes for 20 positions)
##
def solve_held_karp(distance_matrix):
    n = len(distance_matrix)
    dp = {}

    # Initialize base cases
    for k in range(1, n):
        dp[(1 << k, k)] = distance_matrix[0][k]

    # Build up solutions for subsets
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            for k in subset:
                prev_bits = bits & ~(1 << k)
                min_cost = float('inf')
                for m in subset:
                    if m == k:
                        continue
                    cost = dp.get((prev_bits, m), float('inf')) + distance_matrix[m][k]
                    if cost < min_cost:
                        min_cost = cost
                dp[(bits, k)] = min_cost

    # Final step: return to start
    bits = (1 << n) - 2
    min_cost = float('inf')
    last_node = None
    for k in range(1, n):
        cost = dp.get((bits, k), float('inf')) + distance_matrix[k][0]
        if cost < min_cost:
            min_cost = cost
            last_node = k

    # If no path found, return None
    if last_node is None:
        return [], float('inf')

    # Reconstruct path
    path = [0]
    current = last_node
    bits = (1 << n) - 2
    for _ in range(n - 1):
        path.append(current)
        prev_bits = bits & ~(1 << current)
        next_node = None
        for m in range(1, n):
            if prev_bits & (1 << m):
                if dp.get((prev_bits, m), float('inf')) + distance_matrix[m][current] == dp[(bits, current)]:
                    next_node = m
                    break
        current = next_node
        bits = prev_bits
    path.append(0)
    path.reverse()

    return path, min_cost




##==============================================================================
## Run all algorithms, solve the TSP and print results
##==============================================================================
print("\nSequential Inspection:")
start = time.time()
seq_route = list(range(len(distance_matrix)))  # [0, 1, 2, ...]
seq_route.append(0)          # [0, 1, 2, ..., 0]
seq_cost = total_distance(seq_route, distance_matrix)
end = time.time()
print("Path:", seq_route)
print("Cost:", round(seq_cost, 3))
print("Time:", round(end - start, 4), "seconds")

print("\nGreedy Algorithm:")
start = time.time()
greedy_path, greedy_cost = solve_tsp_greedy(distance_matrix)
end = time.time()
print("Path:", greedy_path)
print("Cost:", round(greedy_cost, 3))
print("Time:", round(end - start, 4), "seconds")

print("\nSimulated Annealing:")
start = time.time()
simu_path, simu_cost = simulated_annealing(distance_matrix)
end = time.time()
print("Path:", simu_path)
print("Cost:", round(simu_cost, 3))
print("Time:", round(end - start, 4), "seconds")

print("\nAnt Colony Optimization:")
start = time.time()
aco_path, aco_cost = ant_colony_optimization(distance_matrix)
end = time.time()
print("Path:", aco_path)
print("Cost:", round(aco_cost, 3))
print("Time:", round(end - start, 4), "seconds")

print("\nTabu Search:")
start = time.time()
tabu_path, tabu_cost = tabu_search(distance_matrix, iterations=5, tabu_size=3)
end = time.time()
print("Path:", tabu_path)
print("Cost:", round(tabu_cost, 3))
print("Time:", round(end - start, 4), "seconds")

print("\nLin-Kernighan (2-Opt) Heuristic:")
start = time.time()
lk_path = lin_kernighan(distance_matrix)
lk_cost = total_distance(lk_path, distance_matrix)
end = time.time()
print("Path:", lk_path)
print("Cost:", round(lk_cost, 3))
print("Time:", round(end - start, 4), "seconds")

print("\nGenetic Algorithm:")
start = time.time()
gen_path, gen_cost = genetic_algorithm(distance_matrix)
end = time.time()
print("Path:", gen_path)
print("Cost:", round(gen_cost, 3))
print("Time:", round(end - start, 4), "seconds")

print("\nMixed-Integer Linear Programming (MILP) Algorithm:")
start = time.time()
milp_route, milp_cost = solve_tsp_milp(distance_matrix)
end = time.time()
print("Path:", milp_route)
print("Cost:", round(milp_cost, 3))
print("Time:", round(end - start, 4), "seconds")

print("\nSimplex Method:")
start = time.time()
simpl_route, simpl_cost = solve_tsp_simplex(distance_matrix)
end = time.time()
print("Path:", simpl_route)
print("Cost:", round(simpl_cost, 3))
print("Time:", round(end - start, 4), "seconds")

if len(distance_matrix) <= 12:
    print("\nBrute Force Solver:")
    start = time.time()
    brute_path, brute_cost = solve_brute_force(distance_matrix)
    end = time.time()
    print("Path:", brute_path)
    print("Cost:", round(brute_cost, 3))
    print("Time:", round(end - start, 4), "seconds")
else:
    print("Brute Force Solver skipped: distance_matrix exceeds 12 elements.")

if len(distance_matrix) <= 20:
    print("\nHeld Karp Brute Force Solver:")
    start = time.time()
    held_karp_path, held_karp_cost = solve_held_karp(distance_matrix)
    end = time.time()
    print("Path:", held_karp_path)
    print("Cost:", round(held_karp_cost, 3))
    print("Time:", round(end - start, 4), "seconds")
else:
    print("Held Karp Solver skipped: distance_matrix exceeds 20 elements.")


if minor <= 13: # Python 3.13 and below only
    if len(distance_matrix) <= 44:
        print("\nGurobi Solver:")
        start = time.time()
        gur_path, gur_cost = solve_tsp_gurobi(distance_matrix)
        end = time.time()
        print("Path:", gur_path)
        print("Cost:", round(gur_cost, 3))
        print("Time:", round(end - start, 4), "seconds")
    else:
        print("\nGurobi Solver skipped: distance_matrix exceeds 44 elements.")
