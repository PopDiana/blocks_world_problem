from problem import *
from blocks_world import *
from random_state_generator import *
import time

blocks = int(input("Number of blocks : "))

initial_state = state_generator(blocks)  # Generating random states for initial and goal state
goal_state = state_generator(blocks)
print("\n\nInitial :", initial_state)
print("Goal : ", goal_state, "\n\n")

start = time.time()

problem1 = BlocksWorldH1(initial_state, goal_state)
problem2 = BlocksWorldH2(initial_state, goal_state)

start_time_astar = time.time()
astar = astar_search(problem2)  # A* search using second heuristic
end_time_astar = time.time()

start_time_rbfs = time.time()
rbfs = recursive_best_first_search(problem2)  # Recursive best-first search using second heuristic
end_time_rbfs = time.time()

astar_solution = astar.solution()
astar_path = astar.path()

rbfs_solution = rbfs.solution()
rbfs_path = rbfs.path()

end = time.time()


# Print each state and the action that results to it for both searchers

print("Solution for astar search : \n\n")

print("ACTION :                STATE :             \n")

initial = astar_path[0].state
print("Initial state ->        {0}\n".format(initial))

for iterator in range(1, len(astar_path)):
    state = astar_path[iterator].state
    action = astar_solution[iterator - 1]

    if action[1] != ' ':
        print("Move from {0} to {1}        {2}\n".format(action[0], action[1], state))
    else:
        print("Move from {0} down       {1}\n".format(action[0], state))
print("\n\n\n")

print("Found in {0} \n\n".format(end_time_astar - start_time_astar))  # Execution time for A* search


print("\n\n\nSolution for recursive best first search : \n\n")

print("ACTION :                STATE :             \n")
initial = rbfs_path[0].state
print("Initial state ->        {0}\n".format(initial))

for iterator in range(1, len(astar_path)):
    state = rbfs_path[iterator].state
    action = rbfs_solution[iterator - 1]

    if action[1] != ' ':
        print("Move from {0} to {1}        {2}\n".format(action[0], action[1], state))
    else:
        print("Move from {0} down       {1}\n".format(action[0], state))
print("\n\n\n")

print("Found in {0} \n\n".format(end_time_rbfs - start_time_rbfs))  # Execution time for RBFS

execution_time = end - start
print("\n\nExecution time: ", execution_time)  # Total execution time


# Compare searchers

compare_searchers(problems=[problem1, problem2],header=['Searcher','h1(n)', 'h2(n)'],
                  searchers=[astar_search, recursive_best_first_search])
