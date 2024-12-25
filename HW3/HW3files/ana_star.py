import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
from queue import PriorityQueue
import time

# Helper function to wrap angles to [-pi, pi]
def wrap_to_pi(angle: float) -> float:
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle

# Generate neighbors in 8-connected space
def get_neighbors(node, mode=8):
    dx, dy, dtheta = 0.1, 0.1, np.pi / 2
    if mode == 8:
        moves = [(dx, 0, 0), (0, dy, 0), (-dx, 0, 0), (0, -dy, 0),
                 (dx, dy, 0), (-dx, dy, 0), (dx, -dy, 0), (-dx, -dy, 0),
                 (0, 0, dtheta), (0, 0, -dtheta)]
    return [(node[0] + move[0], node[1] + move[1], wrap_to_pi(node[2] + move[2])) for move in moves]

# Cost function
def cost(node, neighbor):
    dtheta = abs(wrap_to_pi(node[2] - neighbor[2]))
    return np.sqrt((node[0] - neighbor[0])**2 + (node[1] - neighbor[1])**2 + dtheta**2)

# Path reconstruction
def reconstruct_path(close_list, current_node):
    path = []
    while current_node is not None:
        path.append(current_node)
        current_node = close_list.get(current_node)
    return path[::-1]

# A* Algorithm
def a_star(start, goal, collision_fn):
    open_list = PriorityQueue()
    open_list.put((0, start))
    gcosts = {start: 0}
    fcosts = {start: cost(start, goal)}
    search_list = {}
    close_list = set()

    while not open_list.empty():
        _, current_node = open_list.get()

        if current_node in close_list:
            continue

        if abs(current_node[0] - goal[0]) < 1e-4 and abs(current_node[1] - goal[1]) < 1e-4 and abs(wrap_to_pi(current_node[2] - goal[2])) < 1e-4:
            return gcosts[current_node], reconstruct_path(search_list, current_node)

        close_list.add(current_node)

        for neighbor in get_neighbors(current_node):
            if collision_fn(neighbor):
                continue

            proposed_gcost = gcosts[current_node] + cost(current_node, neighbor)

            if neighbor in close_list or (neighbor in gcosts and proposed_gcost >= gcosts[neighbor]):
                continue

            search_list[neighbor] = current_node
            gcosts[neighbor] = proposed_gcost
            fcosts[neighbor] = proposed_gcost + cost(neighbor, goal)
            open_list.put((fcosts[neighbor], neighbor))

    return None, None

# ANA* Algorithm
def ana_star(start, goal, collision_fn, heuristic_fn, timeout=60.0):
    open_list = PriorityQueue()
    gcosts = {start: 0}
    fcosts = {start: heuristic_fn(start, goal)}
    incons = set()
    search_list = {}
    best_path_cost = float('inf')
    best_path = None
    collision_free_list = set()

    start_time = time.time()
    open_list.put((fcosts[start], start))

    while not open_list.empty() and time.time() - start_time < timeout:
        _, current_node = open_list.get()

        if abs(current_node[0] - goal[0]) < 1e-4 and abs(current_node[1] - goal[1]) < 1e-4 and abs(wrap_to_pi(current_node[2] - goal[2])) < 1e-4:
            path = reconstruct_path(search_list, current_node)
            current_cost = gcosts[current_node]
            if current_cost < best_path_cost:
                best_path_cost = current_cost
                best_path = path

        for neighbor in get_neighbors(current_node):
            if collision_fn(neighbor):
                continue

            collision_free_list.add(neighbor)
            proposed_gcost = gcosts[current_node] + cost(current_node, neighbor)

            if neighbor not in gcosts or proposed_gcost < gcosts[neighbor]:
                gcosts[neighbor] = proposed_gcost
                fcosts[neighbor] = proposed_gcost + heuristic_fn(neighbor, goal)
                search_list[neighbor] = current_node
                if fcosts[neighbor] <= best_path_cost:
                    open_list.put((fcosts[neighbor], neighbor))
                else:
                    incons.add(neighbor)

    return best_path_cost, best_path, collision_free_list

# Heuristics
def euclidean_heuristic(node, goal):
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def custom_heuristic(node, goal):
    dtheta = abs(wrap_to_pi(node[2] - goal[2]))
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2) + min(dtheta, 2 * np.pi - dtheta)

# Main Function
def main():
    connect(use_gui=True)
    robots, obstacles = load_env('pr2doorway.json')
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi / 2)
    timeout = 60.0

    # A* with Euclidean heuristic
    print("Running A* with Euclidean heuristic...")
    g_cost, path = a_star(start_config, goal_config, collision_fn)
    if path:
        print(f"A* Euclidean path cost: {g_cost}")

    # ANA* with Euclidean heuristic
    print("Running ANA* with Euclidean heuristic...")
    best_path_cost, best_path, _ = ana_star(start_config, goal_config, collision_fn, euclidean_heuristic, timeout)
    print(f"ANA* Euclidean best path cost: {best_path_cost}")

    # ANA* with Custom heuristic
    print("Running ANA* with Custom heuristic...")
    best_path_cost_custom, best_path_custom, _ = ana_star(start_config, goal_config, collision_fn, custom_heuristic, timeout)
    print(f"ANA* Custom best path cost: {best_path_cost_custom}")

    # Visualization and graphs can be added here if necessary

    disconnect()

if __name__ == '__main__':
    main()
