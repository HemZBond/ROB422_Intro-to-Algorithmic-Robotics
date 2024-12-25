import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue

def normalize_angle(angle: float) -> float:
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle

def find_neighbors(point, mode=4):
    delta_x, delta_y, delta_theta = 0.1, 0.1, np.pi / 2
    if mode == 4:
        motions = [(delta_x, 0, 0), (0, delta_y, 0), (-delta_x, 0, 0), (0, -delta_y, 0), (0, 0, delta_theta), (0, 0, -delta_theta)]
    elif mode == 8:
        motions = [(delta_x, 0, 0), (delta_x, 0, delta_theta), (delta_x, 0, -delta_theta), (0, delta_y, 0),
                   (0, delta_y, delta_theta), (0, delta_y, -delta_theta), (-delta_x, 0, 0), (-delta_x, 0, delta_theta),
                   (-delta_x, 0, -delta_theta), (0, -delta_y, 0), (0, -delta_y, delta_theta), (0, -delta_y, -delta_theta),
                   (0, 0, delta_theta), (0, 0, -delta_theta), (delta_x, delta_y, 0), (delta_x, delta_y, delta_theta),
                   (delta_x, delta_y, -delta_theta), (-delta_x, delta_y, 0), (-delta_x, delta_y, delta_theta),
                   (-delta_x, delta_y, -delta_theta), (delta_x, -delta_y, 0), (delta_x, -delta_y, delta_theta),
                   (delta_x, -delta_y, -delta_theta), (-delta_x, -delta_y, 0), (-delta_x, -delta_y, delta_theta),
                   (-delta_x, -delta_y, -delta_theta)]  # diagonal actions
    return [(point[0] + motion[0], point[1] + motion[1], normalize_angle(point[2] + motion[2])) for motion in motions]

def heuristic(node, goal):
    angle_diff = abs(normalize_angle(node[2] - goal[2]))
    return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2 + min(angle_diff, 2 * np.pi - angle_diff) ** 2)

def rebuild_path(closed_dict, final_node):
    result_path = []
    while final_node is not None:
        result_path.append(final_node)
        final_node = closed_dict.get(final_node)
    return result_path[::-1]

def astar_search(start_state, goal_state, collision_function):
    pq = PriorityQueue()
    pq.put((0, start_state))
    g_costs = {start_state: 0}
    f_costs = {start_state: heuristic(start_state, goal_state)}
    search_dict = {}
    closed_set = set()
    colliding_states = set()
    collision_free_states = set()

    while not pq.empty():
        _, current_state = pq.get()

        if current_state in closed_set:
            continue

        if collision_function(current_state):
            colliding_states.add(current_state)
            continue

        closed_set.add(current_state)
        collision_free_states.add(current_state)

        if abs(current_state[0] - goal_state[0]) < 1e-4 and abs(current_state[1] - goal_state[1]) < 1e-4 and abs(normalize_angle(current_state[2] - goal_state[2])) < 1e-4:
            return g_costs[current_state], colliding_states, collision_free_states, rebuild_path(search_dict, current_state)

        for neighbor in find_neighbors(current_state):
            tentative_g_cost = g_costs[current_state] + heuristic(current_state, neighbor)

            if neighbor in closed_set or (neighbor in g_costs and tentative_g_cost >= g_costs[neighbor]):
                continue

            search_dict[neighbor] = current_state
            g_costs[neighbor] = tentative_g_cost
            f_costs[neighbor] = tentative_g_cost + heuristic(neighbor, goal_state)
            pq.put((f_costs[neighbor], neighbor))

    return g_costs[current_state], colliding_states, collision_free_states, None

#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
    draw_graph = True
    # draw_graph = False
    g_cost = 0.0
    collision_list = set()
    collision_free_list = set()
    g_cost, collision_list, collision_free_list, path = astar_search(start_config, goal_config, collision_fn)
    
    if not path:
        print("No Solution Found.")
    if path:
        print("Path cost: ", g_cost)
    
    if draw_graph:
        for collision in collision_list:
            draw_sphere_marker((collision[0], collision[1], 1.0), 0.05, (1, 0, 0, 1))
        for collision_free in collision_free_list:
            draw_sphere_marker((collision_free[0], collision_free[1], 1.0), 0.05, (0, 0, 1, 1))
        for p in path:
            draw_sphere_marker((p[0], p[1], 1.1), 0.05, (0, 0, 0, 1)) 
    
    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()