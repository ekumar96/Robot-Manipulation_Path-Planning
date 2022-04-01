from __future__ import division
from os import link
import sim
import pybullet as p
import random
import numpy as np
import math
import argparse

MAX_ITERS = 10000
delta_q = 0.3

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)

def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    # ========= DONE: Problem 3 (a), (b) ========
    # Implement RRT code here. Refer to the handout for the pseudocode.
    # This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    # Use visualize_path() to visualize the edges in the exploration tree for part (b)
    # ==================================

    # Function that will either choose the goal configuration or a random point in the configuration space
    def SemiRandomSample(steer_goal_prob):
        if random.uniform(0.0, 1.0) < steer_goal_prob:
            return q_goal
        return np.random.uniform(-2*3.14, 2*3.14, 6)

    # Function that will determine the point in the tree that is closest to any given point
    def Nearest(V, q_rand):
        shortestDist = 1e9
        closest = None
        for vertex in V: # Goes through all vertices
            dist = Distance(q_rand, np.asarray(vertex))
            if dist < shortestDist: 
                # If shortest distance, saves
                shortestDist = dist
                closest = np.asarray(vertex)
        return closest

    # Function to determine distance between two configurations
    def Distance(config_a, config_b):
        #return np.linalg.norm(config_a - config_b, ord=1)
        return np.sqrt(np.sum(np.square(config_a - config_b))) #L2 distance (Euler)
        #return np.sum(np.abs(config_a - config_b)) #L1 (manhattan) distance

    # Function that will return a new point delta_q distance away from q_nearest in the direction of q_rand
    def Steer(q_nearest, q_rand, delta_q):
        # Finds a vector between the points, scaled by the delta_q
        vector = q_rand-q_nearest 
        vector *= delta_q/Distance(q_rand, q_nearest)
        # The new point is the nearest point plus this scaled vector
        q_new = q_nearest+vector
        return q_new

    # Checks if a point is within an obstacle
    def ObstacleFree(q_nearest, q_new):
        return not env.check_collision(q_new)

    # Create the tree
    V = set()
    V.add(tuple(q_init))
    # Dictionary of Edges, which for a tree are E[node] = parent
    E = {} 
    path = []

    #print("\n\nDelta q: " + str(delta_q))
    #print(f"Goal:\t{q_goal}")
    #print(f"Distance:\t{Distance(q_init, q_goal)}")
    for i in range(MAX_ITERS): # Tries to take MAX_ITERS steps to reach goal
        # Gets either a completely random point or the goal point
        q_rand = SemiRandomSample(steer_goal_p)
        # Determines what is the closest point in the current tree to this point
        q_nearest = Nearest(V, q_rand)
        # Gets a new point that is in the direction of this random point delta_q distance away
        q_new = Steer(q_nearest, q_rand, delta_q)

        # If there are no obstacles in this path, adds it to the tree and visualizes the path
        if ObstacleFree(q_nearest, q_new):
            V.add(tuple(q_new))
            E[tuple(q_new)] = tuple(q_nearest)
            visualize_path(q_nearest, q_new, env)
            
            #print(f"Distance Remaining:\t{Distance(q_new, q_goal)}")
            # If this new point is within one step of the goal, then simply add the goal to the tree and declare victory
            if Distance(q_new, q_goal) < delta_q:
                V.add(tuple(q_goal))
                E[tuple(q_goal)] = tuple(q_new)
                thisConfig = tuple(q_goal)
                # Create path by moving up Edge dictionary (tree from node to parent until root (q_init, starting configuration))
                while thisConfig != tuple(q_init):
                    path.insert(0, np.asarray(thisConfig))
                    thisConfig = E[thisConfig]
                path.insert(0, q_init)
                #print(path)
                return path
    # If no path is found in the gievn number of iterations, return no path
    return None

def execute_path(path_conf, env):
    """
    :param path_conf: list of configurations (joint angles) 
    """
    # ========= DONE: Problem 3 (c) ========
    # 1. Execute the path while visualizing the location of joint 5 
    #    (see Figure 2 in homework manual)
    #    You can get the position of joint 5 with:
    #         p.getLinkState(env.robot_body_id, 9)[0]
    #    To visualize the position, you should use sim.SphereMarker
    #    (Hint: declare a list to store the markers)
    # 2. Drop the object (Hint: open gripper, wait, close gripper)
    # 3. Return the robot to original location by retracing the path 
    # ==================================
    pathMarkers = []
    for config in path_conf:
        # Move the robot's joints according to the configurations in the path, 
        # marking the end effector locations with red spheres
        env.move_joints(config) 
        thisMarker = sim.SphereMarker(p.getLinkState(env.robot_body_id, 9)[0])
        pathMarkers.append(thisMarker)
    # Opening the gripper to drop the object, waiting 100 simulation steps, then closing
    env.open_gripper()
    env.step_simulation(100)
    env.close_gripper()
    # Reversing the configuration path and moving in order to return to the original position
    for config in path_conf[::-1]:
        env.move_joints(config)


def get_grasp_position_angle(object_id):
    """
    Get position and orientation (yaw in radians) of the object
    :param object_id: object id
    """
    position, grasp_angle = np.zeros(3), 0
    # ========= DONE: Problem 2 (a) ============
    # You will p.getBasePositionAndOrientation
    # Refer to Pybullet documentation about this method
    # Pay attention that p.getBasePositionAndOrientation returns a position and a quaternion
    # while we want a position and a single angle in radians (yaw)
    # You can use p.getEulerFromQuaternion
    # ==================================

    # We can get the objects base position and orientation using PyBullet's function
    position, quaternion = p.getBasePositionAndOrientation(object_id)

    # We convert the orientation to Euler angles in radians. Since we only want yaw, we take the Z ([2])
    grasp_angle = p.getEulerFromQuaternion(quaternion)[2]

    return position, grasp_angle

def test_robot_movement(num_trials, env):
    # Problem 1: Basic robot movement
    # Implement env.move_tool function in sim.py. More details in env.move_tool description
    passed = 0
    for i in range(num_trials):
        # Choose a reachable end-effector position and orientation
        random_position = env._workspace1_bounds[:, 0] + 0.15 + \
            np.random.random_sample((3)) * (env._workspace1_bounds[:, 1] - env._workspace1_bounds[:, 0] - 0.15)
        random_orientation = np.random.random_sample((3)) * np.pi / 4 - np.pi / 8
        random_orientation[1] += np.pi
        random_orientation = p.getQuaternionFromEuler(random_orientation)
        marker = sim.SphereMarker(position=random_position, radius=0.03, orientation=random_orientation)
        # Move tool
        env.move_tool(random_position, random_orientation)
        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
        link_marker = sim.SphereMarker(link_state[0], radius=0.03, orientation=link_state[1], rgba_color=[0, 1, 0, 0.8])
        # Test position
        delta_pos = np.max(np.abs(np.array(link_state[0]) - random_position))
        delta_orn = np.max(np.abs(np.array(link_state[1]) - random_orientation))
        if  delta_pos <= 1e-3 and delta_orn <= 1e-3:
            passed += 1
        env.step_simulation(1000)
        # Return to robot's home configuration
        env.robot_go_home()
        del marker, link_marker
    print(f"[Robot Movement] {passed} / {num_trials} cases passed")

def test_grasping(num_trials, env):
    # Problem 2: Grasping
    passed = 0
    for _ in range(num_trials):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)

        # Test for grasping success (this test is a necessary condition, not sufficient):
        object_z = p.getBasePositionAndOrientation(object_id)[0][2]
        if object_z >= 0.2:
            passed += 1
        env.reset_objects()
    print(f"[Grasping] {passed} / {num_trials} cases passed")

def test_rrt(num_trials, env):
    # Problem 3: RRT Implementation
    passed = 0
    for _ in range(num_trials):
        # grasp the object
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            # get a list of robot configuration in small step sizes
            path_conf = rrt(env.robot_home_joint_config,
                            env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5, env)
            if path_conf is None:
                print(
                    "no collision-free path is found within the time budget. continuing ...")
            else:
                env.set_joint_positions(env.robot_home_joint_config)
                execute_path(path_conf, env)
            p.removeAllUserDebugItems()

        env.robot_go_home()

        # Test if the object was actually transferred to the second bin
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and\
            object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and\
            object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()

    print(f"[RRT Object Transfer] {passed} / {num_trials} cases passed")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-part', type=str,
                        help='part')
    parser.add_argument('-n', type=int, default=3,
                        help='number of trials')
    parser.add_argument('-disp', action='store_true')
    args = parser.parse_args()

    random.seed(1)
    object_shapes = [
        "assets/objects/cube.urdf",
    ]
    env = sim.PyBulletSim(object_shapes = object_shapes, gui=args.disp)
    num_trials = args.n

    if args.part in ["2", "3", "all"]:
        env.load_gripper()
    if args.part in ["1", 'all']:
        test_robot_movement(num_trials, env)
    if args.part in ["2", 'all']:
        test_grasping(num_trials, env)
    if args.part in ["3", 'all']:
        test_rrt(num_trials, env)
    