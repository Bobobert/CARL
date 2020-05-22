#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:01:57 2020

@author: ebecerra
"""

#Rob was here

import helicopter
import numpy as np
import rollout
import tqdm

# Environment parameters
N_ROW = 7
N_COL = 7
P_FIRE = 0.05
P_TREE = 0.1
# Symbols for cells
TREE = 1
FIRE = 9
EMPTY = 0
FREEZE = 4

# Heuristic

# Check all neighbours and current cell
# If fire in any: random to one of them
# If not fire: move to a tree at random
# If not tree: move at random
def get_action_heuristic(observation, env):

    def get_neighborhood(grid, position, env):
        p_row = position[0] + 1
        p_col = position[1] + 1
        # + 1 to adjust to respect to the padded grid
        Grid = env.ExpandGrid()
        return Grid[p_row-1:p_row+2,p_col-1:p_col+2]
       
    def neighborhood_cells_to_actions(grid, position, env):
        position_to_action = ((1,2,3),
                            (4,5,6),
                            (7,8,9))
        neighborhood = get_neighborhood(grid, position, env)
        fire_actions=[]
        tree_actions=[]
        empty_actions=[]
        for row_idx in range(neighborhood.shape[0]):
            for col_idx in range(neighborhood.shape[1]):
                cell = neighborhood[row_idx, col_idx]
                if cell == env.fire:
                    fire_actions.append(position_to_action[row_idx][col_idx])
                elif cell == env.tree:
                    tree_actions.append(position_to_action[row_idx][col_idx])
                elif cell == env.empty:
                    empty_actions.append(position_to_action[row_idx][col_idx])
                else:
                    raise Exception('Error: Unrecognizable forest cell')
        return fire_actions, tree_actions, empty_actions

    grid, position = observation
    #random_idx = lambda obj: np.random.choice(np.arange(0, len(obj)))
    random_idx = lambda obj: 0
    fire_actions, tree_actions, empty_actions = neighborhood_cells_to_actions(grid, position, env)
    if fire_actions:
        idx = random_idx(fire_actions)
        action = fire_actions[idx]
    elif tree_actions:
        idx = random_idx(tree_actions)
        action = tree_actions[idx]
    elif empty_actions:
        idx = random_idx(empty_actions)
        action = empty_actions[idx]
    else:
        raise Exception('Error: Not an action to take')
    return action

import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Environment
    env = helicopter.Helicopter(n_row = N_ROW, n_col = N_COL,
                            p_fire = P_FIRE, p_tree = P_TREE,
                            tree = TREE, fire = FIRE, empty = EMPTY, 
                            freeze=FREEZE) # Freeze, number of cicles to update the environment

    start = time.time()
    # First observation
    observation = env.reset()
    env_1 = env.Copy()
    observation_1 = observation
    env.frame()
    env_1.frame()
    env_1.gif = 100 # Offset to name the gifs from env_1

    # Making checkpoints
    checkpoint_env = env.make_checkpoint()
    checkpoint_env_1 = env_1.make_checkpoint()

    # Creating a policy
    pi = rollout.Policy()
    # Setting up variables
    N_STEPS = 30
    N_TEST = 20
    K = 5
    A = 0.9
    N_SAMPLES = 20
    RO_RESULTS = []
    H_RESULTS = []
    for n_test in range(N_TEST):
        # Running multiple tests
        ro_test_results = []
        h_test_results = []
        # Setting up
        rollout_cost = 0
        heuristic_cost = 0
        env.load_checkpoint(checkpoint_env)
        env_1.load_checkpoint(checkpoint_env_1)
        for i in tqdm.tqdm(range(FREEZE * N_STEPS)): #Times updates, epochs.
            ro_action = rollout.Rollout(env, 
                                        get_action_heuristic,
                                        alpha=A, 
                                        K=K*FREEZE, 
                                        N_samples=N_SAMPLES, 
                                        n_workers=12)
            h_action = get_action_heuristic(observation_1, env)
            # Updating the policy with the rollout action
            pi.new(env.Encode(), ro_action)
            # Applying the new control and moving on to other state
            observation, ro_cost, _, _ = env.step(ro_action)
            observation_1, h_cost, _, _ = env_1.step(h_action)
            if n_test == N_TEST - 1:
                # Framing just the last round
                env.frame()
                env_1.frame()
            rollout_cost += ro_cost
            heuristic_cost += h_cost
            #tqdm.tqdm.write("Iteration {0}, Rollout Cost: {1}, Heuristic Cost: {2}".format(i, ro_cost,h_cost))
            ro_test_results.append(ro_cost)
            h_test_results.append(h_cost)
        RO_RESULTS.append(ro_test_results)
        H_RESULTS.append(h_test_results)
        print('{0}. Total costs: Rollout {1} Heuristic {2}'.format(n_test,
                                                                    rollout_cost, 
                                                                    heuristic_cost))
    
    print("Total time execution %.3f s"%(time.time()-start))
    env.render()
    env_1.render()
    del env_1 
    
    
    # Trying the new policy
    test_cost = 0
    pi_calls = 0
    observation = env.reset()
    N_STEPS = 40
    for _ in tqdm.tqdm(range(N_STEPS*FREEZE)):
        action_pi = pi.call(env.Encode())
        if action_pi:
            observation, cost, done, info = env.step(action_pi)
            pi_calls += 1
            env.frame(title='R_Policy')
        else:
            observation, cost, done, info = env.step(get_action_heuristic(observation, env))
            env.frame(title='H_Policy')
        #_ = env.render_frame(wait_time=0.2)
        test_cost += cost
    print("Total test cost {}".format(test_cost))
    print("Succeful calls to PI:",pi_calls)
    env.render()
    del env

    # Dimentions (N_TEST, FREEZE*N_STEPS)
    RO_RESULTS = np.array(RO_RESULTS)
    H_RESULTS = np.array(H_RESULTS)
    x = range(RO_RESULTS.shape[1])
    plt.xlabel('Step')
    plt.ylabel('Average Cost')
    plt.title('Rollout Test')
    plt.plot(x, np.mean(RO_RESULTS, axis=0), label='Rollout')
    plt.plot(x, np.mean(H_RESULTS, axis=0), label='Heuristic')
    plt.legend()
    plt.savefig("./Runs/Rollout comparison a{0} k{1} n{2} {3}.png".format(int(A*10),
                                                     K, N_SAMPLES,
        time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime())))