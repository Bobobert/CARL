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
N_ROW = 1
N_COL = 20
P_FIRE = 0.05
P_TREE = 0.1
# Symbols for cells
TREE = 1
FIRE = 9
EMPTY = 0

# Heuristic

# Check all neighbours and current cell
# If fire in any: random to one of them
# If not fire: move to a tree at random
# If not tree: move at random
def get_action_heuristic(observation, env, greedy=False):

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
    random_idx = lambda obj: np.random.choice(np.arange(0, len(obj)))
    fire_actions, tree_actions, empty_actions = neighborhood_cells_to_actions(grid, position, env)
    if fire_actions:
        idx = random_idx(fire_actions)
        action = fire_actions[idx]
    elif greedy:
        action = 5 # Do nothing
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

if __name__ == '__main__':
    # Environment
    env = helicopter.Helicopter(n_row = N_ROW, n_col = N_COL,
                            p_fire = P_FIRE, p_tree = P_TREE,
                            tree = TREE, fire = FIRE, empty = EMPTY, 
                            freeze=3) # Freeze, number of cicles to update the environment

    start = time.time()
    # First observation
    observation = env.reset()
    env.frame()
    pi = rollout.Policy()
    total_cost = 0
    N_STEPS = 50
    for i in tqdm.tqdm(range(env.freeze * N_STEPS)): #Times updates, epochs.
        action = rollout.Rollout(env, get_action_heuristic, K=10*env.freeze, N_samples=20, n_workers=1)
        pi.new(env.Encode(), action)
        # Applying the new control and moving on to other state
        observation, cost, done, info = env.step(action)
        env.frame()
        total_cost += cost
        tqdm.tqdm.write("Iteration {0}, Cost: {1}".format(i, cost))
    env.render()
    print('Total rollout cost: {}'.format(total_cost))
    print("Total time execution %.3f s"%(time.time()-start))

    # Trying the heuristic alone
    observation = env.reset()
    env.frame()
    total_cost = 0
    for i in tqdm.tqdm(range(env.freeze * N_STEPS)):
        action = get_action_heuristic(observation, env)
        observation, cost, done, info = env.step(action)
        env.frame()
        total_cost += cost
    env.render()
    print('Total H cost: {}'.format(total_cost))

    # Trying the new policy
    test_cost = 0
    pi_calls = 0
    observation = env.reset()
    EPOCHS = 20
    for _ in tqdm.tqdm(range(EPOCHS)):
        action_pi = pi.call(env.Encode())
        if action_pi:
            observation, cost, done, info = env.step(action_pi)
            pi_calls += 1
        else:
            observation, cost, done, info = env.step(get_action_heuristic(observation, env))
        env.frame()
        #_ = env.render_frame(wait_time=0.2)
        test_cost += cost
    print("Total test cost {}".format(test_cost))
    print("Succeful calls to PI:",pi_calls)
    env.render()
    #print(pi)
