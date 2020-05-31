import Heuristic
import random
import numpy as np
import math
import helicopter
import copy 


#Rollout Strategy
def rollout(env, H,observation,K,A,N_SAMPLES,vision,epsilon):  
    grid, pos, remain_steps= observation
    #Get available actions for helicopter position
    actions= env.available_actions(pos)   
    #Make an iterable for all available actions
    NEXT_STEPS=ITERABLE_STEPS(env,H,actions)
    q_values={}
    #Iterate over all actions
    for action_step in NEXT_STEPS:
        #Make simulations for each action and return average return rewards       
        action_s, q_value= Simulation(action_step,K,A,N_SAMPLES,vision)        
        q_values[action_s]=q_value
    #Exploration vs explotation based on epsilon value   
    if random.random() > epsilon:
        print("Explotation")
        #Maximization over all possible action controls (max u_xk)
        action_max= max(q_values.keys(), key=(lambda k: q_values[k]))
    else:
        print("Exploration")
        #Random action
        action_max=random.choice(list(q_values.keys()))   
    return(action_max,q_values[action_max])

class ITERABLE_STEPS():
    def __init__(self,env, H, actions):
        self.enviroment=env
        self.H_ref = H      
        self.actions = actions
        self.index = 0
        self.TOP = len(actions)
        return None

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.TOP:
            raise StopIteration
        else:
            i = self.index
            self.index += 1
            return (Copy(self.enviroment), self.actions[i], self.H_ref)     

    def __del__(self):
        return None
    
def Simulation(simulation_args,K,A,N_SAMPLES,vision):    
    total_q_value=0
    alpha=A
    env_c, action, H = simulation_args  
    #Make first step(1-step-lookahead deterministic)
    observation, g_k, done, info = env_c.step(action)
    checkpoint = env_c.make_checkpoint()
    total_q_value=0   
    #Follow the heuristic K-steps   
    for n_samples in range(N_SAMPLES):        
        t_cost=0
        q_value=0        
        env_c.load_checkpoint(checkpoint)      
        for r_iter in range(K):                
                action_H = H(env_c,observation,vision) 
                observation, cost, done, info = env_c.step(action_H)                
                cost=cost*alpha 
                alpha=alpha*A
                t_cost=t_cost + cost               
        q_value=g_k+t_cost #q-value for one sample
        total_q_value=total_q_value+q_value #Adding up values of N_SAMPLES               
    total_q_value=total_q_value/N_SAMPLES #Average of q value cost over all samples   
    del env_c
    return (action,total_q_value)

def Copy(env):
    n_env = helicopter.EnvMakerForestFire(init_pos_row=env.pos_row,init_pos_col=env.pos_col,n_row = env.n_row, n_col = env.n_col,
                                          p_tree = env.p_tree, p_fire =env.p_fire, moves_before_updating = env.moves_before_updating,
                                          reward_type = env.reward_type, reward_tree = env.reward_tree,reward_fire = env.reward_fire,
                                          reward_empty =env.reward_empty, reward_hit = env.reward_hit,sub_tree = env.sub_tree,
                                          sub_empty = env.sub_empty, sub_fire = env.sub_fire, sub_rock = env.sub_rock,sub_lake = env.sub_lake,
                                          ip_tree = env.ip_tree, ip_empty =env.ip_empty, ip_fire =env.ip_fire, ip_rock = env.ip_rock,
                                          ip_lake = env.ip_lake)
    n_env.grid = copy.deepcopy(env.grid)      
    n_env.total_reward = copy.deepcopy(env.total_reward)   
    n_env.total_hits=copy.deepcopy(env.total_hits)
    n_env.remaining_moves=copy.deepcopy(env.remaining_moves)
    
    return n_env