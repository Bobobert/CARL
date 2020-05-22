# Rob was here

# Math
import numpy as np
import math

# Parallel exec
from multiprocessing import Pool
import os

class Policy():
    # Class for a tabular poilcy
    def __init__(self, ):
        self.pi = dict()

    def new(self, key, control):
        try:
            # The key state is already seen
            controls, freq, tot_freq = self.pi[key]
            if control in controls:
                i = 0
                for j in controls:
                    if j == control:
                        break
                    else:
                        i += 1
                freq[i] += 1
            else:
                controls.append(control)
                freq.append(1)
            tot_freq += 1
            self.pi[key] = (controls, freq, tot_freq)
        except:
            # The key state is new, so their values are created
            self.pi[key] = ([control],[1],1)

    def call(self, key):
        # Uniformly distributed
        try:
            controls, freq, tot_freq = self.pi[key]
            freq_r = np.array(freq) / tot_freq
            return np.random.choice(controls, 1, p=freq_r)[0]
        except:
            return False

    def __repr__(self):
        s = "Policy with {0} states".format(len(self.pi.keys()))
        for key in self.pi.keys():
            s+="\nState {0} controls:{1}".format(key,self.pi[key][0])
        return s

    def __del__(self):
        return None

class R_ITER():
    ### Class for to save a bit of space in creation of iterables with repeated items
    ### Env must have a .Copy method
    def __init__(self, env, H, alpha, Action_set, K, N_samples):
        self.env = env
        self.H_ref = H
        self.alpha = alpha
        self.actions = Action_set
        self.K = K
        self.N_samples = N_samples
        self.index = 0
        self.TOP = len(Action_set)
        return None

    def __iter__(self):
        # iterable method
        return self

    def __next__(self):
        # iterating through the object
        if self.index == self.TOP:
            raise StopIteration
        else:
            i = self.index
            self.index += 1
            return (self.env.Copy(), # Always a new copy of the environment. This is a new object.
                    self.actions[i], # Different action
                    self.H_ref, 
                    self.alpha,
                    self.K,
                    self.N_samples)

    def __del__(self):
        return None

def sample_trayectory(ARG):
    # This sampler is design to run in one thread for parallel calling
    # Make sure the env variable is referencing a copy of the original env
    env, action, H, alpha, k, N_samples = ARG 
    initial_checkpoint = env.make_checkpoint()
    total_cost = 0
    for _ in range(N_samples):
        ALPHA = alpha
        # Executing the first action to minimize
        observation, ac_cost, done, info = env.step(action)
        ac_cost = ALPHA * ac_cost
        # Executing the heuristic for k steps
        for _ in range(k):
            action_from_H = H(observation, env)
            observation, cost, done, info = env.step(action_from_H)
            # Extract heuristic cost
            ALPHA = ALPHA * alpha
            ac_cost += cost * ALPHA
        #End of the heuristic running
        # Termination value is estimated to 0 at the moment. 
        # Here could be an approximator from the last observationfrom the environment.
        ac_cost += 0
        # Adding the sample cost to the average cost of the samples
        total_cost += ac_cost / N_samples
        # Restarting the environment from the initial state.
        env.load_checkpoint(initial_checkpoint)
    del env # Closing the copy of the environment
    return (action, total_cost)

def Rollout(env, H, alpha=1, K=-1, N_samples=10, n_workers=-1):
    ### Funtion to do a rollout from the actions available and the use of the Heuristic H.
    ### The argument k is to truncate how many steps the heuristic is going to execute.
    ### Environment has to have a copy function.
    # Theorical variables
    actions = list(env.actions_set)
    # Executive variables
    cpus, parallel = 0, False
    if n_workers == 1:
        parallel = False
        cpus = 1
    elif n_workers < 0 or n_workers > os.cpu_count():
        parallel = True
        cpus = os.cpu_count() // 2  # ALL YOUR BASE ARE BELONG TO US, bye SMT
    else:
        parallel = True
        cpus = n_workers

    if K < 0: # Classic Rollout, the heuristic is ran until the last state is reached
        # To define what's the end
        None
    else: #Truncated Rollout
        # Creation of an iterable with the action set to make the samples
        #to_evaluate = [(env.Copy(), action, H, K, N_samples) for action in actions]
        to_evaluate = R_ITER(env, H, alpha, actions, K, N_samples)
        if parallel:
            # Execute the sampling in multiple threads of the cpu with map function
            # Usually this perfoms better than single thread for long trayectories/number of samples
            # If the trayectories are short, of the number of samples is little, a sequential run could perform better
            p = Pool(processes=cpus)
            costs = p.imap_unordered(sample_trayectory,
                                     to_evaluate, 
                                     math.ceil(len(actions)/cpus))
        else:
            # Execute an evaluation at a time in one cpu thread
            costs = []
            for i in to_evaluate:
                costs.append(sample_trayectory(i))
    # costs is the list of the costs of the sampled trayectories from their respective action
    min_cost, r_action = np.inf, None
    for action, cost in costs:
        if cost < min_cost:
            r_action = action # Saving the action with less cost
            min_cost = cost 
    if parallel: # Terminating the pool of processes in this rollout
        p.close()
        p.join()
    del to_evaluate # Cleaning the iterable
    return r_action