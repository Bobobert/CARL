import numpy as np
import math
import helicopter
import tqdm
import random
import matplotlib.pyplot as plt
import statistics as st
import time
import copy 
import Heuristic 
import Rollout 

#Create a Policy and fills with find states,controls,qvalues and frequency 
class Policy():
    #Creates a new Policy    
    def __init__(self, ):
        self.policy = dict()

    #For every action-state calls this method to update policy    
    def new(self,key,control,value):
        print(key)
        try:
            # The key state is already seen
            controls, values, actions_freq , total_freq = self.policy[key]
            # Iteration over controls available in key until reach actual control
            if control in controls:
                print("Estado con acción ya escogida")
                print(self.policy[key])
                i = 0
                for j in controls:
                    if j == control:
                        break
                    else:
                        i += 1
                #Update qvalue by averaging existing value for this state-action
                values[i]=(values[i]+value)/2
                actions_freq[i]+=1                               
            #If state has not this control, then add it with actual reward and update freq
            else:
                print("Estado con acción aún no vista")
                print(self.policy[key])
                controls.append(control)
                values.append(value)
                actions_freq.append(1)
            total_freq+=1
            self.policy[key] = (controls,values, actions_freq, total_freq)
            print(self.policy[key])
        except:
            # The key state is new, so their values are created
            #Set of controls, Set of q_values,freq of choosing this control, total freq of being in this state
            print("Nuevo estado agregado")            
            self.policy[key] = ([control],[value],[1],1) 
            print(self.policy[key])
    
    #Actual state of Policy        
    def __repr__(self):
        s = "Q-Table with {0} states".format(len(self.policy.keys()))
        for key in self.policy.keys():
            s+="\nState {0} controls:{1}".format(key,self.policy[key][0])
        return s
    
    #Return the best action to take given a key state. Two modes. 
    def call(self, key,mode):        
        try:             
            (controls,q_values,freqs,total_freq)= self.policy[key]
            #Select action based on a probability determined by its frequency and total frequency
            if mode=="stochastic":
                print("Seleccion estocastica de controles")
                freq_r = np.array(freqs) / total_freq
                action_max=np.random.choice(controls, 1, p=freq_r)[0]
                return action_max
            #Select action based only on best qvalue
            elif mode=="deterministic":
                print("selección determinista de controles")
                action_max= controls[q_values.index(max(q_values))]
                return  action_max
            else:
                return False
        except:
            return False

#Copy a given enviroment. Deepcopy is used to create a structure in different memmory position (In order to not modify the original environment variables)
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

# Environment parameters
N_ROW = 16              #Grid Rows
N_COL = 16              #Grid Columns
Init_Row=7              #Helicopter initial row
Init_Col=7              #Helicopter initial col
P_FIRE = 0.03           #Probability to turn a green cell into ared cell
P_TREE = 0.1            #Probability to turn an empty cell into a green cell
# Symbols for cells
TREE = 0
FIRE = 2
EMPTY = 1
FREEZE = 8              #Movements of Helicopter after update Automata

if __name__ == '__main__':
    #Create a new enviroment with initial parameters
    env = helicopter.EnvMakerForestFire(n_row = N_ROW, n_col = N_COL, p_tree = P_TREE, p_fire = P_FIRE,
                 init_pos_row = Init_Row, init_pos_col = Init_Col, moves_before_updating = FREEZE,  
                 tree = TREE, empty = EMPTY, fire = FIRE)  
  
    # First observation
    observation = env.reset()
    
    #Create a copy of enviroment with initial observation
    env_1 = Copy(env)
    observation_1 = observation   
    
    # Making checkpoints
    checkpoint_env = env.make_checkpoint()
    checkpoint_env_1 = env_1.make_checkpoint()
    
    #Create a new empty Policy     
    policy= Policy()
    
    #Rollout Variables
    N_TEST= 20      #Number of total tests simulations
    N_STEPS=50      #Environment updates(50 updates of environment and 20*8 movements of agent)
    A=0.9           #Discount factor for future rewards
    K=10            #Rollout Steps on horizon for the heuristic 
    N_SAMPLES=20    #Number of samples trajectories in rollout to calculate expected value
    vision= 1       #Range to lookup in cells in helicpter heuristic  
    epsilon=0.99    #Epsilon for exploration or explotation in state space 
    
    RO_RESULTS=[]
    H_RESULTS=[]
    RO_RESULTS_C=[]
    H_RESULTS_C=[]

    start = time.time()
    for n_test in range(N_TEST):        
        print("Test:",n_test)           
        env.load_checkpoint(checkpoint_env)
        env_1.load_checkpoint(checkpoint_env_1) 
        rollout_cost=0
        heuristic_cost=0
        rollout_cost_step=[]
        heuristic_cost_step=[]        
        for i in tqdm.tqdm(range(FREEZE * N_STEPS)):
            #Calls Rollout Strategy and returns action,qvalue            
            r_action, q_value=Rollout.rollout(env,Heuristic.Heuristic,observation,K,A,N_SAMPLES,vision,epsilon)
            #Update epsilon it goes from stochastic to deterministic 
            epsilon=epsilon*0.99
            #Calls Heuristic and return best action
            h_action=Heuristic.Heuristic(env,observation_1,vision)
            #Update Policy            
            policy.new(env.Encode(),r_action,q_value)            
            print("Rollout Action:",r_action)
            print("Heuristic Action:",h_action)
            #Helicopter take an action based on Rollout strategy and heuristic
            observation, ro_cost, _, _ = env.step(r_action)
            observation_1, h_cost, _, _ = env_1.step(h_action)  
            #Update Rollout Total cost
            rollout_cost += ro_cost  #Acumulative cost for rollout          
            rollout_cost_step.append(rollout_cost)  #List of cost over time
            #Update Heuristic Total cost
            heuristic_cost += h_cost
            heuristic_cost_step.append(heuristic_cost)
            print("Rollout in step {} is: {}".format(i,rollout_cost))
            print("Heuristic in step {} is: {}".format(i,heuristic_cost))
        print("Rollout:",rollout_cost)
        print("heuristic:",heuristic_cost)
        #Costs p/test
        RO_RESULTS.append(rollout_cost)             
        H_RESULTS.append(heuristic_cost)
        #Cumulative costs p/test
        RO_RESULTS_C.append(rollout_cost_step)
        H_RESULTS_C.append(heuristic_cost_step)
    print("Total time execution %.3f s"%(time.time()-start))
    

    #Plotting and saving results

    #OBTAIN AVERAGE COSTS PER STAGE
    ITER=[]
    ITER2=[]
    RO_RESULTS_C=np.array(RO_RESULTS_C)    
    for i in range(RO_RESULTS_C.shape[1]):
        IT=[]
        IT2=[]
        for j in range(RO_RESULTS_C.shape[0]):
            IT.append(RO_RESULTS_C[j][i])
            IT2.append(H_RESULTS_C[j][i])
        ITER.append(st.mean(IT))
        ITER2.append(st.mean(IT2))        
    

    #Save Average Reward Results
    s = "Average rewards by iteration"  
    s+="\nRollout Avg Results\n" 
    f= open("Avg_1s_1v.txt","w+")
    for i in ITER:
        s+="{0} ".format(i)
    s+="\nHeuristic Avg Results\n"    
    for i in ITER2:
        s+="{0} ".format(i)
    f.write(s)    
    f.close()

    x = np.arange(RO_RESULTS_C.shape[1])    
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average reward over 20 test (1 step-1 vision)')
    plt.plot(x,ITER ,label='Rollout')
    plt.plot(x,ITER2, label='Heuristic')
    plt.legend()
    plt.savefig('AverageRolloutTest.png')
    
    #Graph of total reward by test
    RO_RESULTS = np.array(RO_RESULTS)    
    H_RESULTS = np.array(H_RESULTS)    

    #Save total reward results
    s ="Total reward by Test"  
    s+="\nRollout Results\n" 
    f= open("Total_1s_1v.txt","w+")
    for i in RO_RESULTS:
        s+="{0} ".format(i)
    s+="\nHeuristic Avg Results\n"    
    for i in H_RESULTS:
        s+="{0} ".format(i)
    f.write(s)    
    f.close()

    x = np.arange(3)    
    plt.xlabel('Test')
    plt.ylabel('Final Total Reward (1 step- 1 vision)')
    plt.title('Rollout Tests')
    plt.plot(x,RO_RESULTS ,label='Rollout')
    plt.plot(x,H_RESULTS, label='Heuristic')
    plt.legend() 
    plt.savefig('TotalRewardRolloutTest.png') 
    
    #Simulation of enviroment with Trained Policy    
    observation = env.reset()
    env_1 = env.Copia()
    observation_=observation
    fig=env.render()
    fig.savefig('Pictures/Env.png')
    total_reward = 0
    total_reward_step=[]
    total_reward_h = 0
    total_reward_step_h=[]   
    N_STEPS=50
    for j in tqdm.tqdm(range(N_STEPS*FREEZE)):
        p_action=policy.call(env.Encode(),"stochastic")        
        h_action=Heuristic.Heuristic(env,observation_,vision)
        if p_action:
            print("Tomando Accion de Politica")
            observation, cost, done, info = env.step(p_action)           
            fig=env.render()
            #plt.close()
            s='Pictures/Env' + str(j) + '.png'
            fig.savefig(s)
        else:
            print("Calculando Rollout")
            action=Rollout.rollout(env,Heuristic.Heuristic,observation,K,A,N_SAMPLES,vision,0.1)
            print(action[0])
            observation, cost, done, info = env.step(action[0])
            fig=env.render()
            #plt.close()
            s='Pictures/Env' + str(j) + '.png'
            fig.savefig(s)
        observation_, cost_, done_, info_ = env_1.step(h_action)
        total_reward += cost
        total_reward_h += cost_
        total_reward_step.append(total_reward)
        total_reward_step_h.append(total_reward_h)
        
    print(total_reward)
    print(total_reward_h)    

    #End of simulation
    
    total_reward_step = np.array(total_reward_step)    
    total_reward_step_h = np.array(total_reward_step_h)

    #Save simulation Results
    s = "Total reward by Step"  
    s+="\nRollout Results\n" 
    f= open("TotalStep_1s_1v.txt","w+")
    for i in total_reward_step:
        s+="{0} ".format(i)
    s+="\nHeuristic Avg Results\n"    
    for i in total_reward_step_h:
        s+="{0} ".format(i)
    f.write(s)    
    f.close() 
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward over time')
    plt.plot(total_reward_step ,label='Rollout')
    plt.plot(total_reward_step_h, label='Heuristic')
    plt.legend() 
    plt.savefig('UsingPolicy.png')    
    

    #Save Policy
    s = "Q-Table with {0} states".format(len(policy.policy.keys()))
    states=policy.policy.keys()
    f= open("Policy_1s_1v.txt","w+")
    for state in states:
        s+="\n{0} {1} {2} {3} {4}".format(state,policy.policy[state][0],policy.policy[state][1],policy.policy[state][2],policy.policy[state][3])
    f.write(s)    
    f.close()