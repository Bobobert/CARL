# CARL
Control of Automata with Reinforcement Learning

This strategy uses Rollout as the core Reinforcement Learning technique to maximize total reward
obtained by each step in the environment.

RL_ForestF.py is the main file that contain simulation,environment and rollout parameters. Also 
implement a Policy class that saves the state-action-values and saves results after run.

Heuristic.py is a module that contains the actual heuristic: Search in the neighborhood
of the agent red cells and calculate density by 8 different zones: 
  Up - Down - Left - Right -Up_Left - Up_Right - Down-Left - Down_Right.
and take an action based on max density zones.

Rollout.py implements the rollout strategy by 1-step lookahead (maximize over the first
action-control) and approximate the future reward by using the heuristic k-times). System
dynamic is stochastic (probability f and p) so it implements N-simulations to calculate
expected value over each action. It returns a random action to explore at first stages of the algorithm
and a deterministic action after some time (epsilon value affect this).
