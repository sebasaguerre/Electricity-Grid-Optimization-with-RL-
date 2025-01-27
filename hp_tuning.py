from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

# import numpy as np
# import torch
# import argparse
from env import DataCenterEnv
from train_q_learning_gpu import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"hp tuning Using device: {device}")

gamma = 0.95 # discount factor should be lower
start_epsilon = 0.9
end_epsilon = 0.05
pbounds = {"episodes" : (5000, 12000), "alpha" : (0.01, 0.15), "end_epsilon": (0.01, 0.1)}
# pbounds = {"episodes" : (5000, 12000), "alpha" : (0.01, 0.15), "gamma": (0.99, 0.999), "start_epsilon": (0.5, 1.0), "end_epsilon": (0.01, 0.1)}
def qlearn_wrapper(alpha, gamma, episodes, start_epsilon, end_epsilon):
    episodes = int(round(episodes))
    total_rwd, ep_rwd = main_train_qlearn(alpha, gamma, episodes, device, start_epsilon, end_epsilon)
    return total_rwd


# create instance of optimizer 
optimizer = BayesianOptimization(
    f = qlearn_wrapper,
    pbounds = pbounds,
    random_state = 1
)

# create UtilityFunction object for aqu. function
utility = UtilityFunction(kind = "ei", xi= 0.02)

# set gaussian process parameter
optimizer.set_gp_params(alpha = 1e-6)

# create logger 
logger = JSONLogger(path = "./tunning_qlearn.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# initial search 
optimizer.maximize(
    init_points = 5, # number of random explorations before bayes_opt
    n_iter = 10, # number of bayes_opt iterations
)

# print out the data from the initial run to check if bounds need update 
for i, param in enumerate(optimizer.res):
    print(f"Iteration {i}: \n\t {param}")

# get best parameter
print("Best Parameters found: ")
print(optimizer.max)
