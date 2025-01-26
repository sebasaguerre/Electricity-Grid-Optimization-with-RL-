import gym
from gym import spaces
import numpy as np
import pandas as pd
import argparse
from env import DataCenterEnv

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args.add_argument('--heuristic', type=str, default='avg')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path
environment = gym.register(id = "Data_Env0", entry_point = lambda: DataCenterEnv(path_to_dataset))
env = gym.make("Data_Env0")


class Qlearn():
    def __init__(self, discount_rate):
        self.env = gym.make("Data_Env0")
        self.discount_rate = discount_rate
        self.action_space = [-1, 0, 1]
        self.bin_size= 20
        self.bins_prices = np.linspace(0, 80, self.bin_size)
        self.bins_storage = np.linspace(0, 200, self.bin_size)
        self.bins = [self.bins_prices, self.bins_storage]

    def discretize_space(self, state):
        self.state = state
        discrete_state = []
        for i in range(len(self.bins)):
            discrete_state.append(np.digitize(self.state[i], self.bins[i]) - 1)

        return discrete_state
    
    def create_Q_table(self):

        self.state_space = self.bin_size - 1

        self.Qtable = np.zeros(len(self.bin_storage), len(self.bin_prices), 24, 365, self.action_space)
    
    def train(self, simulations, lr, epsilon = 0.05):

        # set up initial values
        self.lr = lr
        self.epsilon = epsilon 
        
        # loop over the number of simulations
        for i in range(simulations):

            # setup main  variables for training 
            aggregated_reward = 0
            terminated = False
            state = env.reset()
            state = self.discretize_state(state)
            rewards = []

            # main training loop
            while not terminated:
                # probability of choosing random action 
                prob = np.random.uniform()

                # check if we take random action
                if prob >= (1 - self.epsilon):
                    action = np.random(self.action_space)
                else:
                    action = np.argmax(self.Qtable[state[0], state[1], state[2], :])
                
                # take step and go to next state 
                next_state, reward, terminated = self.env.step(action)
                state = self.discretize_state(next_state)
                rewards.append(reward)
                aggregate_reward += reward




