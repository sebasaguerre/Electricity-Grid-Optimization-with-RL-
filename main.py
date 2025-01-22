from env import DataCenterEnv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import gym
from gym import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args.add_argument('--heuristic', type=str, default='qtable')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path
heuristic = args.heuristic

environment = DataCenterEnv(path_to_dataset)

aggregate_reward = 0
# terminated = False
# state = environment.observation()
rewards = []  
epsilon = 0.25
lr = 0.95
discount = 0.99
num_episodes = 500

price_ranges = np.arange(5, 106, 5)
print("price ranges: ", price_ranges)
observation_space = spaces.Discrete(len(price_ranges))
action_space = spaces.Discrete(3)

Q_table = np.zeros([environment.observation_space.n, environment.action_space.n])
# print("Observation space:", observation_space)
# print("Observation space n:", observation_space.n)
# print("Action space:", action_space)
# Q_table = np.zeros([observation_space.n, action_space.n])
# Q_table = torch.zeros((environment.observation_space.n, environment.action_space.n), device=device)


for i in range(num_episodes):
    if i < 10:
        epsilon = 0.8
    elif i < 100:
        epsilon = 0.3
    elif i < 200:
        epsilon = 0.2
    elif i < 300:
        epsilon = 0.1
    else:
        epsilon = 0.02
    terminated = False
    state = environment.reset()
    print("Episode:", i)
    # print("State:", state)
    while not terminated:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.uniform(-1, 1)
        else:
            action = np.argmax(Q_table[int((state[0])/10)])
        next_state, reward, terminated = environment.step(action)
        Q_table[int(state[0]/10), int(action)] = Q_table[int(state[0]/10), int(action)] + lr * (reward + discount * np.max(Q_table[int((next_state[0])/10)]) - Q_table[int(state[0]/10), int(action)])
        state = next_state
        # print("Reward:", reward)
        aggregate_reward += reward
    print("Total reward:", aggregate_reward)
    rewards.append(aggregate_reward)
    reward = 0
    aggregate_reward = 0


test_random_reward = []
for j in range(num_episodes):
    state = environment.reset()
    terminated = False
    aggregate_reward_random = 0
    while not terminated:
        action = np.random.uniform(-1, 1)
        next_state, reward, terminated = environment.step(action)
        aggregate_reward_random += reward
    test_random_reward.append(aggregate_reward_random)
    aggregate_reward_random = 0
# while not terminated:
#     # agent is your own imported agent class
#     # action = agent.act(state)
#     if heuristic == "random":
#         action = np.random.uniform(-1, 1)
#     elif heuristic == "qtable":
#         if np.random.uniform(0, 1) < epsilon:
#             action = np.random.uniform(-1, 1)
#         else:
#             action = np.argmax(environment.Q_table[state])
#     # next_state is given as: [storage_level, price, hour, day]
#     next_state, reward, terminated = environment.step(action)
#     state = next_state
#     rewards.append(reward)
#     aggregate_reward += reward
#     print("Action:", action)
#     print("Next state:", next_state)
#     print("Reward:", reward)


# print('Total reward:', aggregate_reward)

# Plot the rewards over time
plt.figure(figsize=(10, 6))
plt.plot(rewards, label="Q table Reward over episodes", color="blue", linewidth=2)
plt.plot(test_random_reward, label="Random Reward over episodes", color="green", linewidth=2)
plt.title(f"Reward over episodes", fontsize=16)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()