from env import DataCenterEnv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import gym
from gym import spaces

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args.add_argument('--heuristic', type=str, default='avg')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path
heuristic = args.heuristic

environment = DataCenterEnv(path_to_dataset)
gym.register(id="DataCenterEnv-v0", entry_point=lambda: environment)
env = gym.make("DataCenterEnv-v0")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Create a Q table
Q_table = np.zeros([env.observation_space.n, env.action_space.n])


aggregate_reward = 0
terminated = False
state = env.observation()
rewards = []  
threshold_low = 45 
threshold_high = 60
# (35,60)-Total reward: -6627177.819999966
# (35,65)-Total reward: -6607248.919999965
# (35,50)-Total reward: -6675524.519999966
# (20,45)-Total reward: -6901393.099999972
# (30,55)-Total reward: -6521075.419999962

buy_count = 0
sell_count = 0

while not terminated:
    # agent is your own imported agent class
    # action = agent.act(state)
    if heuristic == "random":
        action = np.random.uniform(-1, 1)
    elif heuristic == "avg": #buy at low, sell at high but limit of 17 for buy and 5 for sell
        current_price = state[1]
        if state[2] < 24:
            if (current_price < threshold_low or state[2] < 7) and buy_count < 17:
                action = 1.0
                buy_count += 1
            elif current_price > threshold_high and buy_count > 12 and sell_count < 5:
                action = -1.0
                sell_count += 1
            else:
                action = 0.0
        if state[2] >= 24:
            buy_count = 0
            sell_count = 0
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = env.step(action)
    state = next_state
    rewards.append(reward)
    aggregate_reward += reward
    print("Buy count:", buy_count)
    print("Sell count:", sell_count)
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)


print('Total reward:', aggregate_reward)

# Plot the rewards over time
plt.figure(figsize=(10, 6))
plt.plot(rewards, label="Reward over time", color="blue", linewidth=2)
plt.title(f"Reward over time (Total:{aggregate_reward})", fontsize=16)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()