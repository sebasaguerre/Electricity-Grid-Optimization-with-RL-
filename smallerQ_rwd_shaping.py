import torch
import numpy as np
import argparse
from env import DataCenterEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Discretization Utilities
def discretize_price(price):
    idx = int(price // 20)
    return min(max(idx, 0), 5)

def discretize_storage(storage):
    idx = int(storage // 10)
    return min(max(idx, 0), 21)

def get_state_indices(price, hour):
    p_idx = discretize_price(price)
    if int(hour) < 5:
        h_idx = 0
    elif int(hour) < 10:
        h_idx = 1
    elif int(hour) < 15:
        h_idx = 2
    elif int(hour) < 20:
        h_idx = 3
    else:
        h_idx = 4
    # h_idx = int(hour) - 1
    # d_idx = int(day) - 1
    # return (s_idx, p_idx, h_idx, d_idx)
    return (p_idx, h_idx)

def idx_to_action(action_idx):
    if action_idx == 0:
        return -1.0
    elif action_idx == 1:
        return 0.0
    elif action_idx == 2:
        return 1.0
    else:
        # print("action_idx: ", action_idx)
        raise ValueError("Invalid action_idx")

parser = argparse.ArgumentParser()
parser.add_argument('--train_excel', type=str, default='train.xlsx', help='Excel file for training environment')
parser.add_argument('--val_excel', type=str, default='validate.xlsx', help='Excel file for validation environment')
parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
parser.add_argument('--save_q', type=str, default='small_qtable_noday_nostorage.pth', help='Filename to save learned Q-table')
parser.add_argument('--train', type=bool, default=False, help='Train the Q-learning agent')
parser.add_argument('--validate', type=bool, default=False, help='Validate the Q-learning agent')
args = parser.parse_args()

# Training parameters
alpha = 0.1 # learning rate
gamma = 0.99
start_epsilon = 1.0
end_epsilon = 0.05
episodes = args.episodes
scaling_factor = 0.5
training = args.train
validate = args.validate
remark_save_intermediate_qtable = "small_noday_nostorage"
reward_file_name = "training_rewards_smallq_noday.npy"

def risk_factor(hour, storage):
    min_storage_req = 120
    max_buy_per_hour = 10
    remaining_hours = 24 - hour
    remaining_storage_needed = max(min_storage_req - storage, 0)  # Ensure it's non-negative
    
    # If storage is already sufficient, risk is 0
    if remaining_storage_needed == 0:
        return 0.0

    # Maximum possible storage that can be acquired in the remaining time
    max_possible_storage = remaining_hours * max_buy_per_hour

    # If it's impossible to meet the requirement, assign very high risk
    if max_possible_storage < remaining_storage_needed:
        return 1.0  # Maximum risk

    # Risk increases as the day progresses and remaining storage needed is high
    risk = remaining_storage_needed / max_possible_storage
    return round(risk, 3)  # Rounded for clarity


def train_qlearn(alpha, gamma, episodes, device, start_epsilon, end_epsilon):
    print("Starting training...")
    # Q-table dimensions
    dummy_env = DataCenterEnv(args.train_excel)
    max_days = len(dummy_env.price_values)
    # Q_shape = (22, 6, 5, max_days, 3)
    # Q_shape = (22, 6, 5, 3) # storage, price, hour, action
    Q_shape = (6, 5, 3) # storage, price, hour, action [remove the storage bins]

    # Initialize Q-table on the GPU
    Q = torch.zeros(Q_shape, dtype=torch.float16, device=device)

    # History for tracking performance
    episode_rewards = []

    for ep in range(episodes):
        print(f"Episode {ep + 1}/{episodes}")
        env = DataCenterEnv(args.train_excel)
        state = env.reset() # state = [storage, price, hour, day]
        # s_idx, p_idx, h_idx, d_idx = get_state_indices(*state)
        p_idx, h_idx = get_state_indices(state[1], state[2])

        epsilon = start_epsilon + ep / episodes * (end_epsilon - start_epsilon)
        done = False
        total_reward = 0

        while not done:
            # Select action (epsilon-greedy)
            if torch.rand(1).item() < epsilon:
                a_idx = torch.randint(0, 3, (1,), device=device).item()
            else:
                # a_idx = torch.argmax(Q[s_idx, p_idx, h_idx, d_idx]).item()
                a_idx = torch.argmax(Q[p_idx, h_idx]).item()

            action = idx_to_action(a_idx)
            next_state, reward, done = env.step(action)
            risk_factor_val = risk_factor(next_state[2], next_state[0])
            if risk_factor_val > 0.75:
                # reward -= scaling_factor*(10 * risk_factor_val)
                print(f"Risk factor: {risk_factor_val}")
            if action == 1.0: # electricity bought 
                if state[0] < 120:
                    reward += scaling_factor*(10 * state[1])
                else:
                    reward -= scaling_factor*(10 * state[1])
                # print(f"Action: {action}, Reward: {reward}")
            elif action == -1.0: # electricity sold
                if state[0] < 120:
                    reward -= scaling_factor*(10 * state[1])
                else:
                    reward += scaling_factor*(10 * state[1])
                # print(f"Action: {action}, Reward: {reward}")
            # print("Reward shaping done, rwd: ", reward)

            #risk_ass(hour, storage):
                #return risk_level (float: 1-1.3)
            # dynamic scaling factor depending on the risk
            # TODO: CHECK Q TABLE IN VALIDATION IF WE ENCOUNTER ANY ZEROS!
            # DONE
            # NUMBER OF PRICE BINS -> 5-7
            # EVEN FOR HOUR - MORNING, EVENING, AFTERNOON MAYBE
            # 400-2000 STATES

            total_reward += reward
            # s2_idx, p2_idx, h2_idx, d2_idx = get_state_indices(*next_state)
            # s2_idx, p2_idx, h2_idx = get_state_indices(*next_state)
            p2_idx, h2_idx = get_state_indices(next_state[1], next_state[2])

            # Q-learning update
            # print("starting q update")
            # best_next = torch.max(Q[s2_idx, p2_idx, h2_idx, d2_idx]).item()
            best_next = torch.max(Q[p2_idx, h2_idx]).item()
            # current_q = Q[s_idx, p_idx, h_idx, d_idx, a_idx].item()
            current_q = Q[p_idx, h_idx, a_idx].item()
            # Q[s_idx, p_idx, h_idx, d_idx, a_idx] = torch.tensor(
            #     current_q + alpha * (reward + gamma * best_next - current_q),
            #     device=device
            # )
            Q[p_idx, h_idx, a_idx] = torch.tensor(
                current_q + alpha * (reward + gamma * best_next - current_q),
                device=device
            )
            # print("Q update done")
            # Move to the next state
            # s_idx, p_idx, h_idx, d_idx = s2_idx, p2_idx, h2_idx, d2_idx
            p_idx, h_idx = p2_idx, h2_idx

        episode_rewards.append(total_reward)

        # Log progress every 50 episodes
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{episodes}, Total Reward: {total_reward:.2f}")

        # Save intermediate Q-table every 100 episodes
        if (ep + 1) % 100 == 0:
            torch.save(Q, f"qtable_ep_{ep + 1}_{remark_save_intermediate_qtable}.pth")

    # Save final Q-table and rewards
    torch.save(Q, args.save_q)
    np.save(reward_file_name, np.array(episode_rewards))
    print(f"Q-table saved to {args.save_q}")
    print("Training completed!")
    return Q, episode_rewards

def validate_qlearn(Q, env_path):
    print("Starting validation...")
    env = DataCenterEnv(env_path)
    state = env.reset()
    # s_idx, p_idx, h_idx, d_idx = get_state_indices(*state)
    p_idx, h_idx = get_state_indices(state[1], state[2])
    # print(f"Initial State: {state}")

    total_reward = 0
    done = False

    while not done:
        # Select the best action (greedy policy)
        # a_idx = torch.argmax(Q[s_idx, p_idx, h_idx, d_idx]).item()
        a_idx = torch.argmax(Q[p_idx, h_idx]).item()
        # print(f"Action from Q argmax: {a_idx}")
        action = idx_to_action(a_idx)
        next_state, reward, done = env.step(action)
        total_reward += reward

        # s_idx, p_idx, h_idx, d_idx = get_state_indices(*next_state)
        p_idx, h_idx = get_state_indices(next_state[1], next_state[2])

    print(f"Validation completed! Total Reward: {total_reward:.2f}")
    return total_reward

def visualize_qtable_sliced(Q, day_idx=0, hour_idx=3):
    import matplotlib.pyplot as plt

    # Example: Fix a specific day and hour
    # day_idx = 0  # First day
    # hour_idx = 12  # Midday
    # sliced_Q = Q[:, :, hour_idx, day_idx, :].cpu().numpy()  # Convert to numpy for visualization
    sliced_Q = Q[:, :, hour_idx, :].cpu().numpy()  # Convert to numpy for visualization

    # Visualize the Q-values for each action
    actions = ['-1.0', '0.0', '1.0']
    for action_idx in range(3):
        plt.figure(figsize=(8, 6))
        plt.imshow(sliced_Q[:, :, action_idx], cmap='viridis', aspect='auto')
        plt.colorbar(label="Q-value")
        plt.title(f"Q-values for Action {actions[action_idx]} (Day {day_idx+1}, Hour {hour_idx+1})")
        plt.xlabel("Price State")
        plt.ylabel("Storage State")
        plt.show()


def visualize_rewards(rewards):
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.show()

def states_visited(Q):
    Q_np = Q.cpu().numpy()
    Q_mean = np.mean(Q_np, axis=-1)
    Q_max = np.max(Q_mean, axis=-1)
    states_visited = np.sum(Q_max > 0, axis=0)
    return states_visited


def main():
    # Train the Q-learning agent
    if training:
        Q, train_rewards = train_qlearn(alpha, gamma, episodes, device, start_epsilon, end_epsilon)

    if validate:
        path_to_qtable = "small_qtable_noday_nostorage.pth"
        Q = torch.load(path_to_qtable)
        Q = Q.to(device)
        Q.requires_grad = False
        # Validate the agent
        val_reward = validate_qlearn(Q, args.val_excel)

        # Save training rewards and validation results
        np.save('validation_reward.npy', np.array(val_reward))
        print("Validation results saved!")
    
    # # Visualize Q-table
    # if not training:
    #     path_to_qtable = "rwd_shape_q_scale50.pth"
    #     Q = torch.load(path_to_qtable)
    #     Q_np = Q.cpu().numpy() #if numpy is needed
    
    # day_idx = 0  # First day
    # hour_idx = 3
    # visualize_qtable_sliced(Q, day_idx, hour_idx)

    # Visualize training rewards
    # visualize_rewards(train_rewards)  # Commented out for now
    load_train_rewards = np.load('training_rewards.npy')
    visualize_rewards(load_train_rewards)
if __name__ == "__main__":
    main()
