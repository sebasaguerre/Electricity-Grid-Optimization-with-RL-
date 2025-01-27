import torch
import numpy as np
import argparse
from env import DataCenterEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Discretization Utilities
def discretize_price(price):
    idx = int(price // 5)
    return min(max(idx, 0), 14)

def discretize_storage(storage):
    idx = int(storage // 10)
    return min(max(idx, 0), 21)

def get_state_indices(storage, price, hour, day):
    s_idx = discretize_storage(storage)
    p_idx = discretize_price(price)
    h_idx = int(hour) - 1
    d_idx = int(day) - 1
    return (s_idx, p_idx, h_idx, d_idx)

def idx_to_action(action_idx):
    if action_idx == 0:
        return -1.0
    elif action_idx == 1:
        return 0.0
    elif action_idx == 2:
        return 1.0
    else:
        raise ValueError("Invalid action_idx")

parser = argparse.ArgumentParser()
parser.add_argument('--train_excel', type=str, default='train.xlsx', help='Excel file for training environment')
parser.add_argument('--val_excel', type=str, default='validate.xlsx', help='Excel file for validation environment')
parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
parser.add_argument('--save_q', type=str, default='qtable.pth', help='Filename to save learned Q-table')
parser.add_argument('--train', type=bool, default=False, help='Train the Q-learning agent')
parser.add_argument('--validate', type=bool, default=True, help='Validate the Q-learning agent')
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

def train_qlearn(alpha, gamma, episodes, device, start_epsilon, end_epsilon):
    print("Starting training...")
    # Q-table dimensions
    dummy_env = DataCenterEnv(args.train_excel)
    max_days = len(dummy_env.price_values)
    Q_shape = (22, 15, 24, max_days, 3)

    # Initialize Q-table on the GPU
    Q = torch.zeros(Q_shape, dtype=torch.float16, device=device)

    # History for tracking performance
    episode_rewards = []

    for ep in range(episodes):
        print(f"Episode {ep + 1}/{episodes}")
        env = DataCenterEnv(args.train_excel)
        state = env.reset() # state = [storage, price, hour, day]
        s_idx, p_idx, h_idx, d_idx = get_state_indices(*state)

        epsilon = start_epsilon + ep / episodes * (end_epsilon - start_epsilon)
        done = False
        total_reward = 0

        while not done:
            # Select action (epsilon-greedy)
            if torch.rand(1).item() < epsilon:
                a_idx = torch.randint(0, 3, (1,), device=device).item()
            else:
                a_idx = torch.argmax(Q[s_idx, p_idx, h_idx, d_idx]).item()

            action = idx_to_action(a_idx)
            next_state, reward, done = env.step(action)
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
            
            

            total_reward += reward
            s2_idx, p2_idx, h2_idx, d2_idx = get_state_indices(*next_state)

            # Q-learning update
            best_next = torch.max(Q[s2_idx, p2_idx, h2_idx, d2_idx]).item()
            current_q = Q[s_idx, p_idx, h_idx, d_idx, a_idx].item()
            Q[s_idx, p_idx, h_idx, d_idx, a_idx] = torch.tensor(
                current_q + alpha * (reward + gamma * best_next - current_q),
                device=device
            )

            # Move to the next state
            s_idx, p_idx, h_idx, d_idx = s2_idx, p2_idx, h2_idx, d2_idx

        episode_rewards.append(total_reward)

        # Log progress every 50 episodes
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{episodes}, Total Reward: {total_reward:.2f}")

        # Save intermediate Q-table every 100 episodes
        if (ep + 1) % 100 == 0:
            torch.save(Q, f"qtable_ep_{ep + 1}_rwd_shaped_{scaling_factor}.pth")

    # Save final Q-table and rewards
    torch.save(Q, args.save_q)
    np.save('training_rewards.npy', np.array(episode_rewards))
    print(f"Q-table saved to {args.save_q}")
    print("Training completed!")
    return Q, episode_rewards

def validate_qlearn(Q, env_path):
    print("Starting validation...")
    env = DataCenterEnv(env_path)
    state = env.reset()
    s_idx, p_idx, h_idx, d_idx = get_state_indices(*state)

    total_reward = 0
    done = False

    while not done:
        # Select the best action (greedy policy)
        a_idx = torch.argmax(Q[s_idx, p_idx, h_idx, d_idx]).item()
        action = idx_to_action(a_idx)
        next_state, reward, done = env.step(action)
        total_reward += reward

        s_idx, p_idx, h_idx, d_idx = get_state_indices(*next_state)

    print(f"Validation completed! Total Reward: {total_reward:.2f}")
    return total_reward

def main():
    # Train the Q-learning agent
    if training:
        Q, train_rewards = train_qlearn(alpha, gamma, episodes, device, start_epsilon, end_epsilon)

    if validate:
        path_to_qtable = "intermediate_qtables/qtable_ep_1000_rwd_shaped_0.5.pth"
        Q = torch.load(path_to_qtable)
        Q = Q.to(device)
        Q.requires_grad = False
        # Validate the agent
        val_reward = validate_qlearn(Q, args.val_excel)

        # Save training rewards and validation results
        np.save('validation_reward.npy', np.array(val_reward))
        print("Validation results saved!")

if __name__ == "__main__":
    main()
