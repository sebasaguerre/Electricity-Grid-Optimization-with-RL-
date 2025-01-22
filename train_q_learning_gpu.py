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
parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
parser.add_argument('--save_q', type=str, default='qtable.pth', help='Filename to save learned Q-table')
args = parser.parse_args()

# Training parameters
alpha = 0.1 # learning rate
gamma = 0.99
start_epsilon = 1.0
end_epsilon = 0.05
episodes = args.episodes

def main_train_qlearn(alpha, gamma, episodes, device, start_epsilon, end_epsilon):

    # Q-table dimensions
    dummy_env = DataCenterEnv(args.train_excel)
    max_days = len(dummy_env.price_values)
    Q_shape = (22, 15, 24, max_days, 3)

    # Initialize Q-table on the GPU
    Q = torch.zeros(Q_shape, dtype=torch.float32, device=device)


    # History for tracking performance
    episode_rewards = []

    for ep in range(episodes):
        print(f"Episode {ep + 1}/{episodes}")
        env = DataCenterEnv(args.train_excel)
        state = env.reset()
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
            torch.save(Q, f"qtable_ep_{ep + 1}.pth")

    # Save final Q-table and rewards
    torch.save(Q, args.save_q)
    np.save('episode_rewards.npy', np.array(episode_rewards))
    print(f"Q-table saved to {args.save_q}")
    print("Training completed!")
    return total_reward, episode_rewards

if __name__ == "__main__":
    main()
