# python train_q_learning.py --train_excel train.xlsx --episodes 500 --save_q qtable.npy

import numpy as np
import argparse

from env import DataCenterEnv

###########################################
# Discretization Utilities
###########################################
def discretize_price(price):
    """
    Discretize the price into 15 buckets:
      0..5 => index 0
      5..10 => index 1
      ...
      65..70 => index 13
      >= 70 => index 14
    """
    idx = int(price // 5)
    if idx < 0:
        idx = 0
    if idx > 14:
        idx = 14
    return idx

def discretize_storage(storage):
    """
    Discretize storage into 22 buckets of size 10
    """
    idx = int(storage // 10)
    if idx < 0:
        idx = 0
    if idx > 21:
        idx = 21
    return idx

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_excel', type=str, default='train.xlsx',
                        help='Excel file for training environment')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--save_q', type=str, default='qtable.npy',
                        help='Filename to save learned Q-table')
    args = parser.parse_args()
    
    # 1) Discover shape
    dummy_env = DataCenterEnv(args.train_excel)
    max_days = len(dummy_env.price_values)
    
    # 2) Q-table dims
    Q_shape = (22, 15, 24, max_days, 3)
    Q = np.zeros(Q_shape, dtype=np.float32)
    
    # 记录学习历史
    history = {
        'episode_rewards': [],  # 每个episode的总奖励
        'q_tables': []         # 每100个episode的Q表
    }
    
    # 3) Hyperparams
    alpha = 0.1
    gamma = 0.99
    start_epsilon = 1.0
    end_epsilon = 0.05
    episodes = args.episodes
    
    for ep in range(episodes):
        env = DataCenterEnv(args.train_excel)
        state = env.reset()
        s_idx, p_idx, h_idx, d_idx = get_state_indices(*state)
        
        frac = ep / float(episodes)
        epsilon = start_epsilon + frac * (end_epsilon - start_epsilon)
        
        done = False
        episode_reward = 0  # 记录当前episode的总奖励
        
        while not done:
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(3)
            else:
                a_idx = np.argmax(Q[s_idx, p_idx, h_idx, d_idx, :])
            action = idx_to_action(a_idx)
            
            next_state, reward, done = env.step(action)
            episode_reward += reward  # 累加奖励
            
            s2_idx, p2_idx, h2_idx, d2_idx = get_state_indices(*next_state)
            
            best_next = np.max(Q[s2_idx, p2_idx, h2_idx, d2_idx, :])
            old_val = Q[s_idx, p_idx, h_idx, d_idx, a_idx]
            Q[s_idx, p_idx, h_idx, d_idx, a_idx] = old_val + alpha*(reward + gamma*best_next - old_val)
            
            s_idx, p_idx, h_idx, d_idx = s2_idx, p2_idx, h2_idx, d2_idx
        
        # 记录每个episode的总奖励
        history['episode_rewards'].append(episode_reward)
        
        # 每100个episode保存一次Q表
        if (ep + 1) % 100 == 0:
            history['q_tables'].append(Q.copy())
            print(f"Episode {ep+1}/{episodes} done. Total reward: {episode_reward:.2f}")
            # 保存中间结果
            np.save(f"q_table_ep_{ep+1}.npy", Q)
            
        elif (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} done. Total reward: {episode_reward:.2f}")
    
    # 保存最终的Q表和学习历史
    np.save(args.save_q, Q)
    np.save('training_history.npy', history)
    print(f"Q-table saved to {args.save_q}")
    print(f"Training history saved to training_history.npy")

if __name__ == "__main__":
    main()
