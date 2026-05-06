"""
A2C training 

run with: python3 -m agents.train_a2c

sources:
https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
https://huggingface.co/learn/deep-rl-course/unit6/advantage-actor-critic
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import time

from agents.strategies.a2c_rl import ActorNN, CriticNN

import gymnasium as gym
from gymnasium import spaces


def build_volume_bars(csv_path, volume_threshold=1000):
    print(f"Lecture du fichier : {csv_path}...")
    df = pd.read_csv(csv_path, header=None,
                     names=['id', 'price', 'qty', 'f', 'l', 'timestamp', 'm', 'M'],
                     low_memory=False)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['qty']   = pd.to_numeric(df['qty'],   errors='coerce')
    df = df.dropna(subset=['price', 'qty'])
    if len(df) == 0:
        raise ValueError("ERREUR : Le CSV est vide")

    print(f"Création des Volume Bars (Seuil: {volume_threshold} SOL)...")
    df['cum_vol'] = df['qty'].cumsum()
    df['bar_id']  = df['cum_vol'] // volume_threshold

    ohlcv = df.groupby('bar_id').agg(
        open   =('price', 'first'),
        high   =('price', 'max'),
        low    =('price', 'min'),
        close  =('price', 'last'),
        volume =('qty',   'sum'),
    )
    if ohlcv.iloc[-1]['volume'] < (volume_threshold * 0.9):
        ohlcv = ohlcv.iloc[:-1]

    print(f"Succès : {len(ohlcv)} bougies de volume générées !")
    return ohlcv


class ContinuousBinanceEnv(gym.Env):
    def __init__(self, csv_path, window_size=30, volume_threshold=1000):
        super().__init__()
        ohlcv = build_volume_bars(csv_path, volume_threshold)
        self.closes  = ohlcv['close'].values.astype(np.float32)
        self.volumes = ohlcv['volume'].values.astype(np.float32)
        self.highs   = ohlcv['high'].values.astype(np.float32)
        self.lows    = ohlcv['low'].values.astype(np.float32)

        self.window_size     = window_size
        self.max_inventory   = 10
        self.max_spread_pct  = 0.005

        total_candles = len(self.closes)
        self.max_steps = max(10, total_candles - window_size - 5) if total_candles < 250 else 200

        n_features = window_size * 4 + 1
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_this_episode = 0
        max_start = len(self.closes) - self.max_steps - 2
        if max_start <= self.window_size:
            self.current_step = self.window_size
        else:
            self.current_step = np.random.randint(self.window_size, max_start)
        self.inventory         = 0
        self.cash              = 0.0
        self.old_portfolio_value = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        i, w = self.current_step, self.window_size
        close_w  = self.closes[i-w:i]
        volume_w = self.volumes[i-w:i]
        high_w   = self.highs[i-w:i]
        low_w    = self.lows[i-w:i]

        base        = close_w[-1] + 1e-8
        norm_close  = (close_w  / base) - 1.0
        norm_high   = (high_w   / base) - 1.0
        norm_low    = (low_w    / base) - 1.0
        vol_mean    = volume_w.mean() + 1e-8
        norm_volume = (volume_w / vol_mean) - 1.0
        norm_inv    = self.inventory / self.max_inventory

        return np.concatenate([norm_close, norm_high, norm_low, norm_volume, [norm_inv]]).astype(np.float32)

    def step(self, action):
        current_price = self.closes[self.current_step]
        action = np.asarray(action, dtype=np.float32)

        ask_distance_pct = (action[0] + 1.0) / 2.0 * self.max_spread_pct
        bid_distance_pct = (action[1] + 1.0) / 2.0 * self.max_spread_pct
        agent_ask_price  = current_price * (1.0 + ask_distance_pct)
        agent_bid_price  = current_price * (1.0 - bid_distance_pct)

        self.current_step += 1
        next_high  = self.highs[self.current_step]
        next_low   = self.lows[self.current_step]
        next_close = self.closes[self.current_step]

        if next_low  <= agent_bid_price and self.inventory <  self.max_inventory:
            self.inventory += 1
            self.cash -= agent_bid_price
        if next_high >= agent_ask_price and self.inventory > -self.max_inventory:
            self.inventory -= 1
            self.cash += agent_ask_price

        current_portfolio_value = self.cash + (self.inventory * next_close)
        step_pnl          = current_portfolio_value - self.old_portfolio_value
        inventory_penalty = 0.0001 * (self.inventory ** 2) * next_close
        reward            = np.clip(step_pnl - inventory_penalty, -10.0, 10.0)

        self.old_portfolio_value  = current_portfolio_value
        self.steps_this_episode  += 1
        terminated = self.steps_this_episode >= self.max_steps

        return self._get_obs(), float(reward), terminated, False, {}


def train_a2c(lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, n_episodes=2000, MAX_GRAD_NORM=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[A2C] Training on {device}")

    episode_rewards_history = []
    episode_actions_history = []
    critic_losses_history   = []
    actor_losses_history    = []

    start_time = time.time()

    csv_file = "./data/SOLUSDT-aggTrades-2026-02.csv"
    env = ContinuousBinanceEnv(csv_file, window_size=30, volume_threshold=1000)

    state_dim = env.observation_space.shape[0]   # 121

    actor  = ActorNN(state_dim=state_dim, action_dim=3).to(device)
    critic = CriticNN(state_dim=state_dim).to(device)

    actor_optim  = optim.Adam(actor.parameters(),  lr=lr_actor)
    critic_optim = optim.Adam(critic.parameters(), lr=lr_critic)

    episode_rewards = deque(maxlen=50)
    episode_hold_history = []
    episode_buy_history = []
    episode_sell_history = []

    for ep in range(1, n_episodes + 1):
        
        obs, _ = env.reset()
        done   = False
        episode_reward = 0

        states, actions, rewards, next_states, log_probs, entropies = [], [], [], [], [], []

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            action, log_prob, entropy = actor.get_action(state_t)
            entropies.append(entropy)
            action_idx = action.item()

            if action_idx == 0:
                continuous_action = np.array([-0.5, -0.5])
            elif action_idx == 1:
                continuous_action = np.array([-0.8,  0.2])
            else:
                continuous_action = np.array([ 0.2, -0.8])

            next_obs, reward, terminated, truncated, _ = env.step(continuous_action)
            done = terminated or truncated

            episode_reward += reward
            states.append(obs.copy())
            actions.append(action_idx)
            rewards.append(reward)
            next_states.append(next_obs.copy())
            log_probs.append(log_prob)

            obs = next_obs

        episode_rewards.append(episode_reward)

        #  compute  returns 
        # R_t = r_t + γ * V(s_{t+1})
        with torch.no_grad():
            last_next_state_t = torch.tensor(next_states[-1], dtype=torch.float32).unsqueeze(0).to(device)
            v_last = critic(last_next_state_t).item()   

        returns = []
        R = v_last
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        states_t  = torch.tensor(np.array(states),  dtype=torch.float32).to(device)
        returns_t = torch.tensor(returns,            dtype=torch.float32).unsqueeze(1).to(device)

        v_s = critic(states_t)                       
        critic_loss = F.mse_loss(v_s, returns_t)

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
        critic_optim.step()

        #  actor update 
        #   (advantage = return - V(s_t))
        log_probs_t = torch.stack(log_probs).to(device)
        entropies_t = torch.stack(entropies).to(device)

        with torch.no_grad():
            v_s_detach = critic(states_t)            
        advantages = (returns_t - v_s_detach).squeeze()   # A_t = R_t - V(s_t)

        entropy_coef = max(0.0005, 0.002 * (1 - ep / n_episodes))

        # actor loss: -log pi(a|s) * A_t  - entropy_coef * entropy
        actor_loss = (-log_probs_t * advantages - entropy_coef * entropies_t).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
        actor_optim.step()

        #  metrics
        n_hold = sum(a == 0 for a in actions)
        n_buy  = sum(a == 1 for a in actions)
        n_sell = sum(a == 2 for a in actions)

        episode_hold_history.append(n_hold)
        episode_buy_history.append(n_buy)
        episode_sell_history.append(n_sell)

        episode_rewards_history.append(episode_reward)
        critic_losses_history.append(critic_loss.item())
        actor_losses_history.append(actor_loss.item())
        episode_actions_history.append({'hold': n_hold, 'buy': n_buy, 'sell': n_sell})

        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards)
            elapsed    = time.time() - start_time
            print(f"Ep {ep:4d} | Avg Reward: {avg_reward:+.3f} | "
                  f"Critic Loss: {critic_loss.item():.4f} | Actor Loss: {actor_loss.item():.4f} | "
                  f"Elapsed: {elapsed:.1f}s")
            print(f"Ep {ep:4d} | HOLD: {n_hold:3d} | BUY: {n_buy:3d} | SELL: {n_sell:3d}")

            window    = min(50, len(episode_rewards_history))
            reward_ma = np.convolve(episode_rewards_history, np.ones(window)/window, mode='valid')
            critic_ma = np.convolve(critic_losses_history,   np.ones(window)/window, mode='valid')
            actor_ma  = np.convolve(actor_losses_history,    np.ones(window)/window, mode='valid')

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            ax1.plot(episode_rewards_history, alpha=0.3, color='blue', label='Raw')
            ax1.plot(range(window-1, len(episode_rewards_history)), reward_ma,
                     color='blue', linewidth=3, label='Moving Avg')
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward')
            ax1.legend(); ax1.grid(True, alpha=0.3)

            ax2.plot(critic_losses_history, alpha=0.3, color='red')
            ax2.plot(range(window-1, len(critic_losses_history)), critic_ma,
                     color='red', linewidth=3)
            ax2.set_title('Critic Loss  L(φ) = MSE(R_t , V(s_t))')
            ax2.set_xlabel('Episode'); ax2.set_ylabel('MSE Loss')
            ax2.set_yscale('log'); ax2.grid(True, alpha=0.3)

            ax3.plot(actor_losses_history, alpha=0.3, color='green')
            ax3.plot(range(window-1, len(actor_losses_history)), actor_ma,
                     color='green', linewidth=3)
            ax3.set_title('Actor Loss  −log π(a|s) · A_t')
            ax3.set_xlabel('Episode'); ax3.set_ylabel('Policy Loss')
            ax3.grid(True, alpha=0.3)

            
            ax4.plot(episode_hold_history, color='steelblue', label='HOLD')
            ax4.plot(episode_buy_history, color='seagreen', label='BUY')
            ax4.plot(episode_sell_history, color='tomato', label='SELL')
            ax4.set_title('Actions par épisode')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Nombre d\'actions')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.suptitle(f'A2C Training Progress (Episode {ep})', fontsize=16)
            plt.tight_layout()
            plt.savefig('a2c_training_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()

    os.makedirs("models", exist_ok=True)
    torch.save(actor.state_dict(),  "models/a2c_actor.pth")
    torch.save(critic.state_dict(), "models/a2c_critic.pth")
    elapsed = time.time() - start_time
    print(f"[A2C] Models saved. Total training time: {elapsed:.1f}s")


if __name__ == "__main__":
    train_a2c()
