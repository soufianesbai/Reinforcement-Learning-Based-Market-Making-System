"""
A2C training on Historical Binance Kline Data with Technical Indicators.
run with: python3 -m agents.train_a2c
sources:
https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
https://huggingface.co/learn/deep-rl-course/unit6/advantage-actor-critic
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import time

from agents.strategies.a2c_rl import ActorNN, CriticNN


import random
import numpy as np
from collections import deque

class ToyMarketEnv:
    max_inv = 20.0
    qty = 1.0

    def __init__(self, max_steps: int = 200, sma_window: int = 10):
        self.max_steps = max_steps
        self.sma_window = sma_window

    def reset(self):
        self.step_count = 0
        self.price = 100.0
        self.inventory = 0.0
        self.cash = 0.0
        self.last_price = 100.0
        
        self.price_history = deque([self.price] * self.sma_window, maxlen=self.sma_window)
        
        return self._obs()

    def get_sma(self):
        return sum(self.price_history) / len(self.price_history)

    def _obs(self):
        imb = 1 
        
        spread_bps = 2.0 / 10.0 
        
        current_sma = self.get_sma()
        deviation = (self.price - current_sma) / current_sma * 100.0 / 2.0
        
        norm_inv = self.inventory / self.max_inv
        
        return np.array([imb, spread_bps, deviation, norm_inv], dtype=np.float32)

    def step(self, action: int):
        self.step_count += 1
        self.last_price = self.price
        
        self.price += 1
        
        self.price_history.append(self.price)

        spread = self.price * 0.0002 
        
        if action == 1 and self.inventory < self.max_inv:
            self.cash -= (self.price + spread) * self.qty
            self.inventory += self.qty
        elif action == 2 and self.inventory > -self.max_inv:
            self.cash += (self.price - spread) * self.qty
            self.inventory -= self.qty

        #PnL
        mtm = self.inventory * (self.price - self.last_price)
        
        #penalty = 0.005 * abs(self.inventory)
        penalty = 0.0
        
        reward = mtm - penalty

        done = self.step_count >= self.max_steps
        return self._obs(), reward, done


def train_critic(lr_actor=1e-2, lr_critic=1e-2, gamma=0.99, n_episodes=500, MAX_GRAD_NORM=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[A2C Q-Critic] Training on {device}")

    start_time = time.time()

    env = ToyMarketEnv(max_steps=200)
    actor  = ActorNN(state_dim=4, action_dim=3).to(device)
    critic = CriticNN(state_dim=4, action_dim=3).to(device)

    actor_optim  = optim.Adam(actor.parameters(),  lr=lr_actor)
    critic_optim = optim.Adam(critic.parameters(), lr=lr_critic)

    episode_rewards = deque(maxlen=50)

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        episode_reward = 0

        states, actions, rewards, next_states, log_probs = [], [], [], [], []

        while not done:
            state_t = torch.tensor(obs).unsqueeze(0).to(device)
                        
            action, log_prob = actor.get_action(state_t)
            action_idx = action.item()

            next_obs, reward, done = env.step(action_idx)
            
            reward = reward / 100.0 
            
            episode_reward += reward

            states.append(obs.copy())
            actions.append(action_idx)
            rewards.append(reward)
            next_states.append(next_obs.copy())
            log_probs.append(log_prob)

            obs = next_obs

        episode_rewards.append(episode_reward)

        #max Q(s_T, a') over all actions for the last state (needed for returns)
        with torch.no_grad():
            last_state_t = torch.tensor(next_states[-1]).unsqueeze(0).to(device)
            q_next_max = float('-inf')
            for a in range(3):
                a_idx_t = torch.tensor([a], device=device)
                q_next, _ = critic(last_state_t, a_idx_t)
                q_next_max = max(q_next_max, q_next.item())

        returns = []
        R = q_next_max
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        returns_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(device)

        q_sa, v_s = critic(states_t, actions_t)
        
        q_loss = F.mse_loss(q_sa, returns_t)
        v_loss = F.mse_loss(v_s, returns_t)
        critic_loss = q_loss + v_loss

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
        critic_optim.step()

        log_probs_t = torch.stack(log_probs).to(device)
        
        with torch.no_grad():
            q_sa_detach, v_s_detach = critic(states_t, actions_t)
            advantages = (q_sa_detach - v_s_detach).squeeze()

        actor_loss = -(log_probs_t * advantages).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
        actor_optim.step()

        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards)
            elapsed = time.time() - start_time
            print(f"Ep {ep:4d} | Avg Reward: {avg_reward:+.3f} | "
                  f"Critic Loss: {critic_loss.item():.4f} | Actor Loss: {actor_loss.item():.4f} | "
                  f"Elapsed: {elapsed:.1f}s")
            
            n_hold = sum(a == 0 for a in actions)
            n_buy  = sum(a == 1 for a in actions)
            n_sell = sum(a == 2 for a in actions)

            print(
                f"Ep {ep:4d} | HOLD: {n_hold:3d} | BUY: {n_buy:3d} | SELL: {n_sell:3d}"
            )

    os.makedirs("models", exist_ok=True)
    torch.save(actor.state_dict(),  "models/a2c_actor.pth")
    torch.save(critic.state_dict(), "models/a2c_critic.pth")
    elapsed = time.time() - start_time
    print(f"[A2C] Models saved. Total training time: {elapsed:.1f}s")


if __name__ == "__main__":
    train_critic()
