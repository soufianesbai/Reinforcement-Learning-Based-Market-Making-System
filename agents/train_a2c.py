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
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import time

from agents.strategies.a2c_rl import ActorNN, CriticNN


from collections import deque

import gymnasium as gym
from gymnasium import spaces



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
        deviation = (self.price - current_sma) / (current_sma + 1e-8) * 50.0
        norm_inv = self.inventory / self.max_inv
        price_change = (self.price - self.last_price) / (self.last_price + 1e-8) * 100.0 
        return np.array([imb, spread_bps, deviation, norm_inv, price_change], dtype=np.float32)

    def step(self, action: int):
        self.step_count += 1
        self.last_price = self.price
        
        self.price += 1
        
        self.price_history.append(self.price)

        spread = self.price * 0.0002 
        
        if action == 1 and self.inventory < self.max_inv:
            self.cash -= (self.price + spread) * self.qty
            self.inventory += self.qty
        elif action == 2 and self.inventory > 0:
            self.cash += (self.price - spread) * self.qty
            self.inventory -= self.qty

        #PnL
        mtm = self.inventory * (self.price - self.last_price)
        
        #penalty = 0.005 * abs(self.inventory)
        penalty = 0.0
        
        #reward = mtm - penalty
        price_move = self.price - self.last_price
        opportunity_cost = -0.1 * price_move if self.inventory == 0 else 0.0
        reward = mtm - penalty + opportunity_cost

        done = self.step_count >= self.max_steps
        return self._obs(), reward, done

def build_volume_bars(csv_path, volume_threshold=1000):
    print(f"Lecture du fichier : {csv_path}...")

    df = pd.read_csv(csv_path, header=None,
                     names=['id', 'price', 'qty', 'f', 'l', 'timestamp', 'm', 'M'],
                     low_memory=False)
    
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')

    
    df = df.dropna(subset=['price', 'qty'])  # Supprime les lignes corrompues
    
    if len(df) == 0:
        raise ValueError("ERREUR : Le CSV est vide")

    print(f"Création des Volume Bars (Seuil: {volume_threshold} SOL)...")
    
    # Additionne les volumes trade par trade.
    df['cum_vol'] = df['qty'].cumsum()

    df['bar_id'] = df['cum_vol'] // volume_threshold
    
    ohlcv = df.groupby('bar_id').agg(
        open=('price', 'first'), # Premier prix
        high=('price', 'max'),  # Prix max
        low=('price', 'min'),  # Prix min
        close=('price', 'last'),  # Dernier prix 
        volume=('qty', 'sum')     # La somme des quantités échangées
    )
    
    # Vérifie la toute dernière bougie
    # La supprime si elle est incomplète (moins de 90%).
    if ohlcv.iloc[-1]['volume'] < (volume_threshold * 0.9):
        ohlcv = ohlcv.iloc[:-1]
        
    print(f"Succès : {len(ohlcv)} bougies de volume générées !")
    return ohlcv

class ContinuousBinanceEnv(gym.Env):
    def __init__(self, csv_path, window_size=30, volume_threshold=1000): 
        super().__init__()
        
        # Charge les bougies de volume
        ohlcv = build_volume_bars(csv_path, volume_threshold)
        self.closes = ohlcv['close'].values.astype(np.float32)
        self.volumes = ohlcv['volume'].values.astype(np.float32)
        self.highs = ohlcv['high'].values.astype(np.float32)
        self.lows = ohlcv['low'].values.astype(np.float32)

        self.window_size = window_size
        
        # LIMITES DU MARKET MAKER 
        self.max_inventory = 10 # L'agent ne peut pas détenir plus de 10 SOL
        self.max_spread_pct = 0.005 # Écartement maximum autorisé est 0.5%
        
        total_candles = len(self.closes)

        # Si le fichier de données est très petit
        if total_candles < 250: # On raccourcit la durée de la partie pour éviter de déborder à la fin du fichier.
            self.max_steps = max(10, total_candles - self.window_size - 5)
        
        else:  # Fixe la durée d'une partie d'entraînement (épisodes) à 200 tours 
            self.max_steps = 200 

        # Calcule la taille de l'observation pour l'agent 
        # window size (30 bougies en arriere)  × 4 infos (close, max, min, volume) + son inventaire actuel. 
        n_features = window_size * 4 + 1
        
        # Action : 2 valeurs continues entre -1 et 1 [Distance_Ask, Distance_Bid]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation : Historique des prix + Inventaire actuel.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)

    # Relance une nouvelle épisode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_this_episode = 0

        # Choisit un moment au hasard dans l'historique pour commencer
        # Calcule la limite maximum pour garantir qu'il restera assez de bougies
        # pour finir la partie.
        max_start = len(self.closes) - self.max_steps - 2 

        # Vérifie si le fichier de données est trop petit pour faire un choix au hasard.  
        if max_start <= self.window_size:
            self.current_step = self.window_size
        
        # Tire au sort le point de départ pour obliger l'agent
        # à s'adapter à des situations toujours nouvelles.
        else:
            self.current_step = np.random.randint(self.window_size, max_start)
        
        # Remet le compteur à zero
        self.inventory = 0
        self.cash = 0.0
        self.old_portfolio_value = 0.0
        return self._get_obs(), {}

    # Prépare les données vues par l'agent.
    def _get_obs(self):
        i = self.current_step
        w = self.window_size
        
        close_w  = self.closes[i-w:i]
        volume_w = self.volumes[i-w:i]
        high_w   = self.highs[i-w:i]
        low_w    = self.lows[i-w:i]
        
        base = close_w[-1] + 1e-8 # On normalise par le prix le plus récent
        norm_close  = (close_w / base) - 1.0
        norm_high   = (high_w / base) - 1.0
        norm_low    = (low_w / base) - 1.0
        vol_mean    = volume_w.mean() + 1e-8
        norm_volume = (volume_w / vol_mean) - 1.0
        
        # L'inventaire normalisé entre -1 et 1 
        norm_inventory = self.inventory / self.max_inventory 
        
        obs = np.concatenate([norm_close, norm_high, norm_low, norm_volume, [norm_inventory]])
        return obs.astype(np.float32)

    # LECTURE DES ACTIONS DE l'agent AU TEMPS T
    def step(self, action):
        current_price = self.closes[self.current_step]

        action = np.asarray(action, dtype=np.float32)

        ask_distance_pct = (action[0] + 1.0) / 2.0 * self.max_spread_pct
        bid_distance_pct = (action[1] + 1.0) / 2.0 * self.max_spread_pct

        agent_ask_price = current_price * (1.0 + ask_distance_pct)
        agent_bid_price = current_price * (1.0 - bid_distance_pct)

        self.current_step += 1

        next_high = self.highs[self.current_step]
        next_low = self.lows[self.current_step]
        next_close = self.closes[self.current_step]

        if next_low <= agent_bid_price and self.inventory < self.max_inventory:
            self.inventory += 1
            self.cash -= agent_bid_price

        if next_high >= agent_ask_price and self.inventory > -self.max_inventory:
            self.inventory -= 1
            self.cash += agent_ask_price

        current_portfolio_value = self.cash + (self.inventory * next_close)
        step_pnl = current_portfolio_value - self.old_portfolio_value
        inventory_penalty = 0.0001 * (self.inventory ** 2) * next_close
        reward = step_pnl - inventory_penalty
        reward = np.clip(reward, -10.0, 10.0)

        self.old_portfolio_value = current_portfolio_value
        self.steps_this_episode += 1
        terminated = self.steps_this_episode >= self.max_steps

        return self._get_obs(), float(reward), terminated, False, {}


def train_critic(lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, n_episodes=1000, MAX_GRAD_NORM=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[A2C Q-Critic] Training on {device}")
    episode_rewards_history = []
    episode_actions_history = []
    critic_losses_history = []
    actor_losses_history = []


    start_time = time.time()

    #env = ToyMarketEnv(max_steps=200)
    csv_file = "./data/SOLUSDT-aggTrades-2026-02.csv"  #same as PPO
    env = ContinuousBinanceEnv(csv_file, window_size=30, volume_threshold=1000)

    state_dim = env.observation_space.shape[0] #121
    actor  = ActorNN(state_dim=state_dim, action_dim=3).to(device)
    critic = CriticNN(state_dim=state_dim, action_dim=3).to(device)

    actor_optim  = optim.Adam(actor.parameters(),  lr=lr_actor)
    critic_optim = optim.Adam(critic.parameters(), lr=lr_critic)

    episode_rewards = deque(maxlen=50)

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        states, actions, rewards, next_states, log_probs, entropies = [], [], [], [], [], []

        while not done:
            #state_t = torch.tensor(obs).unsqueeze(0).to(device)
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            action, log_prob, entropy = actor.get_action(state_t)
            entropies.append(entropy)
            action_idx = action.item()

            #next_obs, reward, done = env.step(action_idx)
            
            if action_idx == 0:  # HOLD
                continuous_action = np.array([-0.5, -0.5])  
            elif action_idx == 1:  
                continuous_action = np.array([-0.8, 0.2])   
            elif action_idx == 2:  
                continuous_action = np.array([0.2, -0.8])   
            
            next_obs, reward, terminated, truncated, _ = env.step(continuous_action)
            done = terminated or truncated

            #reward = reward / 100.0 
            
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

        entropies_t = torch.stack(entropies)
        #entropy_coef = 0.1
        entropy_coef = max(0.01, 0.1 * (1 - ep / n_episodes))


        actor_loss = (-log_probs_t * advantages + -entropy_coef * entropies_t).mean()
        # actor_loss = -(log_probs_t * advantages).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
        actor_optim.step()

        n_hold = sum(a == 0 for a in actions)
        n_buy  = sum(a == 1 for a in actions)
        n_sell = sum(a == 2 for a in actions)

        episode_rewards_history.append(episode_reward)
        critic_losses_history.append(critic_loss.item())
        actor_losses_history.append(actor_loss.item())
        episode_actions_history.append({'hold': n_hold, 'buy': n_buy, 'sell': n_sell})

        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards)
            elapsed = time.time() - start_time
            print(f"Ep {ep:4d} | Avg Reward: {avg_reward:+.3f} | "
                  f"Critic Loss: {critic_loss.item():.4f} | Actor Loss: {actor_loss.item():.4f} | "
                  f"Elapsed: {elapsed:.1f}s")
            
            # n_hold = sum(a == 0 for a in actions)
            # n_buy  = sum(a == 1 for a in actions)
            # n_sell = sum(a == 2 for a in actions)

            print(
                f"Ep {ep:4d} | HOLD: {n_hold:3d} | BUY: {n_buy:3d} | SELL: {n_sell:3d}"
            )

            window = min(50, len(episode_rewards_history))
            reward_ma = np.convolve(episode_rewards_history, np.ones(window)/window, mode='valid')
            critic_ma = np.convolve(critic_losses_history, np.ones(window)/window, mode='valid')
            actor_ma = np.convolve(actor_losses_history, np.ones(window)/window, mode='valid')
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            #plot 1: ep rewards
            ax1.plot(episode_rewards_history, alpha=0.3, color='blue', label='Raw')
            ax1.plot(range(window-1, len(episode_rewards_history)), reward_ma, 
                    color='blue', linewidth=3, label='Moving Avg')
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            #plot 2: critic Loss
            ax2.plot(critic_losses_history, alpha=0.3, color='red')
            ax2.plot(range(window-1, len(critic_losses_history)), critic_ma, 
                    color='red', linewidth=3)
            ax2.set_title('Critic Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('MSE Loss')
            ax2.set_yscale('log')  # log scale for loss
            ax2.grid(True, alpha=0.3)
            
            #plot 3: actor Loss
            ax3.plot(actor_losses_history, alpha=0.3, color='green')
            ax3.plot(range(window-1, len(actor_losses_history)), actor_ma, 
                    color='green', linewidth=3)
            ax3.set_title('Actor Loss')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Policy Loss')
            ax3.grid(True, alpha=0.3)
            
            #plot 4: Action Distribution (latest 50 episodes average)
            recent_actions = episode_actions_history[-50:]
            holds = [a['hold'] for a in recent_actions]
            buys = [a['buy'] for a in recent_actions]
            sells = [a['sell'] for a in recent_actions]
            
            x = np.arange(3)
            width = 0.35
            ax4.bar(x - width/2, np.mean(holds), width, label='HOLD')
            ax4.bar(x, np.mean(buys), width, label='BUY')
            ax4.bar(x + width/2, np.mean(sells), width, label='SELL')
            ax4.set_title('Action Distribution (Recent Avg)')
            ax4.set_xlabel('Action')
            ax4.set_ylabel('Actions per Episode')
            ax4.set_xticks(x)
            ax4.set_xticklabels(['HOLD', 'BUY', 'SELL'])
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
    train_critic()
