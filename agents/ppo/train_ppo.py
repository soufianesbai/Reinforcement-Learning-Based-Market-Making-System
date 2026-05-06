import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import os
import matplotlib.pyplot as plt


def build_volume_bars(csv_path, volume_threshold=1000):
    print(f"Lecture du fichier : {csv_path}...")

    df = pd.read_csv(csv_path, header=None,
                     names=['id', 'price', 'qty', 'f', 'l', 'timestamp', 'm', 'M'],
                     low_memory=False)

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df = df.dropna(subset=['price', 'qty'])

    if len(df) == 0:
        raise ValueError("ERREUR : Le CSV est vide")

    print(f"Création des Volume Bars (Seuil: {volume_threshold} SOL)...")

    df['cum_vol'] = df['qty'].cumsum()
    df['bar_id'] = df['cum_vol'] // volume_threshold

    ohlcv = df.groupby('bar_id').agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('qty', 'sum')
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

        self.window_size    = window_size
        self.max_inventory  = 3
        self.max_spread_pct = 0.01

        total_candles = len(self.closes)
        if total_candles < 250:
            self.max_steps = max(10, total_candles - self.window_size - 5)
        else:
            self.max_steps = 200

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

        self.inventory          = 0
        self.cash               = 0.0
        self.old_mtm            = 0.0
        self.episode_trades     = 0
        self.episode_inventories = []

        return self._get_obs(), {}

    def _get_obs(self):
        i = self.current_step
        w = self.window_size

        close_w  = self.closes[i-w:i]
        volume_w = self.volumes[i-w:i]
        high_w   = self.highs[i-w:i]
        low_w    = self.lows[i-w:i]

        base       = close_w[-1] + 1e-8
        norm_close  = (close_w / base) - 1.0
        norm_high   = (high_w / base) - 1.0
        norm_low    = (low_w / base) - 1.0
        vol_mean    = volume_w.mean() + 1e-8
        norm_volume = (volume_w / vol_mean) - 1.0

        norm_inventory = self.inventory / self.max_inventory

        obs = np.concatenate([norm_close, norm_high, norm_low, norm_volume, [norm_inventory]])
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.closes[self.current_step]

        ask_distance_pct = (action[0] + 1.0) / 2.0 * self.max_spread_pct
        bid_distance_pct = (action[1] + 1.0) / 2.0 * self.max_spread_pct

        agent_ask_price = current_price * (1.0 + ask_distance_pct)
        agent_bid_price = current_price * (1.0 - bid_distance_pct)

        self.current_step += 1

        next_high  = self.highs[self.current_step]
        next_low   = self.lows[self.current_step]
        next_close = self.closes[self.current_step]

        filled = False
        if np.random.random() < 0.5:
            if next_low <= agent_bid_price and self.inventory < self.max_inventory:
                self.inventory += 1
                self.cash -= agent_bid_price
                self.episode_trades += 1
                filled = True
            if not filled and next_high >= agent_ask_price and self.inventory > -self.max_inventory:
                self.inventory -= 1
                self.cash += agent_ask_price
                self.episode_trades += 1
        else:
            if next_high >= agent_ask_price and self.inventory > -self.max_inventory:
                self.inventory -= 1
                self.cash += agent_ask_price
                self.episode_trades += 1
                filled = True
            if not filled and next_low <= agent_bid_price and self.inventory < self.max_inventory:
                self.inventory += 1
                self.cash -= agent_bid_price
                self.episode_trades += 1

        self.episode_inventories.append(self.inventory)

        mtm    = self.cash + self.inventory * next_close
        reward = mtm - self.old_mtm
        self.old_mtm = mtm

        reward -= 0.0005 * (self.inventory ** 2) * next_close

        self.steps_this_episode += 1
        terminated = self.steps_this_episode >= self.max_steps

        if terminated and self.inventory != 0:
            reward -= 0.005 * abs(self.inventory) * next_close

        reward = np.clip(reward, -10.0, 10.0)

        return self._get_obs(), float(reward), terminated, False, {}


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class PPOAgent:
    def __init__(self, env, policy_network, value_network,
                 learning_rate=3e-4, gamma=0.99, clip_range=0.2,
                 epochs=30, batch_size=256, lambda_=0.95, entropy_coeff=0.01):
        self.env            = env
        self.policy_network = policy_network
        self.value_network  = value_network
        self.device         = next(policy_network.parameters()).device

        self.log_std = nn.Parameter(torch.zeros(2, device=self.device))

        all_params = (
            list(policy_network.parameters()) +
            list(value_network.parameters()) +
            [self.log_std]
        )
        self.optimizer = optim.Adam(all_params, lr=learning_rate)

        self.gamma         = gamma
        self.clip_range    = clip_range
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.lambda_       = lambda_
        self.entropy_coeff = entropy_coeff

        self.losses        = []
        self.policy_losses = []
        self.value_losses  = []
        self.entropy_losses = []

    def act(self, state, explore=True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu = self.policy_network(state_t).squeeze(0).cpu().numpy()

        if explore:
            std    = torch.exp(self.log_std).detach().cpu().numpy()
            action = mu + np.random.normal(0, std, size=mu.shape)
        else:
            action = mu

        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action, mu

    def _log_prob(self, mu, action, log_std):
        var      = torch.exp(log_std * 2)
        log_prob = -0.5 * torch.sum(
            (action - mu) ** 2 / var + torch.log(2 * np.pi * var), dim=1
        )
        return log_prob

    def _compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta        = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae          = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def observe(self, states, actions, old_mus, rewards, next_states, dones):
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.FloatTensor(actions).to(self.device)
        old_mus_t     = torch.FloatTensor(old_mus).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        rewards       = np.array(rewards, dtype=np.float32)
        dones         = np.array(dones, dtype=np.float32)

        with torch.no_grad():
            values      = self.value_network(states_t).squeeze(-1).cpu().numpy()
            next_values = self.value_network(next_states_t).squeeze(-1).cpu().numpy()

        advantages, returns = self._compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t    = torch.FloatTensor(returns).to(self.device)

        with torch.no_grad():
            old_log_probs = self._log_prob(old_mus_t, actions_t, self.log_std)

        epoch_policy_losses  = []
        epoch_value_losses   = []
        epoch_entropy_losses = []

        n = len(states_t)
        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx   = indices[start:start + self.batch_size]
                idx_t = torch.LongTensor(idx).to(self.device)

                b_states     = states_t[idx_t]
                b_actions    = actions_t[idx_t]
                b_old_lp     = old_log_probs[idx_t]
                b_advantages = advantages_t[idx_t]
                b_returns    = returns_t[idx_t]

                new_mus       = self.policy_network(b_states)
                new_log_probs = self._log_prob(new_mus, b_actions, self.log_std)

                ratio       = torch.exp(new_log_probs - b_old_lp)
                clipped     = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.mean(torch.min(ratio * b_advantages, clipped * b_advantages))

                clipped_log_std = torch.clamp(self.log_std, -10.0, 0.0)
                entropy      = 0.5 * torch.sum(1.0 + torch.log(2 * np.pi * torch.exp(clipped_log_std * 2)))
                entropy_loss = -self.entropy_coeff * entropy

                value_preds = self.value_network(b_states).squeeze(-1)
                value_loss  = torch.mean((b_returns - value_preds) ** 2)

                loss = policy_loss + 0.5 * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                all_params = (
                    list(self.policy_network.parameters()) +
                    list(self.value_network.parameters()) +
                    [self.log_std]
                )
                nn.utils.clip_grad_norm_(all_params, 0.5)
                self.optimizer.step()

                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropy_losses.append(entropy_loss.item())

        self.policy_losses.append(np.mean(epoch_policy_losses))
        self.value_losses.append(np.mean(epoch_value_losses))
        self.entropy_losses.append(np.mean(epoch_entropy_losses))
        self.losses.append(
            np.mean(epoch_policy_losses) +
            0.5 * np.mean(epoch_value_losses) +
            np.mean(epoch_entropy_losses)
        )

    def save(self, policy_path, value_path):
        torch.save(self.policy_network.state_dict(), policy_path)
        torch.save(self.value_network.state_dict(), value_path)

    def load(self, policy_path, value_path):
        self.policy_network.load_state_dict(torch.load(policy_path, map_location=self.device))
        self.value_network.load_state_dict(torch.load(value_path, map_location=self.device))


if __name__ == "__main__":
    print("Création de l'environnement de Market Making")

    csv_file = "../../data/SOLUSDT-aggTrades-2026-02.csv"
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    env       = ContinuousBinanceEnv(csv_file, volume_threshold=100)
    input_dim = env.observation_space.shape[0]

    policy_net = PolicyNetwork(input_dim).to(device)
    value_net  = ValueNetwork(input_dim).to(device)

    agent = PPOAgent(env=env, policy_network=policy_net, value_network=value_net,
                     learning_rate=3e-4, entropy_coeff=0.01)

    num_episodes = 2000

    episode_scores           = []
    episode_trades           = []
    episode_avg_inventories  = []
    episode_stds             = []

    print("Début de l'entraînement")
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        states, actions, old_mus, rewards, next_states, dones = [], [], [], [], [], []

        while not done:
            action, mu = agent.act(state, explore=True)
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            old_mus.append(mu)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))

            state = next_state
            episode_reward += reward

        agent.observe(
            np.array(states), np.array(actions), np.array(old_mus),
            np.array(rewards), np.array(next_states), np.array(dones)
        )

        mean_std = torch.exp(agent.log_std).detach().cpu().numpy().mean()

        episode_scores.append(episode_reward)
        episode_trades.append(env.episode_trades)
        avg_inv = np.mean(np.abs(env.episode_inventories)) if env.episode_inventories else 0.0
        episode_avg_inventories.append(avg_inv)
        episode_stds.append(mean_std)

        print(f"Épisode {episode + 1}/{num_episodes} | "
              f"Score: {episode_reward:.2f} | "
              f"Trades: {env.episode_trades} | "
              f"Inv moy: {avg_inv:.2f} | "
              f"Loss: {agent.losses[-1]:.4f} | "
              f"Std: {mean_std:.4f}")

        if (episode + 1) % 50 == 0:
            os.makedirs("models_checkpoints", exist_ok=True)
            agent.save(
                f"models_checkpoints/ppo_policy_ep{episode+1}.pth",
                f"models_checkpoints/ppo_value_ep{episode+1}.pth"
            )

    print("Entraînement terminé...")

    agent.save(
        "models_checkpoints/ppo_policy_FINAL.pth",
        "models_checkpoints/ppo_value_FINAL.pth"
    )

    smooth = lambda data, w=50: pd.Series(data).rolling(window=w, min_periods=1).mean()

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Métriques d'Entraînement PPO — Market Maker", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(episode_scores, alpha=0.3, color="blue", label="Score brut")
    ax.plot(smooth(episode_scores), color="red", linewidth=2, label="Moyenne mobile (50 ep)")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_title("Score (Récompense) par Épisode")
    ax.set_xlabel("Épisodes")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(agent.losses, alpha=0.3, color="purple")
    ax.plot(smooth(agent.losses), color="purple", linewidth=2, label="Loss totale")
    ax.set_title("Loss Totale")
    ax.set_xlabel("Épisodes")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(smooth(agent.policy_losses), color="orange", linewidth=2, label="Policy Loss")
    ax.plot(smooth(agent.value_losses), color="green", linewidth=2, label="Value Loss")
    ax.plot(smooth(agent.entropy_losses), color="red", linewidth=2, label="Entropy Loss")
    ax.set_title("Décomposition des Losses")
    ax.set_xlabel("Épisodes")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episode_trades, alpha=0.3, color="teal")
    ax.plot(smooth(episode_trades), color="teal", linewidth=2, label="Trades / épisode")
    ax.set_title("Activité de Trading")
    ax.set_xlabel("Épisodes")
    ax.set_ylabel("Nombre de Trades")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(episode_avg_inventories, alpha=0.3, color="brown")
    ax.plot(smooth(episode_avg_inventories), color="brown", linewidth=2, label="Inventaire moyen (|inv|)")
    ax.set_title("Gestion de l'Inventaire")
    ax.set_xlabel("Épisodes")
    ax.set_ylabel("Inventaire Absolu Moyen")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(episode_stds, color="magenta", linewidth=2, label="Mean Std (exploration)")
    ax.set_title("Niveau d'Exploration (σ)")
    ax.set_xlabel("Épisodes")
    ax.set_ylabel("Écart-type moyen")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("courbe_apprentissage.png", dpi=150, bbox_inches="tight")
    print("Courbes sauvegardées sous 'courbe_apprentissage.png'")
    plt.show()
