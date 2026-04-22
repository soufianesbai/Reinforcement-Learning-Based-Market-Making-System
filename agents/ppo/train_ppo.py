import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import os
import matplotlib.pyplot as plt


# Transforme les transactions brutes en bougies de volume.
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


# L'ENVIRONNEMENT DE MARKET MAKING : Simulateur de marché pour l'agent
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
        self.max_inventory = 3 # L'agent ne peut pas détenir plus de 10 SOL
        self.max_spread_pct = 0.01 # Écartement maximum autorisé est 0.5%
        
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
        self.old_cash = 0.0
        self.episode_trades = 0
        self.episode_inventories = []

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
                                                                                                                                            
        ask_distance_pct = (action[0] + 1.0) / 2.0 * self.max_spread_pct                                                                  
        bid_distance_pct = (action[1] + 1.0) / 2.0 * self.max_spread_pct                                                                  
                                                                                                                                            
        agent_ask_price = current_price * (1.0 + ask_distance_pct)                                                                        
        agent_bid_price = current_price * (1.0 - bid_distance_pct)
                                                                                                                                            
        self.current_step += 1                                                                                                            
    
        next_high = self.highs[self.current_step]                                                                                         
        next_low = self.lows[self.current_step]
        next_close = self.closes[self.current_step]                                                                                       

        filled = False                                                                                                                                                                                                                                                        
        if np.random.random() < 0.5:
            # Vérifie le bid d'abord                                                                                                          
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
            # Vérifie le ask d'abord
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

        spread_reward = self.cash - self.old_cash                                                                                         
        self.old_cash = self.cash
                                                                                                                                            
        inventory_penalty = 0.0005 * (self.inventory ** 2) * next_close 

        reward = spread_reward - inventory_penalty
        reward = np.clip(reward, -10.0, 10.0)
                                                                                                                                            
        self.steps_this_episode += 1
        terminated = self.steps_this_episode >= self.max_steps                                                                            
                    
        return self._get_obs(), float(reward), terminated, False, {}

# RÉSEAUX DE NEURONES
def build_policy_network(input_shape):  # Le cerveau
    inputs = layers.Input(shape=input_shape)
    # Première couche cachée de 64 neurones
    x = layers.Dense(64, activation='relu')(inputs)
    # Deuxième couche de 64 neurones
    x = layers.Dense(64, activation='relu')(x)
    # Couche finale qui sort 2 valeurs (Ask et Bid) comprises entre -1 et 1 grâce à 'tanh'.
    outputs = layers.Dense(2, activation='tanh')(x)
    return models.Model(inputs, outputs)

def build_value_network(input_shape): # Le critique
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    # Couche finale qui prédit 1 seul nombre : l'estimation de la récompense future totale
    outputs = layers.Dense(1, activation='linear')(x)
    return models.Model(inputs, outputs)

# L'AGENT PPO 
class PPOAgent:
    def __init__(self, env, policy_network, value_network,
                 learning_rate=3e-4, gamma=0.99, clip_range=0.2,
                 epochs=10, batch_size=256, lambda_=0.95, entropy_coeff=0.01):
        self.env = env
        self.policy_network = policy_network
        self.value_network = value_network
        
        self.log_std = tf.Variable(tf.zeros((2,)), trainable=True, name="log_std")
        
        self.optimizer = Adam(learning_rate=learning_rate)
        self.gamma = gamma
        self.clip_range = clip_range
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.entropy_coeff = entropy_coeff

        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def act(self, state, explore=True):
        state_t = np.expand_dims(state, axis=0)
        mu = self.policy_network(state_t, training=False).numpy()[0]
        
        if explore:
            std = tf.exp(self.log_std).numpy()
            action = mu + np.random.normal(0, std, size=mu.shape)
        else:
            action = mu
            
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action, mu

    def _log_prob(self, mu, action, log_std):
        var = tf.exp(log_std * 2)
        log_prob = -0.5 * tf.reduce_sum(
            tf.square(action - mu) / var + tf.math.log(2 * np.pi * var), axis=1
        )
        return log_prob

    def _compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def observe(self, states, actions, old_mus, rewards, next_states, dones):
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.float32)
        old_mus = tf.cast(old_mus, tf.float32)
        next_states = tf.cast(next_states, tf.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        values = self.value_network(states, training=False).numpy().flatten()
        next_values = self.value_network(next_states, training=False).numpy().flatten()

        advantages, returns = self._compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages = tf.cast(advantages, tf.float32)
        returns = tf.cast(returns, tf.float32)
        
        old_log_probs = self._log_prob(old_mus, actions, self.log_std)

        epoch_policy_losses = []                                                                                                          
        epoch_value_losses = []
        epoch_entropy_losses = []


        n = len(states)                                                                                                                   
        for _ in range(self.epochs):
            indices = np.random.permutation(n)                                                                                            
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]                                                                              
    
                b_states = tf.gather(states, idx)                                                                                         
                b_actions = tf.gather(actions, idx)
                b_old_log_probs = tf.gather(old_log_probs, idx)                                                                           
                b_advantages = tf.gather(advantages, idx)
                b_returns = tf.gather(returns, idx)                                                                                       
                                                                                                                                            
                with tf.GradientTape() as tape:
                    new_mus = self.policy_network(b_states, training=True)                                                                
                    new_log_probs = self._log_prob(new_mus, b_actions, self.log_std)
                                                                                                                                            
                    ratio = tf.exp(new_log_probs - b_old_log_probs)
                    clipped = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)                                           
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * b_advantages, clipped * b_advantages))                               
                                                                                                                                            
                    clipped_log_std = tf.clip_by_value(self.log_std, -10.0, 0.0)                                                          
                    entropy = 0.5 * tf.reduce_sum(1.0 + tf.math.log(2 * np.pi * tf.exp(clipped_log_std * 2)))                             
                    entropy_loss = -self.entropy_coeff * tf.reduce_mean(entropy)                                                          
                    
                    value_preds = tf.squeeze(self.value_network(b_states, training=True))                                                 
                    value_loss = tf.reduce_mean(tf.square(b_returns - value_preds))
                                                                                                                                            
                    loss = policy_loss + 0.5 * value_loss + entropy_loss                                                                  
    
                all_vars = self.policy_network.trainable_variables + self.value_network.trainable_variables + [self.log_std]              
                grads = tape.gradient(loss, all_vars)
                grads = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in grads]                                                  
                self.optimizer.apply_gradients(zip(grads, all_vars))                                                                      
                                                                                                                                            
                epoch_policy_losses.append(float(policy_loss))                                                                            
                epoch_value_losses.append(float(value_loss))                                                                              
                epoch_entropy_losses.append(float(entropy_loss))

        self.policy_losses.append(np.mean(epoch_policy_losses))                                                                           
        self.value_losses.append(np.mean(epoch_value_losses))
        self.entropy_losses.append(np.mean(epoch_entropy_losses))                                                                         
        self.losses.append(np.mean(epoch_policy_losses) + 0.5 * np.mean(epoch_value_losses) + np.mean(epoch_entropy_losses))

        n = len(states)
        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                
                b_states = tf.gather(states, idx)
                b_actions = tf.gather(actions, idx)
                b_old_log_probs = tf.gather(old_log_probs, idx)
                b_advantages = tf.gather(advantages, idx)
                b_returns = tf.gather(returns, idx)

                with tf.GradientTape() as tape:
                    new_mus = self.policy_network(b_states, training=True)
                    new_log_probs = self._log_prob(new_mus, b_actions, self.log_std)
                    
                    ratio = tf.exp(new_log_probs - b_old_log_probs)
                    clipped = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * b_advantages, clipped * b_advantages))
                    
                    # Clipping pour éviter l'explosion de l'incertitude
                    clipped_log_std = tf.clip_by_value(self.log_std, -10.0, 0.0)
                    entropy = 0.5 * tf.reduce_sum(1.0 + tf.math.log(2 * np.pi * tf.exp(clipped_log_std * 2)))
                    entropy_loss = -self.entropy_coeff * tf.reduce_mean(entropy)
                    
                    value_preds = tf.squeeze(self.value_network(b_states, training=True))
                    value_loss = tf.reduce_mean(tf.square(b_returns - value_preds))
                    
                    loss = policy_loss + 0.5 * value_loss + entropy_loss

                all_vars = self.policy_network.trainable_variables + self.value_network.trainable_variables + [self.log_std]
                grads = tape.gradient(loss, all_vars)
                grads = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in grads]
                self.optimizer.apply_gradients(zip(grads, all_vars))
                self.losses.append(loss.numpy())

    def save(self, policy_path, value_path):
        self.policy_network.save(policy_path)
        self.value_network.save(value_path)


if __name__ == "__main__":
    print("Création de l'environnement de Market Making")
    
    csv_file = "../../data/SOLUSDT-aggTrades-2026-02.csv" 
    
    env = ContinuousBinanceEnv(csv_file, volume_threshold=1000)
    obs_shape = env.observation_space.shape

    policy_net = build_policy_network(obs_shape)
    value_net = build_value_network(obs_shape)

    agent = PPOAgent(env=env, policy_network=policy_net, value_network=value_net,
                     learning_rate=3e-4, entropy_coeff=0.01)

    num_episodes = 2000

    episode_scores = []
    episode_trades = []
    episode_avg_inventories = []
    episode_stds = []

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

        current_stds = tf.exp(agent.log_std).numpy()                                                                                  
        mean_std = np.mean(current_stds)
                                                                                                                                      
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
                f"models_checkpoints/ppo_policy_ep{episode+1}.keras",
                f"models_checkpoints/ppo_value_ep{episode+1}.keras"                                                                   
            )                                     
    
    print("Entraînement terminé...")
        
    agent.save(
        "models_checkpoints/ppo_policy_FINAL.keras",
        "models_checkpoints/ppo_value_FINAL.keras"
    )

    # ==========================================                                                                                      
    # GRAPHIQUES DE CONVERGENCE
    # ==========================================                                                                                      
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

