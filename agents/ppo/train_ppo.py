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
        
        # On convertit les actions [-1, 1] en pourcentages d'écartement [0%, 0.5%]
        ask_distance_pct = (action[0] + 1.0) / 2.0 * self.max_spread_pct
        bid_distance_pct = (action[1] + 1.0) / 2.0 * self.max_spread_pct

        # Calcule du prix exacts des ordres
        agent_ask_price = current_price * (1.0 + ask_distance_pct)
        agent_bid_price = current_price * (1.0 - bid_distance_pct)

        # Avance d'une étape dans le temps (T + 1)
        self.current_step += 1
        
        next_high = self.highs[self.current_step]
        next_low = self.lows[self.current_step]
        next_close = self.closes[self.current_step]

        # SIMULATION DU CARNET D'ORDRES 

        # Si le marché chute (Low <= Bid), on achète 1 SOL.
        if next_low <= agent_bid_price and self.inventory < self.max_inventory:
            self.inventory += 1
            self.cash -= agent_bid_price 

        # Si le marché monte (High >= Ask), on vend 1 SOL
        if next_high >= agent_ask_price and self.inventory > -self.max_inventory:
            self.inventory -= 1
            self.cash += agent_ask_price 

        # Calcul de la récompense 
        current_portfolio_value = self.cash + (self.inventory * next_close)
        
        # PnL du step
        step_pnl = current_portfolio_value - self.old_portfolio_value
        
        # Pénalité d'inventaire (Plus on s'éloigne de 0, plus ça coûte cher)
        # Multiplié par le prix actuel pour que la pénalité soit à la même échelle que le PnL
        inventory_penalty = 0.0001 * (self.inventory ** 2) * next_close 
        
        # Récompense finale = Profit - Risque.
        reward = step_pnl - inventory_penalty
        
        # Bloque les valeurs extrêmes 
        reward = np.clip(reward, -10.0, 10.0)

        # Mise à jour pour le prochain tour
        self.old_portfolio_value = current_portfolio_value
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

    num_episodes = 1000

    episode_scores = []

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

        last_loss = agent.losses[-1] if agent.losses else 0.0
        
        # On affiche la moyenne des deux écart-types
        current_stds = tf.exp(agent.log_std).numpy()
        mean_std = np.mean(current_stds)
        
        print(f"Épisode {episode + 1}/{num_episodes} | Score: {episode_reward:.2f} | Loss: {last_loss:.4f} | Mean Std: {mean_std:.4f}")

        episode_scores.append(episode_reward)

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

    plt.figure(figsize=(10, 5))
    plt.plot(episode_scores, label="Score par épisode", color="blue", alpha=0.6)
    
    if len(episode_scores) >= 50:
        moving_avg = pd.Series(episode_scores).rolling(window=50).mean()
        plt.plot(moving_avg, label="Moyenne mobile (50 ep)", color="red", linewidth=2)

    plt.title("Évolution du Score ")
    plt.xlabel("Épisodes")
    plt.ylabel("Score (Récompense)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("courbe_apprentissage.png")
    print("Courbe sauvegardée sous le nom 'courbe_apprentissage.png'")
    
    plt.show()
