import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

# ==========================================
# 1. ARCHITECTURE PROFONDE (Papier Wang et al.)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        layers = []
        for _ in range(4): # Le papier définit 4 couches par bloc
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU() # L'activation Swish
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) # Connexion résiduelle

class DeepEncoder(nn.Module):
    """Sert d'Acteur (State+Goal -> Action) et de Critique (State+Action -> Embed, Goal -> Embed)"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, depth: int = 16):
        super().__init__()
        num_blocks = max(1, depth // 4)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

# ==========================================
# 2. SIMULATEUR OFFLINE
# ==========================================
class OfflineMarketEnv:
    def __init__(self):
        self.state_dim = 4  # [imbalance, spread_bps, vol, inventory]
        self.action_dim = 3 # 0: HOLD, 1: BUY, 2: SELL
        self.goal_dim = 1   # [target_inventory]
        self.qty_per_trade = 5.0 # Quantité par trade dans votre bot
        self.max_steps = 200

    def reset(self):
        self.inventory = 0.0
        self.current_step = 0
        self.price = 100.0
        self.sma = 100.0
        return self._get_obs()

    def _get_obs(self):
        # MOCK : Simulation d'un carnet d'ordres bruité en RAM.
        # PLUS TARD : Remplacez ceci en piochant dans un historique de vos market_state.json !
        imb = random.uniform(-1.0, 1.0)
        spread_bps = random.uniform(1.0, 5.0) / 10.0
        deviation = ((self.price - self.sma) / self.sma * 100.0) / 2.0
        inv = self.inventory / 50.0  # Normalisé sur max 50 lots
        return np.array([imb, spread_bps, deviation, inv], dtype=np.float32)

    def step(self, action: int):
        self.current_step += 1
        
        # Simulation d'évolution des prix
        self.price += random.gauss(0, 0.1)
        self.sma = 0.9 * self.sma + 0.1 * self.price
        
        # Simulation d'exécution d'ordres
        if action == 1:   self.inventory += self.qty_per_trade
        elif action == 2: self.inventory -= self.qty_per_trade
        self.inventory = np.clip(self.inventory, -50.0, 50.0)
        
        obs = self._get_obs()
        # En RL Goal-Conditioned, l'état atteint (Achieved Goal) est l'inventaire courant
        achieved_goal = np.array([self.inventory / 50.0], dtype=np.float32) 
        done = self.current_step >= self.max_steps
        
        return obs, achieved_goal, done

# ==========================================
# 3. HINDSIGHT EXPERIENCE REPLAY (HER)
# ==========================================
class HERBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, episode):
        # episode: liste de (state, action, achieved_goal)
        self.buffer.append(episode)

    def sample(self, batch_size):
        states, actions, future_goals = [], [], []
        episodes = random.choices(self.buffer, k=batch_size)
        
        for ep in episodes:
            if len(ep) < 2: continue
            t = random.randint(0, len(ep) - 2)
            state, action, _ = ep[t]
            
            # TRICHE DU HER : Le but de la trajectoire devient un état futur de la même trajectoire !
            t_future = random.randint(t + 1, len(ep) - 1)
            _, _, future_goal = ep[t_future]
            
            states.append(state)
            actions.append(action)
            future_goals.append(future_goal)
            
        return (torch.FloatTensor(np.array(states)), 
                torch.LongTensor(np.array(actions)), 
                torch.FloatTensor(np.array(future_goals)))

# ==========================================
# 4. BOUCLE D'ENTRAÎNEMENT CONTRASTIVE RL
# ==========================================
def train_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Lancement de l'entraînement sur {device}")

    env = OfflineMarketEnv()
    buffer = HERBuffer(capacity=50000)

    BATCH_SIZE = 1024 
    NUM_EPOCHS = 500
    DEPTH = 16      
    REPR_DIM = 64

    # Initialisation de l'Acteur et du Critique 
    actor = DeepEncoder(env.state_dim + env.goal_dim, env.action_dim, depth=DEPTH).to(device)
    critic_sa = DeepEncoder(env.state_dim + env.action_dim, REPR_DIM, depth=DEPTH).to(device)
    critic_g = DeepEncoder(env.goal_dim, REPR_DIM, depth=DEPTH).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(list(critic_sa.parameters()) + list(critic_g.parameters()), lr=3e-4)

    # 1. Remplissage du buffer avec une politique exploratoire
    print("Pré-remplissage du Replay Buffer...")
    for _ in range(100):
        obs = env.reset()
        ep_data = []
        done = False
        while not done:
            action = random.randint(0, env.action_dim - 1)
            next_obs, achieved_goal, done = env.step(action)
            ep_data.append((obs, action, achieved_goal))
            obs = next_obs
        buffer.add_episode(ep_data)

    # 2. Entraînement Principal
    print("Apprentissage en cours...")
    for epoch in range(1, NUM_EPOCHS + 1):
        
        # --- A. Collecte d'un épisode (Exploration) ---
        obs = env.reset()
        ep_data = []
        target_goal = np.array([random.uniform(-1.0, 1.0)], dtype=np.float32)
        done = False
        
        actor.eval()
        while not done:
            if random.random() < 0.15: # Epsilon-greedy
                action = random.randint(0, env.action_dim - 1)
            else:
                with torch.no_grad():
                    st_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    g_t = torch.FloatTensor(target_goal).unsqueeze(0).to(device)
                    action = torch.argmax(actor(torch.cat([st_t, g_t], dim=-1)), dim=-1).item()
            
            next_obs, achieved_goal, done = env.step(action)
            ep_data.append((obs, action, achieved_goal))
            obs = next_obs
        buffer.add_episode(ep_data)

        # --- B. Mise à jour des Poids ---
        actor.train()
        critic_losses, actor_losses = [], []
        
        for _ in range(40): # 40 updates par epoch
            states, actions, goals = buffer.sample(BATCH_SIZE)
            if len(states) == 0: continue
            states, actions, goals = states.to(device), actions.to(device), goals.to(device)

            actions_onehot = F.one_hot(actions, num_classes=env.action_dim).float()

            # --- MISE A JOUR DU CRITIQUE (Loss InfoNCE Matrix) ---
            phi_sa = critic_sa(torch.cat([states, actions_onehot], dim=-1)) # [B, 64]
            psi_g = critic_g(goals) # [B, 64]
            
            # Matrice des distances entre TOUS les états et TOUS les buts du batch
            dist_matrix = torch.cdist(phi_sa, psi_g, p=2) 
            
            # InfoNCE: La diagonale représente les paires positives (le reste = négatif)
            logits = -dist_matrix 
            labels = torch.arange(len(states)).to(device)
            
            loss_c = F.cross_entropy(logits, labels)
            
            opt_critic.zero_grad()
            loss_c.backward()
            opt_critic.step()
            critic_losses.append(loss_c.item())

            # --- MISE A JOUR DE L'ACTEUR ---
            actor_logits = actor(torch.cat([states, goals], dim=-1)) # [B, 3]
            actor_probs = F.softmax(actor_logits, dim=-1)

            # Évaluer mathématiquement la Q-Value des 3 actions possibles
            with torch.no_grad():
                q_values = torch.zeros(len(states), env.action_dim).to(device)
                psi_g_detached = critic_g(goals)
                for a in range(env.action_dim):
                    a_hot = F.one_hot(torch.full((len(states),), a, device=device), num_classes=env.action_dim).float()
                    phi = critic_sa(torch.cat([states, a_hot], dim=-1))
                    q_values[:, a] = -torch.norm(phi - psi_g_detached, p=2, dim=1)

            # L'acteur maximise la Q-value espérée
            expected_q = torch.sum(actor_probs * q_values, dim=1)
            loss_a = -expected_q.mean()

            opt_actor.zero_grad()
            loss_a.backward()
            opt_actor.step()
            actor_losses.append(loss_a.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | Critic Loss: {np.mean(critic_losses):.4f} | Actor Loss: {np.mean(actor_losses):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(actor.state_dict(), "models/gcrl_actor.pth")
    print("✅ Modèle sauvegardé dans models/gcrl_actor.pth")

if __name__ == "__main__":
    train_agent()
