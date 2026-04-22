import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional

from ..interfaces import BaseStrategy, StrategySignal
from ..types import Side
from ..market_data import MarketData

#A2C (Advantage Actor-Critic) strategy


#nn architecture for the A2C agent
class ActorNN(nn.Module):
    """
    Actor  -> action logits (hold, buy, sell)
    """
    def __init__(self, state_dim: int = 121, action_dim: int = 3, hidden_dim: int = 512): 
        super().__init__()
        self.backbone = nn.Sequential( 
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden_dim // 2, action_dim)  #actor's output, 3 scores for(hold,buy,sell)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.actor_head(features) #scores for each action
        return logits

    def get_action(self, x: torch.Tensor):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

class CriticNN(nn.Module):
    def __init__(self, state_dim: int = 121, action_dim: int = 3, hidden_dim: int = 512):
        super().__init__()
        # Q-value branch: uses (state + action)
        self.q_input = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # V-value branch: uses state only
        self.v_input = nn.Linear(state_dim, hidden_dim)
        self.v_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state, action):
        if action.ndim > 1 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        action_onehot = F.one_hot(action.long(), num_classes=3).float()  # convert action index to one-hot vector
        st_action = torch.cat([state, action_onehot], dim=-1)

        q_features = torch.relu(self.q_input(st_action))
        q = self.q_head(q_features)

        v_features = torch.relu(self.v_input(state))
        v = self.v_head(v_features)

        return q, v  # Q(s,a) and V(s)


#strategy
class A2CStrategy(BaseStrategy):
    """
    Q-critic A2C
    Loads the pre-trained networks and runs greedy inference.
    Train with: python3 -m agents.train_a2c
    """

    def __init__(self, symbol: str = "SOLUSDT",
                 actor_path: str = "models/a2c_actor.pth",
                 critic_path: str = "models/a2c_critic.pth",
                 quantity_per_trade: float = 5.0):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use gpu if available (faster inference), otherwise cpu
        self.market_data = MarketData(window_size=10) #helper from market_data.py
        self.quantity    = quantity_per_trade #how much to buy/sell per trade (can be changed based on risk appetite)
        self.model_loaded = False
        self.current_inventory = 0.0
        self.observation: Optional[np.ndarray] = None #current state vector(updated in update_data every tick)

        self.actor = ActorNN(state_dim=4, action_dim=3).to(self.device) #nn architecture (random weights)
        self.critic = CriticNN(state_dim=4, action_dim=3).to(self.device) 

        actor_loaded = self._load_model(self.actor, actor_path)
        critic_loaded = self._load_model(self.critic, critic_path)
        self.model_loaded = actor_loaded and critic_loaded

        if self.model_loaded:
            self.actor.eval()
            self.critic.eval()
            print(f"[A2C Q-critic] Both models loaded successfully")
        else:
            print(f"[A2C] Missing models. Run 'python3 -m agents.train_a2c' first.")       

    def _load_model(self, model, path):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
            return True
        return False
    
    # def _build_observation(self) -> Optional[np.ndarray]:
    #     """Build normalised state vector — identical to training environment."""
    #     if len(self.market_data.prices) < 3: 
    #          return None

    #     b_qty = sum(b["quantity"] for b in self.market_data.bids[:3]) if self.market_data.bids else 0 
    #     a_qty = sum(a["quantity"] for a in self.market_data.asks[:3]) if self.market_data.asks else 0 
    #     imb   = (b_qty - a_qty) / (b_qty + a_qty) if (b_qty + a_qty) > 0 else 0.0 #orderbook imbalance

    #     norm_imb    = float(np.clip(imb, -1.0, 1.0))
    #     norm_spread = (self.market_data.spread / self.market_data.mid_price * 10000) / 10.0 \
    #                   if self.market_data.mid_price else 0.0
    #     norm_dev    = (self.market_data.get_price_deviation() or 0.0) / 2.0
    #     norm_inv    = self.current_inventory / 50.0

    #     return np.array([norm_imb, norm_spread, norm_dev, norm_inv], dtype=np.float32)

    def build_observation(self) -> Optional[np.ndarray]:
        """
        Build 121-dim observation vector identical to PPO training env:
        [norm_close(30), norm_high(30), norm_low(30), norm_volume(30), norm_inventory]
        """
        #get last 30 prices/volumes from marketdata (same window as PPO)
        if len(self.marketdata.prices) < 30:
            return None
        
        prices = np.array(self.marketdata.prices[-30:])  # last 30 closes

        high = prices * 1.001   # fake high (0.1% above close)
        low  = prices * 0.999   #fake low (0.1% below close)
        volume = np.ones(30)    #fake constant volume (PPO normalizes it anyway)
        
        base = prices[-1] + 1e-8 
        norm_close = (prices / base) - 1.0
        norm_high  = (high / base) - 1.0
        norm_low   = (low / base) - 1.0
        
        vol_mean = volume.mean() + 1e-8
        norm_volume = (volume / vol_mean) - 1.0
        
        norm_inventory = self.current_inventory / 10.0 
        
        obs = np.concatenate([
            norm_close, norm_high, norm_low, norm_volume, [norm_inventory]
        ])
        
        return obs.astype(np.float32)

    #BaseStrategy Interface

    def update_data(self, market_state: Dict) -> None:
        self.market_data.update(market_state)
        self.observation = self._build_observation()

    def compute_signal(self) -> StrategySignal:
        if self.observation is None or not self.model_loaded:
            return StrategySignal(None, 0, 0, "A2C: warming up...", {})

        state_t = torch.tensor(self.observation).unsqueeze(0).to(self.device) #convert state to tensor and add batch dimension (1,4)

        with torch.no_grad(): #no need to compute gradients during inference
            logits = self.actor(state_t)
            probs = torch.softmax(logits, dim=1)
            action_idx = torch.argmax(logits, dim=-1).item() #greedy
            confidence = probs[0, action_idx].item()

            q_value, _ = self.critic(state_t, torch.tensor([action_idx], device=self.device))
            q_value = q_value.item()

        mid    = int(self.market_data.mid_price * 100) 
        offset = int(self.market_data.spread * 100 / 2) #place buy/sell orders at the best bid/ask (mid ± half spread)

        if action_idx == 1:   # BUY
            self.current_inventory += self.quantity
            return StrategySignal(
                Side.BUY, mid - offset, self.quantity,
                f"A2C BUY (conf: {confidence:.2f})",
                {"action": "BUY", "q_value": q_value, "confidence": confidence}
            )
        elif action_idx == 2:  # SELL
            self.current_inventory -= self.quantity
            return StrategySignal(
                Side.SELL, mid + offset, self.quantity,
                f"A2C SELL (conf: {confidence:.2f})",
                {"action": "SELL", "qvalue": q_value, "confidence": confidence}
            )

        return StrategySignal(None, 0, 0, f"A2C HOLD (conf: {confidence:.2f})", 
                              {"action": "HOLD", "q_value": q_value}) # HOLD

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "A2C Agent",
            "model_loaded": self.model_loaded,
            "inventory": self.current_inventory,
            "last_obs": self.observation.tolist() if self.observation is not None else None,
        }
