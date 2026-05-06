"""
A2C (Advantage Actor-Critic) strategy 
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional

from ..interfaces import BaseStrategy, StrategySignal
from ..types import Side
from ..market_data import MarketData


class ActorNN(nn.Module):
    """
    Actor -> action logits (hold, buy, sell)
    Input:  s_t  (state vector)
    Output: logits for Categorical distribution over actions
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
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.actor_head(features)
        return logits

    def get_action(self, x: torch.Tensor):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class CriticNN(nn.Module):
    """
    Critic (Value net) — pure V(s) network, matches the lecture slide.

    Input:  s_t  only (no action input, no Q head)
    Output: V_φ(s_t)  — scalar estimate of the state value

    Used twice during training:
      - V_φ(s_t)   → compute advantage A_t = R_t - V_φ(s_t)
      - V_φ(s_t+1) → bootstrap return target R_t = r_t + γ * V_φ(s_t+1)
    """
    def __init__(self, state_dim: int = 121, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),   # scalar V(s)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            value: (batch, 1)  — V_φ(s)
        """
        return self.net(state)


class A2CStrategy(BaseStrategy):
    """
    Standard VPG/A2C inference strategy.
    Loads the pre-trained actor and critic, runs greedy inference.
    Train with: python3 -m agents.train_a2c
    """

    def __init__(self, symbol: str = "SOLUSDT",
                 actor_path: str = "models/a2c_actor.pth",
                 critic_path: str = "models/a2c_critic.pth",
                 quantity_per_trade: float = 5.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.market_data = MarketData(window_size=10)
        self.quantity = quantity_per_trade
        self.model_loaded = False
        self.current_inventory = 0.0
        self.observation: Optional[np.ndarray] = None

        self.actor  = ActorNN(state_dim=121, action_dim=3).to(self.device)
        self.critic = CriticNN(state_dim=121).to(self.device)

        actor_loaded  = self._load_model(self.actor,  actor_path)
        critic_loaded = self._load_model(self.critic, critic_path)
        self.model_loaded = actor_loaded and critic_loaded

        if self.model_loaded:
            self.actor.eval()
            self.critic.eval()
            print("[A2C] Both models loaded successfully")
        else:
            print("[A2C] Missing models. Run 'python3 -m agents.train_a2c' first.")

    def _load_model(self, model, path):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
            return True
        return False

    def build_observation(self) -> Optional[np.ndarray]:
        """
        Build 121-dim observation vector identical to training env:
        [norm_close(30), norm_high(30), norm_low(30), norm_volume(30), norm_inventory]
        """
        if len(self.market_data.prices) < 30:
            return None

        prices = np.array(self.market_data.prices[-30:])
        high   = prices * 1.001
        low    = prices * 0.999
        volume = np.ones(30)

        base       = prices[-1] + 1e-8
        norm_close  = (prices  / base) - 1.0
        norm_high   = (high    / base) - 1.0
        norm_low    = (low     / base) - 1.0
        vol_mean    = volume.mean() + 1e-8
        norm_volume = (volume  / vol_mean) - 1.0
        norm_inv    = self.current_inventory / 10.0

        return np.concatenate([norm_close, norm_high, norm_low, norm_volume, [norm_inv]]).astype(np.float32)

    # BaseStrategy interface
    def update_data(self, market_state: Dict) -> None:
        self.market_data.update(market_state)
        self.observation = self.build_observation()

    def compute_signal(self) -> StrategySignal:
        if self.observation is None or not self.model_loaded:
            return StrategySignal(None, 0, 0, "A2C: warming up...", {})

        state_t = torch.tensor(self.observation, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits    = self.actor(state_t)
            probs     = torch.softmax(logits, dim=1)
            action_idx = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, action_idx].item()

            value = self.critic(state_t).item()

        mid    = int(self.market_data.mid_price * 100)
        offset = int(self.market_data.spread * 100 / 2)

        if action_idx == 1:   # BUY
            self.current_inventory += self.quantity
            return StrategySignal(Side.BUY, mid - offset, self.quantity,
                                  f"A2C BUY (conf: {confidence:.2f})",
                                  {"action": "BUY", "value": value, "confidence": confidence})

        elif action_idx == 2:  # SELL
            self.current_inventory -= self.quantity
            return StrategySignal(Side.SELL, mid + offset, self.quantity,
                                  f"A2C SELL (conf: {confidence:.2f})",
                                  {"action": "SELL", "value": value, "confidence": confidence})

        return StrategySignal(None, 0, 0, f"A2C HOLD (conf: {confidence:.2f})",
                              {"action": "HOLD", "value": value})

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "A2C Agent",
            "model_loaded": self.model_loaded,
            "inventory": self.current_inventory,
            "last_obs": self.observation.tolist() if self.observation is not None else None,
        }
