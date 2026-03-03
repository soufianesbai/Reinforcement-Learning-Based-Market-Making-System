import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

from ..interfaces import BaseStrategy, StrategySignal
from ..types import Side
from ..market_data import MarketData

#A2C (Advantage Actor-Critic) strategy


#nn architecture for the A2C agent
class ActorCriticNetwork(nn.Module):
    """
    Shared layers + separate Actor and Critic outputs.
    Actor  -> probability distribution over actions (BUY / SELL / HOLD)
    Critic -> scalar state value V(s)
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 3, hidden_dim: int = 128): 
        super().__init__()
        self.backbone = nn.Sequential( 
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ) #shared layers to understand the market state (spread, inventory, imbalance, deviation)
        self.actor_head  = nn.Linear(hidden_dim, action_dim)  #actor's output, 3 scores for(hold,buy,sell) (logits->softmax)
        self.critic_head = nn.Linear(hidden_dim, 1) #critic's output,scalar V(s);the estimated future reward from this state

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.actor_head(features) #scores for each action
        value  = self.critic_head(features) #estimated value of the current state
        return logits, value

    def get_action(self, x: torch.Tensor): 
        """Sample action + return log_prob and value for training."""
        logits, value = self.forward(x)
        dist   = torch.distributions.Categorical(logits=logits) #turns logits into a probability distribution
        action = dist.sample() #pick an action in random weighted by the probabilities (exploration)
        return action, dist.log_prob(action), value 


# ── Strategy (Live Inference) ─────────────────────────────────────────────────

class A2CStrategy(BaseStrategy):
    """
    A2C (Advantage Actor-Critic) strategy.
    Loads a pre-trained ActorCriticNetwork and runs greedy inference.
    Train with: python3 -m agents.train_a2c
    """

    def __init__(self, symbol: str = "SOLUSDT",
                 model_path: str = "models/a2c_actor_critic.pth",
                 quantity_per_trade: float = 5.0):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.market_data = MarketData(window_size=10)
        self.quantity    = quantity_per_trade
        self.model_loaded = False
        self.current_inventory = 0.0
        self.observation: Optional[np.ndarray] = None

        self.net = ActorCriticNetwork(state_dim=4, action_dim=3).to(self.device)

        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.net.eval()
            self.model_loaded = True
            print(f"[A2C] Model loaded from {model_path}")
        else:
            print(f"[A2C] No model found at {model_path}. Run 'python3 -m agents.train_a2c' first.")

    def _build_observation(self) -> Optional[np.ndarray]:
        """Build normalised state vector — identical to training environment."""
        if len(self.market_data.prices) < 3:
            return None

        b_qty = sum(b["quantity"] for b in self.market_data.bids[:3]) if self.market_data.bids else 0
        a_qty = sum(a["quantity"] for a in self.market_data.asks[:3]) if self.market_data.asks else 0
        imb   = (b_qty - a_qty) / (b_qty + a_qty) if (b_qty + a_qty) > 0 else 0.0

        norm_imb    = float(np.clip(imb, -1.0, 1.0))
        norm_spread = (self.market_data.spread / self.market_data.mid_price * 10000) / 10.0 \
                      if self.market_data.mid_price else 0.0
        norm_dev    = (self.market_data.get_price_deviation() or 0.0) / 2.0
        norm_inv    = self.current_inventory / 50.0

        return np.array([norm_imb, norm_spread, norm_dev, norm_inv], dtype=np.float32)

    # ── BaseStrategy Interface ────────────────────────────────────────────────

    def update_data(self, market_state: Dict) -> None:
        self.market_data.update(market_state)
        self.observation = self._build_observation()

    def compute_signal(self) -> StrategySignal:
        if self.observation is None or not self.model_loaded:
            return StrategySignal(None, 0, 0, "A2C: warming up...", {})

        state_t = torch.tensor(self.observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.net(state_t)
            # Greedy during live trading (no sampling)
            action_idx = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1)[0, action_idx].item()

        mid    = int(self.market_data.mid_price * 100)
        offset = int(self.market_data.spread * 100 / 2)

        if action_idx == 1:   # BUY
            self.current_inventory += self.quantity
            return StrategySignal(
                Side.BUY, mid - offset, self.quantity,
                f"A2C BUY (conf: {confidence:.2f})",
                {"action": "BUY", "value": value.item(), "confidence": confidence}
            )
        elif action_idx == 2:  # SELL
            self.current_inventory -= self.quantity
            return StrategySignal(
                Side.SELL, mid + offset, self.quantity,
                f"A2C SELL (conf: {confidence:.2f})",
                {"action": "SELL", "value": value.item(), "confidence": confidence}
            )

        return StrategySignal(None, 0, 0, f"A2C HOLD (conf: {confidence:.2f})",
                              {"action": "HOLD", "value": value.item()})

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "A2C Agent",
            "model_loaded": self.model_loaded,
            "inventory": self.current_inventory,
            "last_obs": self.observation.tolist() if self.observation is not None else None,
        }
