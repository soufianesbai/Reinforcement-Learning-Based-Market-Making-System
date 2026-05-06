"""
Dreamer v3 agent for crypto trading.
Model-based RL: learns a world model, then trains actor-critic in imagination.
Inference: python -c "from agents.strategies.dreamer_agent import DreamerStrategy"
Training: python -m agents.train_dreamer
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..interfaces import BaseStrategy, StrategySignal
from ..types import Side
from ..market_data import MarketData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -20.0, 20.0)
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# Neural network modules
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, obs_dim: int = 121, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical latent variables."""

    def __init__(self, action_dim: int = 3, embed_dim: int = 256,
                 deter_dim: int = 512, stoch_classes: int = 32,
                 stoch_categories: int = 32):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_classes = stoch_classes
        self.stoch_categories = stoch_categories
        self.stoch_dim = stoch_classes * stoch_categories  # 1024

        # GRU input projection: cat(z_prev, action_onehot) -> deter_dim
        gru_input_dim = self.stoch_dim + action_dim  # 1027
        self.gru_input = nn.Sequential(
            nn.Linear(gru_input_dim, deter_dim),
            nn.LayerNorm(deter_dim),
            nn.SiLU(),
        )
        self.gru_cell = nn.GRUCell(deter_dim, deter_dim)

        # Prior: h -> logits
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, self.stoch_dim),
        )

        # Posterior: cat(h, embed) -> logits
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, self.stoch_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def _sample_straight_through(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-through categorical sampling. logits: [B, 32, 32]"""
        probs = F.softmax(logits, dim=-1)
        indices = torch.argmax(probs, dim=-1)  # [B, 32]
        sample = F.one_hot(indices, self.stoch_classes).float()  # [B, 32, 32]
        z = (sample + probs - probs.detach()).flatten(-2)  # [B, 1024]
        return z

    def _step_gru(self, h: torch.Tensor, z: torch.Tensor,
                  action: torch.Tensor) -> torch.Tensor:
        """Run one GRU step. action: one-hot [B, action_dim]"""
        inp = torch.cat([z, action], dim=-1)
        inp = self.gru_input(inp)
        h_new = self.gru_cell(inp, h)
        return h_new

    def observe_step(self, h: torch.Tensor, z: torch.Tensor,
                     action: torch.Tensor, embed: torch.Tensor):
        """One step with observation (posterior). action: one-hot [B, 3]"""
        h_new = self._step_gru(h, z, action)

        prior_logits = self.prior_net(h_new).view(-1, self.stoch_categories, self.stoch_classes)
        post_logits = self.posterior_net(
            torch.cat([h_new, embed], dim=-1)
        ).view(-1, self.stoch_categories, self.stoch_classes)

        z_new = self._sample_straight_through(post_logits)
        return h_new, z_new, prior_logits, post_logits

    def imagine_step(self, h: torch.Tensor, z: torch.Tensor,
                     action: torch.Tensor):
        """One step without observation (prior only). action: one-hot [B, 3]"""
        h_new = self._step_gru(h, z, action)
        prior_logits = self.prior_net(h_new).view(-1, self.stoch_categories, self.stoch_classes)
        z_new = self._sample_straight_through(prior_logits)
        return h_new, z_new


class Decoder(nn.Module):
    def __init__(self, state_dim: int = 1536, obs_dim: int = 121):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, obs_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class RewardPredictor(nn.Module):
    def __init__(self, state_dim: int = 1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ContinuePredictor(nn.Module):
    def __init__(self, state_dim: int = 1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Actor(nn.Module):
    def __init__(self, state_dim: int = 1536, action_dim: int = 3, unimix: float = 0.01):
        super().__init__()
        self.action_dim = action_dim
        self.unimix = unimix
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns mixed probabilities with unimix exploration."""
        logits = self.net(state)
        probs = (1.0 - self.unimix) * F.softmax(logits, dim=-1) + \
                self.unimix * (1.0 / self.action_dim)
        return probs

    def get_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int = 1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---------------------------------------------------------------------------
# Strategy (inference)
# ---------------------------------------------------------------------------

class DreamerStrategy(BaseStrategy):
    """
    Dreamer v3 trading strategy.
    Loads pre-trained world model + actor and runs greedy inference.
    Train with: python3 -m agents.train_dreamer
    """

    def __init__(self, symbol: str = "SOLUSDT",
                 model_path: str = "models/dreamer_agent.pth",
                 quantity_per_trade: float = 5.0,
                 window_size: int = 30):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.symbol = symbol
        self.quantity = quantity_per_trade
        self.window_size = window_size
        self.market_data = MarketData()
        self.model_loaded = False
        self.current_inventory = 0.0

        # Own price/volume lists (PPO pattern, avoids MarketData bugs)
        self.closes = []
        self.highs = []
        self.lows = []
        self.volumes = []

        # Build modules
        self.encoder = Encoder().to(self.device)
        self.rssm = RSSM().to(self.device)
        self.actor = Actor().to(self.device)

        # RSSM persistent state. prev_action is the one-hot action that LED
        # to the current observation; on the very first step there is no
        # previous action, so we use a zero vector as a "no-action" marker.
        self.h, self.z = self.rssm.initial_state(1, self.device)
        self.prev_action = torch.zeros(1, 3, device=self.device)

        self._load_model(model_path)

    def _load_model(self, path: str):
        if not os.path.exists(path):
            print(f"[Dreamer] Model not found at {path}. Run 'python3 -m agents.train_dreamer' first.")
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.rssm.load_state_dict(checkpoint["rssm"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.encoder.eval()
        self.rssm.eval()
        self.actor.eval()
        self.model_loaded = True
        print(f"[Dreamer] Model loaded successfully from {path}")

    def _build_observation(self) -> Optional[np.ndarray]:
        if len(self.closes) < self.window_size:
            return None

        close_w = np.array(self.closes[-self.window_size:])
        high_w = np.array(self.highs[-self.window_size:])
        low_w = np.array(self.lows[-self.window_size:])
        volume_w = np.array(self.volumes[-self.window_size:])

        base = close_w[-1] + 1e-8
        norm_close = (close_w / base) - 1.0
        norm_high = (high_w / base) - 1.0
        norm_low = (low_w / base) - 1.0
        vol_mean = volume_w.mean() + 1e-8
        norm_volume = (volume_w / vol_mean) - 1.0
        norm_inventory = self.current_inventory / 10.0

        obs = np.concatenate([norm_close, norm_high, norm_low, norm_volume, [norm_inventory]])
        return obs.astype(np.float32)

    # --- BaseStrategy interface ---

    def update_data(self, market_state: Dict) -> None:
        self.market_data.update(market_state)

        raw_mid = market_state.get("mid_price", 0)
        current_price = raw_mid / 100.0 if raw_mid > 0 else 0.0
        if current_price == 0.0:
            return

        self.closes.append(current_price)
        self.highs.append(current_price)
        self.lows.append(current_price)
        self.volumes.append(1.0)

        if len(self.closes) > self.window_size:
            self.closes.pop(0)
            self.highs.pop(0)
            self.lows.pop(0)
            self.volumes.pop(0)

    def compute_signal(self) -> StrategySignal:
        obs = self._build_observation()
        if obs is None or not self.model_loaded:
            return StrategySignal(None, 0, 0, "Dreamer: warming up...", {})

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            embed = self.encoder(obs_t)
            self.h, self.z, _, _ = self.rssm.observe_step(
                self.h, self.z, self.prev_action, embed
            )
            state = torch.cat([self.h, self.z], dim=-1)
            probs = self.actor(state)
            action_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action_idx].item()

            # Update prev_action for next step
            self.prev_action = F.one_hot(
                torch.tensor([action_idx], device=self.device), num_classes=3
            ).float()

        mid = self.market_data.mid_price * 100
        offset = self.market_data.spread * 100 / 2

        if action_idx == 1:  # BUY
            self.current_inventory += self.quantity
            return StrategySignal(
                Side.BUY, mid - offset, self.quantity,
                f"Dreamer BUY (conf: {confidence:.2f})",
                {"action": "BUY", "confidence": confidence},
            )
        elif action_idx == 2:  # SELL
            self.current_inventory -= self.quantity
            return StrategySignal(
                Side.SELL, mid + offset, self.quantity,
                f"Dreamer SELL (conf: {confidence:.2f})",
                {"action": "SELL", "confidence": confidence},
            )

        return StrategySignal(
            None, 0, 0, f"Dreamer HOLD (conf: {confidence:.2f})",
            {"action": "HOLD", "confidence": confidence},
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "Dreamer v3 Agent",
            "model_loaded": self.model_loaded,
            "inventory": self.current_inventory,
            "rssm_h_norm": float(self.h.norm().item()) if self.h is not None else 0,
        }
