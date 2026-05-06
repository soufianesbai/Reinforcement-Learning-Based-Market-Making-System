"""
Soft Actor-Critic (SAC) Market Making Agent.

Implements a proper market making policy: the agent learns to place
bid/ask quotes around the mid-price to capture the spread while managing
inventory risk.  Unlike the directional agents (RSI, Bollinger, DQN),
this agent never bets on price direction — it earns by being the
counterparty on both sides and keeping its inventory near zero.

References:
  Haarnoja et al., 2018 — Soft Actor-Critic (original SAC paper)
  Avellaneda & Stoikov, 2008 — analytical market making baseline
  Guéant, 2020 — Deep RL for Market Making

Action space  (2-dimensional, continuous):
  bid_spread ∈ [0, max_spread_pct] — how far below mid to quote bids
  ask_spread ∈ [0, max_spread_pct] — how far above mid to quote asks

State space (STATE_DIM = 15):
  inventory_norm, 10×log_return, volatility,
  buyer_frac, market_spread_bps, volume_ratio
"""

import os
import random
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..interfaces import BaseStrategy, StrategySignal
from ..market_data import MarketData
from ..types import Side

# ── Constants ─────────────────────────────────────────────────────────────────
STATE_DIM = 15
ACTION_DIM = 2
MAX_SPREAD_PCT = 0.005      # 0.5% max quote distance from mid
LOG_STD_MIN = -5
LOG_STD_MAX = 2
ALPHA_FIXED = 0.2           # fixed entropy coeff (auto-tuning unstable for tiny action spaces)


def build_sac_state(
    inventory: float,
    log_rets: np.ndarray,
    volatility: float,
    buyer_frac: float,
    market_spread_bps: float,
    vol_ratio: float,
    inventory_scale: float,
) -> np.ndarray:
    """Builds the normalized SAC state vector used by both training and live inference."""
    inv_scale = max(float(inventory_scale), 1e-6)
    returns = np.asarray(log_rets, dtype=np.float32)
    return np.array([
        np.clip(float(inventory) / inv_scale, -1.0, 1.0),
        *np.clip(returns * 100.0, -5.0, 5.0),
        np.clip(float(volatility) * 100.0, 0.0, 5.0),
        np.clip(float(buyer_frac), 0.0, 1.0),
        np.clip(float(market_spread_bps) / 10.0, 0.0, 5.0),
        np.clip(float(vol_ratio), 0.0, 5.0),
    ], dtype=np.float32)


# ── Networks ──────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """
    Stochastic policy: outputs (mean, log_std) of a Gaussian, then applies
    tanh squashing and rescales to [0, max_spread_pct].
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = _mlp(state_dim, hidden=256, out_dim=256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log-probability.
        Returns (action ∈ [0, max_spread_pct]², log_prob).
        """
        mean, log_std = self(state)
        std = log_std.exp()
        u = mean + std * torch.randn_like(std)
        # tanh squash → (-1, 1), then shift/scale → (0, max_spread_pct)
        a_raw = torch.tanh(u)
        action = (a_raw + 1.0) * 0.5 * MAX_SPREAD_PCT

        # Log-prob with Jacobian correction for tanh + affine transform
        log_prob = (
            -0.5 * ((u - mean) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
            - torch.log(1.0 - a_raw.pow(2) + 1e-6)
            - np.log(MAX_SPREAD_PCT / 2.0)
        ).sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self(state)
        return (torch.tanh(mean) + 1.0) * 0.5 * MAX_SPREAD_PCT


class Critic(nn.Module):
    """Twin Q-network — takes (state, action) and returns two Q-values."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.q1 = _mlp(state_dim + action_dim, out_dim=1)
        self.q2 = _mlp(state_dim + action_dim, out_dim=1)

    def forward(self, state: torch.Tensor, action: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(state, action)
        return torch.min(q1, q2)


# ── Replay buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 500_000):
        self.buf = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, s, a, r, s2, done):
        self.buf.append((
            np.asarray(s, np.float32), np.asarray(a, np.float32),
            float(r),
            np.asarray(s2, np.float32), float(done),
        ))

    def sample(self, batch: int, device: torch.device):
        items = random.sample(self.buf, batch)
        s, a, r, s2, d = zip(*items)
        to = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)
        return to(s), to(a), to(r).unsqueeze(1), to(s2), to(d).unsqueeze(1)


# ── SAC algorithm ─────────────────────────────────────────────────────────────

class SACAgent:
    """
    Off-policy SAC with automatic entropy tuning.
    Trains the actor to quote optimal bid/ask spreads that maximise
    cumulative mark-to-market PnL minus inventory risk.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        replay_capacity: int = 500_000,
        device: Optional[torch.device] = None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=lr)

        # Fixed entropy coefficient — auto-tuning diverges for action spaces this small
        self.log_alpha = torch.tensor(
            [float(np.log(ALPHA_FIXED))], dtype=torch.float32, device=self.device
        )

        self.buffer = ReplayBuffer(replay_capacity)
        self.updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                return self.actor.deterministic(s).squeeze(0).cpu().numpy()
            action, _ = self.actor.sample(s)
            return action.squeeze(0).cpu().numpy()

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, done = self.buffer.sample(self.batch_size, self.device)

        # ── Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            a2, log_p2 = self.actor.sample(s2)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_target = r + self.gamma * (1 - done) * (
                torch.min(q1_t, q2_t) - self.alpha * log_p2
            )

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        # ── Actor update ───────────────────────────────────────────────────
        a_new, log_p = self.actor.sample(s)
        q_new = self.critic.min_q(s, a_new)
        actor_loss = (self.alpha.detach() * log_p - q_new).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        # ── Soft update of target critic ───────────────────────────────────
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.updates += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": -log_p.mean().item(),
        }


# ── Live strategy (BaseStrategy interface) ────────────────────────────────────

class SACStrategy(BaseStrategy):
    """
    SAC market making strategy for live trading.

    At each tick, builds a state vector from the live order book and
    infers the optimal (bid_spread, ask_spread).  The tighter side
    determines whether the next market order is a BUY or SELL.
    This is a simplification of true limit-order market making
    compatible with the existing Binance testnet infrastructure.
    """

    def __init__(
        self,
        symbol: str = "SOLUSDT",
        model_path: str = "models/sac_agent.pth",
        quantity_per_trade: float = 0.1,
        history_len: int = 10,
        min_spread_asymmetry: float = 0.15,
        inventory_norm_scale: float = 5.0,
    ):
        self.symbol = symbol
        self.quantity = quantity_per_trade
        self.model_path = model_path
        self.history_len = history_len
        self.inventory_norm_scale = max(float(inventory_norm_scale), 1e-6)
        # Minimum asymmetry between bid/ask spread to trigger a trade
        self.min_asymmetry = min_spread_asymmetry

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor().to(self.device)
        self.actor.eval()
        self.model_loaded = False
        self._load_model()

        self.market_data = MarketData()
        self._mid_history: deque = deque(maxlen=history_len + 1)
        self._vol_history: deque = deque(maxlen=20)
        self._inventory: float = 0.0

    def set_runtime_context(self, inventory: Optional[float] = None, cash: Optional[float] = None) -> None:
        """Sync runtime fields from the orchestrator (TradingAgent)."""
        if inventory is not None:
            self._inventory = float(inventory)

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[!] SAC: model not found at {self.model_path}")
            return
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.actor.load_state_dict(ckpt["actor_state_dict"])
            self.actor.eval()
            self.model_loaded = True
            ep = ckpt.get("episode", "?")
            reward = ckpt.get("best_reward", "?")
            print(f"[*] SAC model loaded (ep {ep}, best_reward={reward})")
        except Exception as exc:
            print(f"[!] SAC: failed to load model: {exc}")

    def _build_state(self) -> Optional[np.ndarray]:
        if len(self._mid_history) < self.history_len + 1:
            return None
        mids = np.array(self._mid_history, dtype=np.float64)
        log_rets = np.diff(np.log(mids + 1e-10))[-self.history_len:]
        if len(log_rets) < self.history_len:
            return None

        volatility = float(log_rets.std()) if len(log_rets) > 1 else 0.0
        mid = float(self._mid_history[-1])
        bid = self.market_data.bid_price
        ask = self.market_data.ask_price
        market_spread_bps = ((ask - bid) / mid * 10000) if mid > 0 else 0.0

        bids = self.market_data.bids
        asks = self.market_data.asks
        bid_vol = sum(float(b.get("quantity", 0)) for b in bids[:5]) / 100.0
        ask_vol = sum(float(a.get("quantity", 0)) for a in asks[:5]) / 100.0
        total_vol = bid_vol + ask_vol
        buyer_frac = bid_vol / (total_vol + 1e-8)

        vol_history = list(self._vol_history) + [total_vol]
        mean_vol = float(np.mean(vol_history)) if vol_history else 1.0
        vol_ratio = total_vol / (mean_vol + 1e-8)

        return build_sac_state(
            inventory=self._inventory,
            log_rets=log_rets,
            volatility=volatility,
            buyer_frac=buyer_frac,
            market_spread_bps=market_spread_bps,
            vol_ratio=vol_ratio,
            inventory_scale=self.inventory_norm_scale,
        )

    def update_data(self, market_state: Dict) -> None:
        runtime_inventory = market_state.get("agent_inventory", market_state.get("inventory"))
        if runtime_inventory is not None:
            try:
                self._inventory = float(runtime_inventory)
            except (TypeError, ValueError):
                pass

        self.market_data.update(market_state)
        mid = self.market_data.mid_price
        if mid > 0:
            self._mid_history.append(mid)

        bids = market_state.get("bids", [])
        asks = market_state.get("asks", [])
        bv = sum(float(b.get("quantity", 0)) for b in bids[:5]) / 100.0
        av = sum(float(a.get("quantity", 0)) for a in asks[:5]) / 100.0
        self._vol_history.append(bv + av)

    def compute_signal(self) -> StrategySignal:
        if not self.model_loaded:
            return StrategySignal(
                side=None, price=0.0, quantity=0.0, reason="SAC model offline"
            )
        state = self._build_state()
        if state is None:
            n = len(self._mid_history)
            return StrategySignal(
                side=None, price=0.0, quantity=0.0,
                reason=f"SAC warming up ({n}/{self.history_len + 1})"
            )

        s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.deterministic(s_t).squeeze(0).cpu().numpy()

        bid_spread, ask_spread = float(action[0]), float(action[1])
        mid = self.market_data.mid_price
        bid_price = mid * (1.0 - bid_spread)
        ask_price = mid * (1.0 + ask_spread)

        # Determine side from spread asymmetry
        asymmetry = (ask_spread - bid_spread) / (ask_spread + bid_spread + 1e-8)
        if asymmetry > self.min_asymmetry:
            # Agent is quoting tighter on bid → wants to buy
            return StrategySignal(
                side=Side.BUY,
                price=int(round(bid_price * 100)),
                quantity=self.quantity,
                reason=f"SAC BUY  bid_s={bid_spread*100:.3f}% ask_s={ask_spread*100:.3f}%",
                metadata={"bid_spread": bid_spread, "ask_spread": ask_spread,
                          "asymmetry": asymmetry},
            )
        elif asymmetry < -self.min_asymmetry:
            # Agent is quoting tighter on ask → wants to sell
            return StrategySignal(
                side=Side.SELL,
                price=int(round(ask_price * 100)),
                quantity=self.quantity,
                reason=f"SAC SELL bid_s={bid_spread*100:.3f}% ask_s={ask_spread*100:.3f}%",
                metadata={"bid_spread": bid_spread, "ask_spread": ask_spread,
                          "asymmetry": asymmetry},
            )
        return StrategySignal(
            side=None, price=0.0, quantity=0.0,
            reason=f"SAC HOLD bid_s={bid_spread*100:.3f}% ask_s={ask_spread*100:.3f}%",
            metadata={"bid_spread": bid_spread, "ask_spread": ask_spread},
        )

    def analyze(self, state_dict: Dict) -> Dict:
        self.update_data(state_dict)
        signal = self.compute_signal()
        return {
            "action": signal.side.value if signal.side else None,
            "side": signal.side,
            "price": int(signal.price),
            "quantity": signal.quantity,
            "reason": signal.reason,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "SAC Market Maker",
            "model_loaded": self.model_loaded,
            "history": len(self._mid_history),
            "inventory": self._inventory,
            "device": str(self.device),
        }
