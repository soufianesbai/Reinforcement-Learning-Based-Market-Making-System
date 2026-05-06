"""
DQN Trading Strategy — PyTorch inference wrapper.

Loads the pre-trained dqn_agent.pt checkpoint (state_dim=121, actions=3).
State vector: 4 features × 30 candles + 1 inventory = 121 dims.
  features: norm_close, norm_high, norm_low, norm_volume (normalized by current price)
Actions: 0=SELL, 1=HOLD, 2=BUY
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..interfaces import BaseStrategy, StrategySignal
from ..types import Side

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = str(ROOT / "models" / "dqn_agent.pt")

WINDOW = 30
FEATURES = 4        # close, high, low, volume
STATE_DIM = WINDOW * FEATURES + 1   # 121
MAX_INVENTORY = 3
DEFAULT_QTY = 5.0


class _QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNStrategy(BaseStrategy):
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_loaded = False

        self.policy_net = _QNet().to(self.device)
        self.target_net = _QNet().to(self.device)

        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(sd)
            self.target_net.load_state_dict(sd)
            self.policy_net.eval()
            self.target_net.eval()
            self.model_loaded = True
        else:
            print(f"[DQN] Warning: model not found at {model_path}, running random policy")

        self._window: deque[dict] = deque(maxlen=WINDOW)
        self._latest: Optional[dict] = None
        self.current_inventory: float = 0.0

    def update_data(self, market_state: Dict) -> None:
        mid_raw = market_state.get("mid_price", 0)
        bid_raw = market_state.get("bid_price", mid_raw)
        ask_raw = market_state.get("ask_price", mid_raw)

        mid = mid_raw / 100.0 if mid_raw > 500 else float(mid_raw)
        bid = bid_raw / 100.0 if bid_raw > 500 else float(bid_raw)
        ask = ask_raw / 100.0 if ask_raw > 500 else float(ask_raw)

        bids = market_state.get("bids") or []
        vol = float(sum(b.get("quantity", 0) for b in bids[:5])) + 1e-8

        self._window.append({"close": mid, "high": ask, "low": bid, "volume": vol})
        self._latest = market_state

    def _build_state(self) -> Optional[np.ndarray]:
        if len(self._window) < WINDOW:
            return None

        closes  = np.array([d["close"]  for d in self._window], dtype=np.float32)
        highs   = np.array([d["high"]   for d in self._window], dtype=np.float32)
        lows    = np.array([d["low"]    for d in self._window], dtype=np.float32)
        volumes = np.array([d["volume"] for d in self._window], dtype=np.float32)

        base = closes[-1] + 1e-8
        norm_close  = (closes  / base) - 1.0
        norm_high   = (highs   / base) - 1.0
        norm_low    = (lows    / base) - 1.0
        norm_volume = (volumes / (volumes.mean() + 1e-8)) - 1.0
        norm_inv    = np.array([self.current_inventory / MAX_INVENTORY], dtype=np.float32)

        return np.concatenate([norm_close, norm_high, norm_low, norm_volume, norm_inv])

    def compute_signal(self) -> StrategySignal:
        if self._latest is None:
            return StrategySignal(side=None, price=0, quantity=0, reason="No data yet")

        mid_raw = self._latest.get("mid_price", 0)
        bid_raw = self._latest.get("bid_price", mid_raw)
        ask_raw = self._latest.get("ask_price", mid_raw)

        state = self._build_state()
        if state is None:
            return StrategySignal(side=None, price=mid_raw, quantity=0,
                                  reason=f"Warming up ({len(self._window)}/{WINDOW})")

        if self.model_loaded:
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q = self.policy_net(t).squeeze(0).cpu().numpy()
            action = int(np.argmax(q))
        else:
            action = 1  # HOLD if no model

        if action == 2:  # BUY
            if self.current_inventory >= MAX_INVENTORY:
                return StrategySignal(side=None, price=bid_raw, quantity=0,
                                      reason="Max inventory reached (hold)")
            return StrategySignal(side=Side.BUY, price=bid_raw,
                                  quantity=DEFAULT_QTY, reason="DQN → BUY")

        if action == 0:  # SELL
            if self.current_inventory <= 0:
                return StrategySignal(side=None, price=ask_raw, quantity=0,
                                      reason="No inventory to sell (hold)")
            return StrategySignal(side=Side.SELL, price=ask_raw,
                                  quantity=DEFAULT_QTY, reason="DQN → SELL")

        return StrategySignal(side=None, price=mid_raw, quantity=0, reason="DQN → HOLD")

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model_loaded,
            "window_size": len(self._window),
            "inventory": self.current_inventory,
        }
