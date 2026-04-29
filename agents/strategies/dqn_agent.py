"""
Baseline DQN trading strategy.

This module contains the readable, reference implementation of the Deep
Q-Network strategy used by the trading agent. It is intentionally written as
the clear and maintainable version of the strategy so it can serve as the
baseline for experiments, debugging, and comparison with faster variants.

The strategy follows the standard `BaseStrategy` interface and uses a dueling
Q-network with the following market inputs:

- order book imbalance
- spread pressure
- price deviation from the recent average
- short-term momentum
- realized short-window volatility
- normalized inventory

Why this structure matters:

- **Dueling architecture**: separates the value of being in a state from the
    advantage of BUY/SELL actions. In trading, HOLD is often the best action, so
    value/advantage separation helps the network learn that better.
- **LayerNorm + SiLU**: stabilizes learning on noisy market features and helps
    avoid brittle behavior when price regimes change.
- **Target network**: reduces bootstrap instability by updating a secondary
    network less frequently.
- **Replay buffer**: reuses prior experiences so the model can learn from more
    than the most recent sample and avoid overfitting to the latest regime.
- **Inventory-aware trading**: the strategy blocks unsafe sells when inventory
    is empty and caps buys when inventory is already too large.

This file is the "normal" version. It favors clarity and correctness over raw
training throughput. The faster implementation lives in
`agents/strategies/dqn_agent_faster.py` and keeps the same trading logic while
reducing Python overhead and improving GPU utilization.
"""

import os
import random
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..interfaces import BaseStrategy, StrategySignal
from ..market_data import MarketData
from ..types import Side


class TradingQNetwork(nn.Module):
    """Dueling DQN network tuned for noisy trading features."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, int, int] = (256, 256, 128)):
        super().__init__()
        h1, h2, h3 = hidden_dims

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Dropout(p=0.10),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Dropout(p=0.10),
            nn.Linear(h2, h3),
            nn.LayerNorm(h3),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(h3, 1)
        self.advantage_head = nn.Linear(h3, output_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / np.sqrt(max(1, fan_in))
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.backbone(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.buffer = deque(maxlen=capacity)
        self.state_dim = int(state_dim)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((np.asarray(state, dtype=np.float32), int(action), float(reward), np.asarray(next_state, dtype=np.float32), bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )


class DQNStrategy(BaseStrategy):
    """
    Deep Q-Network trading strategy.

    The strategy uses PyTorch and automatically selects CUDA when available.
    Inference is greedy by default; epsilon-greedy exploration is enabled only
    during training mode.
    """

    def __init__(
        self,
        symbol: str = "SOLUSDT",
        model_path: str = "models/dqn_agent.pth",
        quantity_per_trade: float = 5.0,
        window_size: int = 10,
        max_inventory: float = 10.0,
        gamma: float = 0.995,
        learning_rate: float = 3e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 256,
        replay_capacity: int = 200_000,
        target_update_steps: int = 1_000,
        device: Optional[Any] = None,
    ):
        self.symbol = symbol
        self.quantity = quantity_per_trade
        self.window_size = window_size
        self.max_inventory = max_inventory
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.model_path = model_path

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The normal version stays intentionally simpler and slightly slower.
        # We keep CUDA available, but we do not use AMP so it behaves as the
        # clearer baseline implementation instead of the throughput-optimized one.
        self.use_amp = False

        self.market_data = MarketData(window_size=window_size)
        self.current_inventory = 0.0
        self.training_mode = False
        self.epsilon = epsilon_end
        self.step_count = 0

        self.state_dim = 6
        self.action_dim = 3
        self.last_observation: Optional[np.ndarray] = None
        self.last_q_values: Optional[np.ndarray] = None
        self.last_mid_price: float = 0.0

        self.policy_net = TradingQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = TradingQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.replay_buffer = ReplayBuffer(replay_capacity, self.state_dim)

        self.model_loaded = False
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            print(f"[!] DQN checkpoint not found at {model_path}. The strategy will still run with its current weights.")
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            optimizer_state = None

            if isinstance(checkpoint, dict):
                if "policy_net" in checkpoint and isinstance(checkpoint["policy_net"], dict):
                    policy_state = checkpoint["policy_net"]
                    target_state = checkpoint.get("target_net")
                    optimizer_state = checkpoint.get("optimizer")
                elif "policy_state_dict" in checkpoint:
                    policy_state = checkpoint["policy_state_dict"]
                    target_state = checkpoint.get("target_state_dict")
                    optimizer_state = checkpoint.get("optimizer_state_dict")
                elif "state_dict" in checkpoint:
                    policy_state = checkpoint["state_dict"]
                    target_state = checkpoint.get("target_state_dict")
                    optimizer_state = checkpoint.get("optimizer_state_dict")
                else:
                    policy_state = checkpoint
                    target_state = None
            else:
                policy_state = checkpoint
                target_state = None

            self.policy_net.load_state_dict(policy_state)
            if target_state is not None:
                self.target_net.load_state_dict(target_state)
            else:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if optimizer_state is not None:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except Exception:
                    pass

            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            self.policy_net.train(self.training_mode)
            self.target_net.eval()
            self.model_loaded = True
            print(f"[*] DQN model loaded from {model_path} on {self.device}")
        except Exception as exc:
            print(f"[!] Failed to load DQN model from {model_path}: {exc}")

    def save_model(self, model_path: Optional[str] = None) -> None:
        path = model_path or self.model_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "gamma": self.gamma,
                    "learning_rate": self.learning_rate,
                    "epsilon_end": self.epsilon_end,
                    "batch_size": self.batch_size,
                },
            },
            path,
        )

    def set_training_mode(self, enabled: bool) -> None:
        self.training_mode = enabled
        self.policy_net.train(enabled)
        self.target_net.eval()
        self.epsilon = self.epsilon_start if enabled else self.epsilon_end

    def _build_observation(self) -> Optional[np.ndarray]:
        if len(self.market_data.prices) < 3 or self.market_data.mid_price <= 0:
            return None

        bids = self.market_data.bids[:3] if self.market_data.bids else []
        asks = self.market_data.asks[:3] if self.market_data.asks else []
        bid_qty = sum(float(b.get("quantity", 0.0)) for b in bids)
        ask_qty = sum(float(a.get("quantity", 0.0)) for a in asks)
        imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else 0.0

        spread_feature = (
            (self.market_data.spread / self.market_data.mid_price * 10000.0) / 10.0
            if self.market_data.mid_price
            else 0.0
        )
        deviation_feature = (self.market_data.get_price_deviation() or 0.0) / 2.0

        current_mid = self.market_data.mid_price
        momentum_feature = 0.0
        if self.last_mid_price > 0.0:
            momentum_feature = ((current_mid - self.last_mid_price) / self.last_mid_price * 100.0) / 2.0

        prices = np.asarray(self.market_data.prices, dtype=np.float32)
        if len(prices) >= 2:
            returns = np.diff(prices) / np.maximum(prices[:-1], 1e-8)
            volatility_feature = float(np.std(returns) * 10.0)
        else:
            volatility_feature = 0.0

        inventory_feature = self.current_inventory / self.max_inventory if self.max_inventory else 0.0

        return np.array(
            [
                np.clip(imbalance, -1.0, 1.0),
                np.clip(spread_feature, -1.0, 1.0),
                np.clip(deviation_feature, -1.0, 1.0),
                np.clip(momentum_feature, -1.0, 1.0),
                np.clip(volatility_feature, -1.0, 1.0),
                np.clip(inventory_feature, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def update_data(self, market_state: Dict) -> None:
        self.last_mid_price = self.market_data.mid_price
        self.market_data.update(market_state)
        self.last_observation = self._build_observation()

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _select_action(self, obs: np.ndarray) -> Tuple[int, np.ndarray]:
        with torch.no_grad():
            q_values = self.policy_net(self._tensor(obs)).squeeze(0)
        q_np = q_values.detach().cpu().numpy().astype(np.float32)
        return int(torch.argmax(q_values).item()), q_np

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def optimize_model(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = self._tensor(states)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = self._tensor(next_states)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
            loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.step_count += 1
        if self.step_count % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.training_mode and self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return float(loss.detach().cpu().item())

    def compute_signal(self) -> StrategySignal:
        if self.last_observation is None:
            return StrategySignal(side=None, price=0.0, quantity=0.0, reason="Warming up buffer...", metadata={})

        if not self.model_loaded and not self.training_mode:
            return StrategySignal(side=None, price=0.0, quantity=0.0, reason="DQN model offline", metadata={})

        if self.training_mode and random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
            with torch.no_grad():
                q_values = self.policy_net(self._tensor(self.last_observation)).squeeze(0)
            q_np = q_values.detach().cpu().numpy().astype(np.float32)
        else:
            action_idx, q_np = self._select_action(self.last_observation)

        mid_price = self.market_data.mid_price
        bid_price = self.market_data.bid_price if self.market_data.bid_price > 0 else mid_price
        ask_price = self.market_data.ask_price if self.market_data.ask_price > 0 else mid_price
        bid_cents = int(round(bid_price * 100))
        ask_cents = int(round(ask_price * 100))

        self.last_q_values = q_np

        if action_idx == 1:
            if self.current_inventory >= self.max_inventory:
                return StrategySignal(
                    side=None,
                    price=0.0,
                    quantity=0.0,
                    reason="Inventory cap reached (hold)",
                    metadata={"action": "BUY_BLOCKED", "inventory": self.current_inventory, "q_values": q_np.tolist()},
                )

            return StrategySignal(
                side=Side.BUY,
                price=bid_cents,
                quantity=self.quantity,
                reason="DQN BUY",
                metadata={"action": "BUY", "inventory": self.current_inventory, "epsilon": self.epsilon, "q_values": q_np.tolist()},
            )

        if action_idx == 2:
            if self.current_inventory <= 0.0:
                return StrategySignal(
                    side=None,
                    price=0.0,
                    quantity=0.0,
                    reason="No inventory to sell (hold)",
                    metadata={"action": "SELL_BLOCKED", "inventory": self.current_inventory, "q_values": q_np.tolist()},
                )

            return StrategySignal(
                side=Side.SELL,
                price=ask_cents,
                quantity=self.quantity,
                reason="DQN SELL",
                metadata={"action": "SELL", "inventory": self.current_inventory, "epsilon": self.epsilon, "q_values": q_np.tolist()},
            )

        return StrategySignal(
            side=None,
            price=0.0,
            quantity=0.0,
            reason="DQN HOLD",
            metadata={"action": "HOLD", "inventory": self.current_inventory, "epsilon": self.epsilon, "q_values": q_np.tolist()},
        )

    def analyze(self, state_dict: Dict) -> Dict:
        """Adapter for TradingAgent which calls strategy.analyze(state)."""
        self.update_data(state_dict)
        signal = self.compute_signal()
        # DQN price is already in integer cents (bid_cents / ask_cents)
        return {
            "action": signal.side.value if signal.side else None,
            "side": signal.side,
            "price": int(signal.price),
            "quantity": signal.quantity,
            "reason": signal.reason,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "DQN Agent",
            "device": str(self.device),
            "model_loaded": self.model_loaded,
            "training_mode": self.training_mode,
            "epsilon": self.epsilon,
            "inventory": self.current_inventory,
            "buffer_size": len(self.replay_buffer),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "last_observation": self.last_observation.tolist() if self.last_observation is not None else None,
            "last_q_values": self.last_q_values.tolist() if self.last_q_values is not None else None,
        }