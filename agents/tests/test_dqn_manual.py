"""
Manual validation script for the DQN strategy.

The script uses a deterministic mock Q-network so we can verify that the
strategy follows the BaseStrategy interface and maps Q-values to BUY/SELL/HOLD
correctly.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())

from agents.strategies.dqn_agent_faster import DQNStrategy
from agents.types import Side


class FixedQNetwork(nn.Module):
    def __init__(self, q_values):
        super().__init__()
        self.q_values = q_values

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        q = torch.tensor(self.q_values, dtype=torch.float32, device=x.device)
        return q.unsqueeze(0).repeat(batch_size, 1)


def build_market_state(mid_price_cents: int = 10000):
    return {
        "mid_price": mid_price_cents,
        "bid_price": mid_price_cents - 10,
        "ask_price": mid_price_cents + 10,
        "spread": 20,
        "bids": [
            {"price": mid_price_cents - 10, "quantity": 12.0},
            {"price": mid_price_cents - 20, "quantity": 9.0},
            {"price": mid_price_cents - 30, "quantity": 6.0},
        ],
        "asks": [
            {"price": mid_price_cents + 10, "quantity": 8.0},
            {"price": mid_price_cents + 20, "quantity": 7.0},
            {"price": mid_price_cents + 30, "quantity": 5.0},
        ],
    }


def warmup_strategy(strategy: DQNStrategy):
    state = build_market_state()
    for _ in range(3):
        strategy.update_data(state)


def test_buy_path():
    strategy = DQNStrategy(model_path="/tmp/no_model_needed.pth", device="cpu")
    strategy.model_loaded = True
    strategy.policy_net = FixedQNetwork([0.1, 2.5, -1.0]).to(strategy.device)
    strategy.target_net = strategy.policy_net
    strategy.current_inventory = 0.0
    warmup_strategy(strategy)

    signal = strategy.compute_signal()
    assert signal.side == Side.BUY, f"Expected BUY, got {signal.side}"
    assert signal.price == 9990, f"Expected bid price in cents, got {signal.price}"
    assert signal.quantity == 5.0, f"Expected default quantity 5.0, got {signal.quantity}"
    print("BUY path validated")


def test_sell_path():
    strategy = DQNStrategy(model_path="/tmp/no_model_needed.pth", device="cpu")
    strategy.model_loaded = True
    strategy.policy_net = FixedQNetwork([0.1, -0.2, 3.1]).to(strategy.device)
    strategy.target_net = strategy.policy_net
    strategy.current_inventory = 5.0
    warmup_strategy(strategy)

    signal = strategy.compute_signal()
    assert signal.side == Side.SELL, f"Expected SELL, got {signal.side}"
    assert signal.price == 10010, f"Expected ask price in cents, got {signal.price}"
    print("SELL path validated")


def test_inventory_guard():
    strategy = DQNStrategy(model_path="/tmp/no_model_needed.pth", device="cpu")
    strategy.model_loaded = True
    strategy.policy_net = FixedQNetwork([0.0, -1.0, 5.0]).to(strategy.device)
    strategy.target_net = strategy.policy_net
    strategy.current_inventory = 0.0
    warmup_strategy(strategy)

    signal = strategy.compute_signal()
    assert signal.side is None, f"Expected HOLD when inventory is empty, got {signal.side}"
    assert signal.reason == "No inventory to sell (hold)", f"Unexpected reason: {signal.reason}"
    print("Inventory guard validated")


if __name__ == "__main__":
    test_buy_path()
    test_sell_path()
    test_inventory_guard()
    print("\nAll DQN checks passed.")