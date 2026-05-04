import numpy as np
from typing import Dict, Any, Optional

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

from ..interfaces import BaseStrategy, StrategySignal
from ..types import Side


class PPOStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str = "SOLUSDT",
        policy_model_path: str = "agents/ppo/models_checkpoints/ppo_policy_FINAL.keras",
        quantity_per_trade: float = 5.0,
        window_size: int = 30,
    ):
        self.symbol = symbol
        self.quantity = quantity_per_trade
        self.window_size = window_size

        self.closes = []
        self.highs = []
        self.lows = []
        self.volumes = []

        self.inventory = 0
        self.max_inventory = 10
        self.max_spread_pct = 0.005

        self.next_side_to_quote = Side.BUY
        self.model_loaded = False
        self.policy_network = None

        if not _TF_AVAILABLE:
            print("PPO: TensorFlow not installed. Running in HOLD mode.")
            return

        print(f"Loading PPO Policy Model from {policy_model_path}...")
        try:
            self.policy_network = tf.keras.models.load_model(policy_model_path, compile=False)
            self.model_loaded = True
            print("PPO model loaded successfully.")
        except Exception as e:
            print(f"PPO: Failed to load model ({e}). Running in HOLD mode.")

    def update_data(self, market_state: Dict) -> None:
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

    def _get_observation(self) -> Optional[np.ndarray]:
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
        norm_inventory = self.inventory / self.max_inventory
        obs = np.concatenate([norm_close, norm_high, norm_low, norm_volume, [norm_inventory]])
        return obs.astype(np.float32)

    def compute_signal(self) -> StrategySignal:
        if not self.model_loaded:
            return StrategySignal(side=None, price=0.0, quantity=0.0, reason="PPO model offline")

        obs = self._get_observation()
        if obs is None:
            return StrategySignal(side=None, price=0.0, quantity=0.0, reason="Warming up buffer...")

        obs_batch = np.expand_dims(obs, axis=0)
        raw_output = self.policy_network(obs_batch, training=False).numpy()[0]
        action_ask = float(raw_output[0])
        action_bid = float(raw_output[1])

        ask_distance_pct = (action_ask + 1.0) / 2.0 * self.max_spread_pct
        bid_distance_pct = (action_bid + 1.0) / 2.0 * self.max_spread_pct
        current_price = self.closes[-1]
        agent_ask_price = current_price * (1.0 + ask_distance_pct)
        agent_bid_price = current_price * (1.0 - bid_distance_pct)

        if self.next_side_to_quote == Side.BUY:
            self.next_side_to_quote = Side.SELL
            if self.inventory < self.max_inventory:
                return StrategySignal(
                    side=Side.BUY,
                    price=agent_bid_price,
                    quantity=self.quantity,
                    reason=f"AI MM Bid (dist: {bid_distance_pct*100:.3f}%)",
                    metadata={"ai_action_raw": action_bid, "inventory": self.inventory},
                )
            return StrategySignal(side=None, price=0.0, quantity=0.0, reason="Inventory Full (Hold)")
        else:
            self.next_side_to_quote = Side.BUY
            if self.inventory > -self.max_inventory:
                return StrategySignal(
                    side=Side.SELL,
                    price=agent_ask_price,
                    quantity=self.quantity,
                    reason=f"AI MM Ask (dist: {ask_distance_pct*100:.3f}%)",
                    metadata={"ai_action_raw": action_ask, "inventory": self.inventory},
                )
            return StrategySignal(side=None, price=0.0, quantity=0.0, reason="Inventory Empty (Hold)")

    def analyze(self, state_dict: Dict) -> Dict:
        """Adapter for TradingAgent which calls strategy.analyze(state)."""
        self.update_data(state_dict)
        signal = self.compute_signal()
        # PPO price is a real price (float), convert to integer cents
        price_cents = int(round(signal.price * 100)) if signal.price else 0
        return {
            "action": signal.side.value if signal.side else None,
            "side": signal.side,
            "price": price_cents,
            "quantity": signal.quantity,
            "reason": signal.reason,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "PPO Market Maker Strategy",
            "tf_available": _TF_AVAILABLE,
            "model_loaded": self.model_loaded,
            "current_inventory": self.inventory,
            "max_inventory": self.max_inventory,
        }
