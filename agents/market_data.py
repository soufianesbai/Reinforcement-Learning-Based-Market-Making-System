"""
Market data - stores current market state from C++
"""

import statistics
from collections import deque
from typing import Dict, Optional


class MarketData:
    """Stores current market state from C++"""

    def __init__(self, window_size: int = 10):
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.mid_price = 0.0
        self.spread = 0.0
        self.bids = []
        self.asks = []
        self.last_price = 0.0
        self.prices: deque = deque(maxlen=max(window_size, 3))

    def update(self, state_dict: Dict):
        """Update market state from C++"""
        self.bid_price = state_dict.get("bid_price", 0) / 100.0
        self.ask_price = state_dict.get("ask_price", 0) / 100.0
        self.mid_price = state_dict.get("mid_price", 0) / 100.0
        self.spread = state_dict.get("spread", 0) / 100.0
        self.bids = state_dict.get("bids", [])
        self.asks = state_dict.get("asks", [])
        self.last_price = self.mid_price
        if self.mid_price > 0:
            self.prices.append(self.mid_price)

    def get_sma(self) -> Optional[float]:
        """Return the simple moving average of recent prices."""
        if len(self.prices) < 3:
            return None
        return statistics.mean(self.prices)

    def get_price_deviation(self) -> Optional[float]:
        """Return % deviation of current price from rolling SMA."""
        if len(self.prices) < 3:
            return None
        sma = statistics.mean(self.prices)
        if sma == 0:
            return None
        return (self.prices[-1] - sma) / sma * 100
