"""
Market data handling and technical analysis
"""

from collections import deque
from typing import Dict, Optional
import statistics


class MarketData:
    """Stores market data from C++"""
    
    def __init__(self, window_size: int = 10):
        self.prices = deque(maxlen=window_size)
        self.bids = []
        self.asks = []
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.mid_price = 0.0
        self.spread = 0.0
        
    def update(self, state_dict: Dict):
        """Update from market state"""
        self.bid_price = state_dict.get("bid_price", 0) / 100.0
        self.ask_price = state_dict.get("ask_price", 0) / 100.0
        self.mid_price = state_dict.get("mid_price", 0) / 100.0
        self.spread = state_dict.get("spread", 0) / 100.0
        self.bids = state_dict.get("bids", [])
        self.asks = state_dict.get("asks", [])
        
        # Add to price history
        self.prices.append(self.mid_price)
        
    def get_sma(self) -> Optional[float]:
        """Get Simple Moving Average"""
        if len(self.prices) < 3:
            return None
        return statistics.mean(list(self.prices))
    
    def get_price_deviation(self) -> Optional[float]:
        """Get price deviation from SMA"""
        if len(self.prices) < 3:
            return None
        sma = self.get_sma()
        if sma == 0:
            return None
        current_price = self.prices[-1]
        return (current_price - sma) / sma * 100
