"""
Market data - stores current market state from C++
"""

from typing import Dict, Optional


class MarketData:
    """Stores current market state from C++"""
    
    def __init__(self):
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.mid_price = 0.0
        self.spread = 0.0
        self.bids = []
        self.asks = []
        self.last_price = 0.0
        
    def update(self, state_dict: Dict):
        """Update market state from C++"""
        self.bid_price = state_dict.get("bid_price", 0) / 100.0
        self.ask_price = state_dict.get("ask_price", 0) / 100.0
        self.mid_price = state_dict.get("mid_price", 0) / 100.0
        self.spread = state_dict.get("spread", 0) / 100.0
        self.bids = state_dict.get("bids", [])
        self.asks = state_dict.get("asks", [])
        self.last_price = self.mid_price
