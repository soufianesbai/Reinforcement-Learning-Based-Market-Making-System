"""
Trading strategy implementation - Mean reversion
"""

from typing import Dict
from .types import Side
from .market_data import MarketData


class TradingStrategy:
    """Mean reversion strategy"""
    
    def __init__(self, symbol: str = "SOLUSDT", quantity_per_trade: float = 5.0):
        self.symbol = symbol
        self.quantity_per_trade = quantity_per_trade
        self.market_data = MarketData()
        self.threshold = 0.01 # 1.5% 
        
    def analyze(self, state_dict: Dict) -> Dict:
        """Analyze market and return decision"""
        self.market_data.update(state_dict)
        
        result = {
            "action": None,
            "side": None,
            "price": None,
            "quantity": self.quantity_per_trade,
            "reason": "",
        }
        
        # Need at least 3 prices for SMA
        if len(self.market_data.prices) < 3:
            num_prices = len(self.market_data.prices)
            result["reason"] = f"Collecting data... ({num_prices}/3) - Prices: {list(self.market_data.prices)}"
            return result
        
        sma = self.market_data.get_sma()
        deviation = self.market_data.get_price_deviation()
        
        if deviation is None or sma is None:
            result["reason"] = "Insufficient data"
            return result
        
        # Mean reversion strategy
        if deviation < -self.threshold:
            result["action"] = "BUY"
            result["side"] = Side.BUY
            result["price"] = int(self.market_data.bid_price * 100)
            result["reason"] = f"Price {deviation:.2f}% below SMA ({sma:.2f})"
            
        elif deviation > self.threshold:
            result["action"] = "SELL"
            result["side"] = Side.SELL
            result["price"] = int(self.market_data.ask_price * 100)
            result["reason"] = f"Price {deviation:.2f}% above SMA ({sma:.2f})"
        else:
            result["reason"] = f"Deviation: {deviation:.2f}% (threshold: ±{self.threshold}%) - {len(self.market_data.prices)} prices"
        
        result["sma"] = sma
        result["deviation"] = deviation
        return result
