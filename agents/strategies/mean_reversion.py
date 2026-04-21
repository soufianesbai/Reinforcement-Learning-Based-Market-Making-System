"""
Trading strategy implementation - Mean reversion
"""

from typing import Dict, Optional
from collections import deque
import statistics
from ..trading_types import Side
from ..market_data import MarketData


class MeanReversionStrategy:
    """Mean reversion strategy"""
    
    def __init__(self, symbol: str = "SOLUSDT", quantity_per_trade: float = 0.1):
        self.symbol = symbol
        self.quantity_per_trade = quantity_per_trade
        self.market_data = MarketData()
        self.threshold = 0.01  # 1% deviation
        
        # Strategy-specific price history for SMA
        self.prices = deque(maxlen=10)
        
    def _calculate_sma(self) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(self.prices) < 3:
            return None
        return statistics.mean(list(self.prices))
    
    def _calculate_deviation(self) -> Optional[float]:
        """Calculate price deviation from SMA"""
        if len(self.prices) < 3:
            return None
        sma = self._calculate_sma()
        if sma == 0:
            return None
        current_price = self.prices[-1]
        return (current_price - sma) / sma * 100
        
    def analyze(self, state_dict: Dict) -> Dict:
        """Analyze market and return decision"""
        self.market_data.update(state_dict)
        
        # Add price to strategy history
        self.prices.append(self.market_data.mid_price)
        
        result = {
            "action": None,
            "side": None,
            "price": None,
            "quantity": self.quantity_per_trade,
            "reason": "",
        }
        
        # Need at least 3 prices for SMA
        if len(self.prices) < 3:
            num_prices = len(self.prices)
            result["reason"] = f"Collecting data... ({num_prices}/3)"
            return result
        
        sma = self._calculate_sma()
        deviation = self._calculate_deviation()
        
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
            result["reason"] = f"Deviation: {deviation:.2f}% (threshold: ±{self.threshold*100:.1f}%)"
        
        return result
