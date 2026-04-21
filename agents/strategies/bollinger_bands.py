"""
Bollinger Bands Trading Strategy
"""

from typing import Dict, Optional
from collections import deque
import statistics
from ..trading_types import Side
from ..market_data import MarketData


class BollingerBandsStrategy:
    """Bollinger Bands strategy - trades based on price hitting upper/lower bands"""
    
    def __init__(self, symbol: str = "SOLUSDT", quantity_per_trade: float = 0.1, period: int = 20, std_dev: float = 2.0):
        self.symbol = symbol
        self.quantity_per_trade = quantity_per_trade
        self.market_data = MarketData()
        self.period = period  # Period for SMA (default 20)
        self.std_dev_multiplier = std_dev  # Number of standard deviations (default 2)
        
        # Strategy-specific price history
        self.prices = deque(maxlen=period)
        
    def _calculate_bollinger_bands(self) -> Optional[Dict]:
        """Calculate Bollinger Bands"""
        if len(self.prices) < self.period:
            return None
        
        prices_list = list(self.prices)
        sma = statistics.mean(prices_list)
        
        # Calculate standard deviation
        variance = sum((x - sma) ** 2 for x in prices_list) / self.period
        std_dev = variance ** 0.5
        
        upper_band = sma + (std_dev * self.std_dev_multiplier)
        lower_band = sma - (std_dev * self.std_dev_multiplier)
        
        return {
            "sma": sma,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "std_dev": std_dev,
            "current_price": prices_list[-1]
        }
        
    def analyze(self, state_dict: Dict) -> Dict:
        """Analyze market using Bollinger Bands and return decision"""
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
        
        # Need enough data for Bollinger Bands
        if len(self.prices) < self.period:
            num_prices = len(self.prices)
            result["reason"] = f"Collecting data... ({num_prices}/{self.period})"
            return result
        
        bands = self._calculate_bollinger_bands()
        
        if bands is None:
            result["reason"] = "Insufficient data for Bollinger Bands"
            return result
        
        current_price = bands["current_price"]
        upper = bands["upper_band"]
        lower = bands["lower_band"]
        sma = bands["sma"]
        
        # Bollinger Bands Strategy: 
        # Buy when price touches lower band (oversold)
        # Sell when price touches upper band (overbought)
        
        if current_price <= lower:  # Price at or below lower band
            result["action"] = "BUY"
            result["side"] = Side.BUY
            result["price"] = int(self.market_data.bid_price * 100)
            result["reason"] = f"Price ${current_price:.2f} hit lower band ${lower:.2f}"
            
        elif current_price >= upper:  # Price at or above upper band
            result["action"] = "SELL"
            result["side"] = Side.SELL
            result["price"] = int(self.market_data.ask_price * 100)
            result["reason"] = f"Price ${current_price:.2f} hit upper band ${upper:.2f}"
        
        else:
            position = ((current_price - lower) / (upper - lower)) * 100
            result["reason"] = f"Price ${current_price:.2f} in band (L:${lower:.2f} M:${sma:.2f} U:${upper:.2f}) - {position:.0f}%"
        
        return result
