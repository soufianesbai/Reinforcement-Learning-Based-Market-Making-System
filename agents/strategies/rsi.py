"""
RSI (Relative Strength Index) Trading Strategy
"""

from typing import Dict, Optional
from collections import deque
from ..trading_types import Side
from ..market_data import MarketData


class RSIStrategy:
    """RSI-based trading strategy"""
    
    def __init__(self, symbol: str = "SOLUSDT", quantity_per_trade: float = 0.1, period: int = 14):
        self.symbol = symbol
        self.quantity_per_trade = quantity_per_trade
        self.market_data = MarketData()
        self.period = period
        
        # Strategy-specific price history
        self.prices = deque(maxlen=period + 1)
        
    def _calculate_rsi(self) -> Optional[float]:
        """Calculate RSI (Relative Strength Index)"""
        if len(self.prices) < self.period + 1:
            return None
        
        prices_list = list(self.prices)
        deltas = [prices_list[i] - prices_list[i-1] for i in range(1, len(prices_list))]
        
        gains = sum(d for d in deltas if d > 0) / self.period
        losses = sum(-d for d in deltas if d < 0) / self.period
        
        if losses == 0:
            return 100.0 if gains > 0 else 0.0
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def analyze(self, state_dict: Dict) -> Dict:
        """Analyze market using RSI and return decision"""
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
        
        # Need enough data for RSI
        if len(self.prices) < self.period + 1:
            num_prices = len(self.prices)
            result["reason"] = f"Collecting data... ({num_prices}/{self.period + 1})"
            return result
        
        rsi = self._calculate_rsi()
        
        if rsi is None:
            result["reason"] = "Insufficient data for RSI"
            return result
        
        # RSI Strategy: Buy on oversold, Sell on overbought
        if rsi < 30:  # Oversold
            result["action"] = "BUY"
            result["side"] = Side.BUY
            result["price"] = int(self.market_data.bid_price * 100)
            result["reason"] = f"RSI oversold: {rsi:.1f} (< 30)"
            
        elif rsi > 70:  # Overbought
            result["action"] = "SELL"
            result["side"] = Side.SELL
            result["price"] = int(self.market_data.ask_price * 100)
            result["reason"] = f"RSI overbought: {rsi:.1f} (> 70)"
        
        else:
            result["reason"] = f"RSI neutral: {rsi:.1f}"
        
        return result
