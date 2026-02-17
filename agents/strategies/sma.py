from typing import Dict, Any, Optional
from ..interfaces import BaseStrategy, StrategySignal
from ..types import Side
from ..market_data import MarketData

class SMAStrategy(BaseStrategy):
    """
    Implementation of the Mean Reversion strategy using SMA.
    """
    
    def __init__(self, symbol: str, quantity_per_trade: float = 5.0, window_size: int = 10, threshold: float = 0.01):
        self.symbol = symbol
        self.quantity = quantity_per_trade
        self.threshold = threshold
        # Reuse your existing helper class for data management
        self.market_data = MarketData(window_size=window_size)
        
        # Cache for the last calculated deviation for display purposes
        self.last_sma: Optional[float] = None
        self.last_deviation: Optional[float] = None

    def update_data(self, market_state: Dict) -> None:
        self.market_data.update(market_state)

    def compute_signal(self) -> StrategySignal:
        # Default signal: HOLD
        signal = StrategySignal(side=None, price=0.0, quantity=0.0, reason="", metadata={})

        if len(self.market_data.prices) < 3:
            signal.reason = f"Collecting data... ({len(self.market_data.prices)} prices)"
            return signal

        sma = self.market_data.get_sma()
        deviation = self.market_data.get_price_deviation()
        
        # Update internal state for get_status
        self.last_sma = sma
        self.last_deviation = deviation

        if deviation is None or sma is None:
            signal.reason = "Insufficient data"
            return signal

        # Logic Logic
        if deviation < -self.threshold:
            signal.side = Side.BUY
            signal.price = int(self.market_data.bid_price * 100) # Price in cents
            signal.quantity = self.quantity
            signal.reason = f"Price {deviation:.2f}% below SMA"
            
        elif deviation > self.threshold:
            signal.side = Side.SELL
            signal.price = int(self.market_data.ask_price * 100) # Price in cents
            signal.quantity = self.quantity
            signal.reason = f"Price {deviation:.2f}% above SMA"
        else:
            signal.reason = f"Deviation {deviation:.2f}% within threshold"

        signal.metadata = {
            "sma": sma,
            "deviation": deviation,
            "threshold": self.threshold
        }
        
        return signal
    # Dans agents/strategies/sma.py

    def get_status(self) -> Dict[str, Any]:
        """
        Retourne l'état actuel calculé directement depuis MarketData.
        Cela permet d'avoir la SMA même si on n'a pas encore demandé de signal de trading.
        """
        return {
            "name": "SMA Mean Reversion",
            "sma": self.market_data.get_sma(),             # Appel direct (Live)
            "deviation": self.market_data.get_price_deviation() # Appel direct (Live)
        }
