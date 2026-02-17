from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Any
from .types import Side

@dataclass
class StrategySignal:
    """
    Standardized output from a strategy.
    Encapsulates the intent to trade without executing it.
    """
    side: Optional[Side]  # BUY, SELL, or None (HOLD)
    price: float
    quantity: float
    reason: str
    metadata: Dict[str, Any] = None  # Extra data for logging (e.g., deviation value)

class BaseStrategy(ABC):
    """
    Abstract Base Class that all trading strategies must implement.
    The engine uses this interface and knows nothing about the concrete implementation.
    """

    @abstractmethod
    def update_data(self, market_state: Dict) -> None:
        """
        Ingest new market data.
        :param market_state: The raw JSON dictionary from the C++ feed.
        """
        pass

    @abstractmethod
    def compute_signal(self) -> StrategySignal:
        """
        Calculate the trading decision based on current internal state.
        :return: A StrategySignal object containing the decision.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Return internal state for display/debugging (e.g. current SMA, neural net weights).
        """
        pass
