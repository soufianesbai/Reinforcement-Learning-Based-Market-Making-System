"""
Type definitions and enumerations for trading agent
"""

from enum import Enum
from dataclasses import dataclass


class Side(Enum):
    """Trading side - BUY or SELL"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    """Represents trading position"""
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    
    def __str__(self):
        pct = (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100 if (self.entry_price * self.quantity) > 0 else 0
        return f"Qty: {self.quantity:8.2f} | Entry: ${self.entry_price:8.2f} | Current: ${self.current_price:8.2f} | PnL: ${self.unrealized_pnl:8.2f} ({pct:+.2f}%)"
