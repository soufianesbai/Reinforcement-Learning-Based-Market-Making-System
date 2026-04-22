"""
Type definitions and enumerations for trading agent
"""

from enum import Enum


class Side(Enum):
    """Trading side - BUY or SELL"""
    BUY = "BUY"
    SELL = "SELL"
