"""
Trading Agent Package - Modular organization
"""

from .types import Side, Position
from .market_data import MarketData
from .interfaces import BaseStrategy, StrategySignal
from .engine import MarketMakerEngine

# Optionnel : Exposer les stratégies directement
from .strategies.sma import SMAStrategy
# from .strategies.rl_agent import RLAgentStrategy
#from .strategies.a2c_rl import A2CStrategy

__all__ = [
    "Side", 
    "Position",
    "MarketData",
    "BaseStrategy",
    "StrategySignal",
    "MarketMakerEngine",
    "SMAStrategy",
]
