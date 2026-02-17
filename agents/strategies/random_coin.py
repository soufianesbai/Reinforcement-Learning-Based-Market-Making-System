import random
from typing import Dict, Any
from ..interfaces import BaseStrategy, StrategySignal
from ..types import Side

class RandomStrategy(BaseStrategy):
    def __init__(self):
        pass

    def update_data(self, market_state: Dict) -> None:
        # On s'en fiche des données, on joue au hasard !
        pass

    def compute_signal(self) -> StrategySignal:
        dice = random.random()
        
        if dice < 0.1: # 10% de chance d'acheter
            return StrategySignal(Side.BUY, 100, 10, "Pile !", {"dice": dice})
        elif dice > 0.9: # 10% de chance de vendre
            return StrategySignal(Side.SELL, 100, 10, "Face !", {"dice": dice})
        else:
            return StrategySignal(None, 0, 0, "Tranche...", {"dice": dice})

    def get_status(self) -> Dict[str, Any]:
        return {"name": "Random Coin Toss"}
