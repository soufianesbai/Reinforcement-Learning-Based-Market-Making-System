#!/usr/bin/env python3
from .engine import MarketMakerEngine
from .strategies.sma import SMAStrategy  
from .strategies.random_coin import RandomStrategy
from .strategies.rl_agent import RLAgentStrategy

def main():
    # Va charger automatiquement models/gcrl_actor.pth
    strategy = RLAgentStrategy(symbol="SOLUSDT")
    engine = MarketMakerEngine(strategy=strategy, symbol="SOLUSDT")
    engine.run(duration_seconds=180)

if __name__ == "__main__":
    main()
