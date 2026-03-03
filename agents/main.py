#!/usr/bin/env python3
from .engine import MarketMakerEngine
from .strategies.sma import SMAStrategy  
from .strategies.random_coin import RandomStrategy
from .strategies.rl_agent import RLAgentStrategy
from agents.strategies.a2c_rl import A2CStrategy

def main():
    # Va charger automatiquement models/gcrl_actor.pth
    strategy = RLAgentStrategy(symbol="SOLUSDT")
    #strategy = A2CStrategy(symbol="SOLUSDT")
    engine = MarketMakerEngine(strategy=strategy, symbol="SOLUSDT")
    engine.run(duration_seconds=180)

if __name__ == "__main__":
    main()
