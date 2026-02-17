#!/usr/bin/env python3
from .engine import MarketMakerEngine
from .strategies.sma import SMAStrategy  
from .strategies.random_coin import RandomStrategy
# from .strategies.rl_agent import RLAgentStrategy

def main():
    # 1. Configuration
    symbol = "SOLUSDT"
    
    # 2. Choix de la stratégie (Instantiation)
    # C'est ici qu'on injecte la dépendance. 
    # Pour utiliser le RL, on ferait: strategy = RLAgentStrategy("model.pt")


    #strategy = SMAStrategy(symbol,
    #                      quantity_per_trade=100,
    #                      window_size=50,
    #                      threshold=0.00001)
    
    # 3. Initialisation du moteur avec la stratégie choisie
    #engine = MarketMakerEngine(strategy=strategy, symbol=symbol)
    
    # 4. Lancement
    #engine.run(duration_seconds=180)

    strategy = RandomStrategy()    # <-- On branche la nouvelle
    
    engine = MarketMakerEngine(strategy=strategy, symbol="SOLUSDT")
    engine.run()


    

if __name__ == "__main__":
    main()
