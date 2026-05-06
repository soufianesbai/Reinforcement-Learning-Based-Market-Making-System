#!/usr/bin/env python3
"""
Entry point for Trading Agent with Binance testnet integration
Choose strategy with STRATEGY environment variable: 'mean_reversion' (default) or 'rsi'
"""

import os
from .agent import TradingAgent
from .exchange.config import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET


def main():
    """Main entry point"""

    if BINANCE_TESTNET_API_KEY == "your_api_key_here" or not BINANCE_TESTNET_API_KEY:
        print("Error: Please configure your Binance testnet API keys in agents/exchange/config.py")
        print("   You can get them from: https://testnet.binance.vision/")
        return

    try:
        strategy_name = os.getenv("STRATEGY", "mean_reversion").lower()
        symbol = os.getenv("SYMBOL")

        strategies = {
            "rsi": (".strategies.rsi", "RSIStrategy"),
            "bollinger": (".strategies.bollinger_bands", "BollingerBandsStrategy"),
            "sma": (".strategies.sma", "SMAStrategy"),
            "random": (".strategies.random_coin", "RandomStrategy"),
            "a2c": (".strategies.a2c_rl", "A2CStrategy"),
            "dqn": (".strategies.dqn_agent_faster", "DQNStrategy"),
            "ppo": (".strategies.PPO_agent", "PPOStrategy"),
        }

        if strategy_name == "sac":
            from .strategies.sac_agent import SACStrategy
            strategy_class = SACStrategy
            print("Using SAC Market Maker")
        elif strategy_name in strategies:
            module_path, class_name = strategies[strategy_name]
            try:
                import importlib
                mod = importlib.import_module(module_path, package="agents")
                strategy_class = getattr(mod, class_name)
                print(f"Using {class_name}")
            except ImportError as e:
                print(f"{strategy_name.upper()} import failed: {e}")
                return
        else:
            from .strategies.mean_reversion import MeanReversionStrategy
            strategy_class = MeanReversionStrategy
            print("Using Mean Reversion Strategy")

        agent = TradingAgent(
            symbol=symbol,
            strategy_class=strategy_class,
            api_key=BINANCE_TESTNET_API_KEY,
            api_secret=BINANCE_TESTNET_API_SECRET
        )

        agent.run()

    except Exception as e:
        print(f"Failed to start agent: {e}")


if __name__ == "__main__":
    main()
