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
    
    # Check if API keys are configured
    if BINANCE_TESTNET_API_KEY == "your_api_key_here" or not BINANCE_TESTNET_API_KEY:
        print("Error: Please configure your Binance testnet API keys in agents/exchange/config.py")
        print("   You can get them from: https://testnet.binance.vision/")
        return
    
    try:
        # Choose strategy from environment variable
        strategy_name = os.getenv("STRATEGY", "mean_reversion").lower()
        symbol = os.getenv("SYMBOL")

        if strategy_name == "rsi":
            from .strategies.rsi import RSIStrategy
            strategy_class = RSIStrategy
            print("Using RSI Strategy")
        elif strategy_name == "bollinger":
            from .strategies.bollinger_bands import BollingerBandsStrategy
            strategy_class = BollingerBandsStrategy
            print("Using Bollinger Bands Strategy")
        elif strategy_name == "dqn":
            try:
                from .strategies.dqn_agent import DQNStrategy
                strategy_class = DQNStrategy
                print("Using DQN Strategy")
            except ImportError as e:
                print(f"DQN requires PyTorch (pip install torch): {e}")
                return
        elif strategy_name == "ppo":
            try:
                from .strategies.PPO_agent import PPOStrategy
                strategy_class = PPOStrategy
                print("Using PPO Strategy")
            except ImportError as e:
                print(f"PPO requires TensorFlow (pip install tensorflow): {e}")
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
        
        # Run indefinitely until manually stopped
        agent.run()
        
    except Exception as e:
        print(f"Failed to start agent: {e}")


if __name__ == "__main__":
    main()
