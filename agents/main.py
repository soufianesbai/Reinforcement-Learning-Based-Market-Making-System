#!/usr/bin/env python3
"""
Entry point for Trading Agent
"""

from .agent import TradingAgent


def main():
    """Main entry point"""
    agent = TradingAgent(symbol="SOLUSDT")
    
    # Run for 60 seconds (C++ LiveMarketData continuously updates)
    agent.run(duration_seconds=60)


if __name__ == "__main__":
    main()
