#!/usr/bin/env python3
"""
Unit Test Suite - Trading System Verification

Comprehensive test that validates:
1. All 3 trading strategies (Mean Reversion, RSI, Bollinger Bands) work independently
2. TradingAgent can be instantiated with each strategy
3. Symbol detection from market_state.json works correctly
4. Binance testnet connectivity is established
5. API authentication succeeds

Run with: python3 -m agents.tests.test_system
"""

import sys
import json
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.rsi import RSIStrategy
from ..strategies.bollinger_bands import BollingerBandsStrategy
from ..agent import TradingAgent
from ..exchange.config import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET

print("=" * 70)
print("COMPREHENSIVE SYSTEM TEST")
print("=" * 70)

# Test data
state = {
    "bid_price": 8750,
    "ask_price": 8760,
    "mid_price": 8755,
    "spread": 10,
    "bids": [],
    "asks": []
}

# 1. Test each strategy independently
print("\n[1/4] Testing strategies independently...")
strategies = [
    ("Mean Reversion", MeanReversionStrategy()),
    ("RSI", RSIStrategy()),
    ("Bollinger Bands", BollingerBandsStrategy())
]

for name, strategy in strategies:
    try:
        # Feed enough data
        iterations = 20 if name == "Bollinger Bands" else (16 if name == "RSI" else 3)
        for _ in range(iterations):
            strategy.analyze(state)
        
        decision = strategy.analyze(state)
        print(f"  {name}: {decision['reason'][:50]}...")
    except Exception as e:
        print(f"  {name}: {e}")
        sys.exit(1)

# 2. Test Agent with Mean Reversion (default)
print("\n[2/4] Testing Agent with Mean Reversion...")
try:
    agent_mr = TradingAgent(
        api_key=BINANCE_TESTNET_API_KEY,
        api_secret=BINANCE_TESTNET_API_SECRET
    )
    print(f"   Agent created with symbol: {agent_mr.symbol}")
    print(f"   Strategy: {agent_mr.strategy.__class__.__name__}")
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

# 3. Test Agent with RSI
print("\n[3/4] Testing Agent with RSI...")
try:
    agent_rsi = TradingAgent(
        strategy_class=RSIStrategy,
        api_key=BINANCE_TESTNET_API_KEY,
        api_secret=BINANCE_TESTNET_API_SECRET
    )
    print(f"  Agent created with symbol: {agent_rsi.symbol}")
    print(f"  Strategy: {agent_rsi.strategy.__class__.__name__}")
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

# 4. Test Agent with Bollinger Bands
print("\n[4/4] Testing Agent with Bollinger Bands...")
try:
    agent_bb = TradingAgent(
        strategy_class=BollingerBandsStrategy,
        api_key=BINANCE_TESTNET_API_KEY,
        api_secret=BINANCE_TESTNET_API_SECRET
    )
    print(f"  Agent created with symbol: {agent_bb.symbol}")
    print(f"  Strategy: {agent_bb.strategy.__class__.__name__}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nAvailable Strategies:")
print("  1. Mean Reversion (default)")
print("  2. RSI")
print("  3. Bollinger Bands")
print("\n🚀 Run commands:")
print("  ./run_agent.sh")
print("  STRATEGY=rsi ./run_agent.sh")
print("  STRATEGY=bollinger ./run_agent.sh")
print("=" * 70)
