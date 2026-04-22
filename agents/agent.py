"""
Trading Agent - Main orchestrator with direct Binance testnet execution
"""

import json
import os
from typing import Dict, Optional

from .trading_types import Side
from .strategies.mean_reversion import MeanReversionStrategy
from .exchange.binance_client import BinanceTestnetClient
from .display import print_decision, print_execution, print_final_summary


class TradingAgent:
    """Advanced trading agent that executes orders directly on Binance testnet"""
    
    def __init__(self, symbol: str = None, strategy_class=None, api_key: str = None, api_secret: str = None):
        # Files - define this first
        self.state_file = "data/market_state.json"
        
        # Priority: explicit arg -> SYMBOL env -> market_state.json -> fallback
        if symbol is None:
            symbol = os.environ.get("SYMBOL") or self._get_symbol_from_market_state()
        
        self.symbol = symbol or "SOLUSDT"
        print(f"📌 Symbol: {self.symbol}")
        
        # Use provided strategy or default to MeanReversionStrategy
        if strategy_class is None:
            strategy_class = MeanReversionStrategy
        self.strategy = strategy_class(self.symbol, quantity_per_trade=0.1)
        
        # Binance client for direct order execution
        if not api_key or not api_secret:
            raise ValueError("API key and secret are required. Please provide them or set them in config.py")
        
        self.binance = BinanceTestnetClient(api_key, api_secret)
        
        # Tracking
        self.market_state = {}
        self.inventory: float = 0.0
        self.total_trades = 0
        self.last_order_id = None
        
        # Test connectivity
        if not self.binance.test_connectivity():
            raise ConnectionError("Failed to connect to Binance testnet. Check your internet connection.")
        
        print(f"✅ Connected to Binance testnet")
        
        # Get initial USDT balance
        balance = self.binance.get_balance("USDT")
        if balance is not None:
            self.cash = balance
            print(f"💰 Initial USDT balance: {self.cash:.2f}")
        else:
            print("⚠️  Could not retrieve balance")
        
        # Check if we already have inventory for this asset
        asset = self._extract_base_asset(self.symbol)
        asset_balance = self.binance.get_balance(asset)
        if asset_balance and asset_balance > 0:
            print(f"📦 Found existing {asset} inventory: {asset_balance:.8f}")
            self.inventory = asset_balance
        else:
            print(f"📦 No existing {asset} inventory")
            self.inventory = 0.0
    
    def _extract_base_asset(self, symbol: str) -> str:
        """Extract base asset from symbol (e.g., SOLUSDT -> SOL)"""
        # Remove common quote currencies
        for quote in ['USDT', 'BUSD', 'BTC', 'ETH', 'BNB']:
            if symbol.endswith(quote):
                return symbol[:-len(quote)]
        return symbol
        
    def _get_symbol_from_market_state(self) -> Optional[str]:
        """Get symbol from market_state.json if available"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    return state.get("symbol")
        except Exception:
            pass
        return None
    
    def read_market_state(self) -> Optional[Dict]:
        """Read market state from C++"""
        try:
            if not os.path.exists(self.state_file):
                return None
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Ignore stale market state from another symbol during symbol switch
            file_symbol = state.get("symbol")
            if file_symbol and file_symbol != self.symbol:
                print(f"⏳ Waiting for {self.symbol} data (file has {file_symbol})...")
                return None

            return state
        except Exception as e:
            print(f"❌ Error reading market state: {e}")
            return None
    
    def execute_order(self, decision: Dict) -> Optional[Dict]:
        """Execute order directly on Binance testnet
        
        Returns:
            Order response from Binance or None if failed
        """
        if not decision["action"]:
            return None
        
        try:
            # Calculate quantity (convert from decision to actual quantity)
            quantity = decision["quantity"]
            
            # Verify we have inventory before SELL
            if decision["side"] == Side.SELL:
                if quantity > self.inventory:
                    print(f"❌ Cannot SELL: Insufficient inventory (need {quantity:.8f}, have {self.inventory:.8f})")
                    return None

            # Verify we have enough cash before BUY
            if decision["side"] == Side.BUY:
                if self.cash is None:
                    balance = self.binance.get_balance("USDT")
                    if balance is not None:
                        self.cash = balance
                est_price = decision.get("price", 0) / 100.0
                est_cost = quantity * est_price if est_price > 0 else 0
                if self.cash is not None and est_cost > 0 and self.cash < est_cost:
                    print(f"❌ Cannot BUY: Insufficient USDT (need {est_cost:.2f}, have {self.cash:.2f})")
                    return None
            
            # Place market order on Binance
            print(f"📤 Sending {decision['side'].value} order to Binance testnet...")
            
            order_response = self.binance.place_order(
                symbol=self.symbol,
                side=decision["side"],
                quantity=quantity,
                order_type="MARKET"
            )
            
            if order_response:
                self.last_order_id = order_response.get('orderId')
                print(f"✅ Order executed! Order ID: {self.last_order_id}")
                
                # Update balance and inventory from Binance
                cash_balance = self.binance.get_balance("USDT")
                if cash_balance is not None:
                    self.cash = cash_balance
                
                # Update inventory from actual Binance balance
                asset = self._extract_base_asset(self.symbol)
                inventory_balance = self.binance.get_balance(asset)
                if inventory_balance is not None:
                    self.inventory = inventory_balance
                
                return order_response
            else:
                print(f"❌ Order failed")
                return None
                
        except Exception as e:
            print(f"❌ Error executing order: {e}")
            return None
    
    def run(self, duration_seconds: int = None):
        """Run agent indefinitely until stopped manually (Ctrl+C or SIGTERM)."""
        import signal
        import time
        import traceback

        # Store initial cash for summary
        initial_cash = self.cash

        print(f"\n🤖 Trading Agent Started - {self.symbol}")
        print(f"📁 Reading market data from: {self.state_file}")
        print(f"🔗 Executing orders on: Binance Testnet")

        stop_flag = {"stop": False}

        def _handle_signal(sig, frame):
            stop_flag["stop"] = True

        signal.signal(signal.SIGTERM, _handle_signal)

        iteration = 0

        try:
            while not stop_flag["stop"]:
                # Read market state
                state = self.read_market_state()
                if not state:
                    print("⏳ Waiting for market state from C++...")
                    time.sleep(0.5)
                    continue
                
                iteration += 1
                
                # Analyze market
                decision = self.strategy.analyze(state)
                print_decision(decision)
                
                # Execute order directly on Binance
                order_response = self.execute_order(decision)
                order_placed = order_response is not None
                
                # Get actual execution price if available
                exec_price = None
                if order_placed and order_response.get('fills'):
                    fills = order_response.get('fills', [])
                    total_cost = sum(float(fill['price']) * float(fill['qty']) for fill in fills)
                    total_qty = sum(float(fill['qty']) for fill in fills)
                    exec_price = total_cost / total_qty if total_qty > 0 else None
                
                # Count trades
                if order_placed:
                    self.total_trades += 1

                print_execution(order_placed, decision, self.total_trades, self.cash, exec_price, self.inventory)
                
                # Wait before next iteration
                time.sleep(2)  # Longer wait for real trading
        
        except KeyboardInterrupt:
            print("\n\n⛔ Agent stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
        
        # Final summary
        print_final_summary(self.cash, initial_cash, self.total_trades, self.inventory)
