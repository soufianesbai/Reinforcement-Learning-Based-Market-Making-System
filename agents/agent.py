"""
Trading Agent - Main orchestrator
"""

import json
import os
from typing import Dict, Optional

from .types import Side, Position
from .strategy import TradingStrategy
from .display import (
    print_market_display,
    print_decision,
    print_execution,
    print_position,
    print_summary,
)


class TradingAgent:
    """Advanced trading agent that works with C++ LiveMarketData"""
    
    def __init__(self, symbol: str = "SOLUSDT"):
        self.symbol = symbol
        self.strategy = TradingStrategy(symbol, quantity_per_trade=100)
        
        # Files
        self.state_file = "data/market_state.json"
        self.orders_file = "data/agent_orders.json"
        
        # Tracking
        self.market_state = {}
        self.orders_placed = []
        self.position: Optional[Position] = None
        self.cash = 10000.0
        self.total_trades = 0
        
        # Initialize orders file
        self._init_orders_file()
        
    def _init_orders_file(self):
        """Initialize orders file"""
        with open(self.orders_file, 'w') as f:
            json.dump([], f)
    
    def read_market_state(self) -> Optional[Dict]:
        """Read market state from C++"""
        try:
            if not os.path.exists(self.state_file):
                return None
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            return state
        except Exception as e:
            print(f"❌ Error reading market state: {e}")
            return None
    
    def send_order(self, decision: Dict) -> bool:
        """Send order to C++ via JSON file"""
        if not decision["action"]:
            return False
        
        try:
            order = {
                "side": decision["side"].value,
                "price": int(decision["price"]),
                "quantity": int(decision["quantity"])
            }
            
            # Write order to file
            with open(self.orders_file, 'w') as f:
                json.dump([order], f)
            
            return True
        except Exception as e:
            print(f"❌ Error sending order: {e}")
            return False
    
    def update_position(self, decision: Dict, executed: bool = True):
        """Update position tracking"""
        if not executed or not decision["action"]:
            return
        
        price = decision["price"] / 100.0
        qty = decision["quantity"]
        
        if decision["side"] == Side.BUY:
            if self.position:
                # Add to position
                avg_price = (self.position.entry_price * self.position.quantity + price * qty) / (self.position.quantity + qty)
                self.position.quantity += qty
                self.position.entry_price = avg_price
            else:
                self.position = Position(
                    quantity=qty,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0
                )
        elif decision["side"] == Side.SELL and self.position:
            # Reduce position
            self.position.quantity -= qty
            if self.position.quantity <= 0:
                self.position = None
    
    def run(self, duration_seconds: int = 60):
        """Run agent connected to C++ LiveMarketData"""
        import time
        import traceback
        
        print(f"\n🤖 Python Agent Started - {self.symbol}")
        print(f"📁 Reading from: {self.state_file}")
        print(f"📝 Writing to:   {self.orders_file}")
        print(f"⏱️  Running for {duration_seconds} seconds...\n")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while (time.time() - start_time) < duration_seconds:
                # Read market state
                state = self.read_market_state()
                if not state:
                    print("⏳ Waiting for market state from C++...")
                    time.sleep(0.5)
                    continue
                
                iteration += 1
                
                # Analyze
                decision = self.strategy.analyze(state)
                print_decision(decision)
                
                # Execute
                order_placed = self.send_order(decision)
                print_execution(order_placed, decision, self.total_trades)
                
                # Update position
                if order_placed:
                    self.update_position(decision)
                    self.total_trades += 1
                    # Update current price
                    if self.position:
                        self.position.current_price = state.get("mid_price", 0) / 100.0
                        self.position.unrealized_pnl = (self.position.current_price - self.position.entry_price) * self.position.quantity
                    
                    # Afficher position et résumé seulement après un trade
                    print_position(self.position)
                    print_summary(self.cash, self.total_trades, self.position)
                
                # Wait a bit before next iteration
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\n⛔ Agent interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
        
        print(f"\n{'═'*90}")
        print(f"🏁 Agent Stopped")
        print(f"{'═'*90}\n")
