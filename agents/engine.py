import json
import time
import os
import traceback
from typing import Dict, Optional

from .interfaces import BaseStrategy
from .types import Side, Position
from .display import print_market_display, print_decision, print_execution, print_position, print_summary

class MarketMakerEngine:
    """
    The engine handles the infrastructure:
    - File I/O with C++
    - Position Tracking
    - Order Execution
    - Delegating logic to the injected Strategy
    """
    
    def __init__(self, strategy: BaseStrategy, symbol: str = "SOLUSDT"):
        self.symbol = symbol
        # DEPENDENCY INJECTION: The strategy is passed in, not created inside.
        self.strategy = strategy
        
        # Files
        self.state_file = "data/market_state.json"
        self.orders_file = "data/agent_orders.json"
        
        # State
        self.position: Optional[Position] = None
        self.cash = 10000.0
        self.total_trades = 0
        
        self._init_orders_file()

    def _init_orders_file(self):
        with open(self.orders_file, 'w') as f:
            json.dump([], f)

    def read_market_state(self) -> Optional[Dict]:
        try:
            if not os.path.exists(self.state_file):
                return None
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def execute_signal(self, signal) -> bool:
        """
        Translates the StrategySignal into a JSON order for the C++ engine.
        """
        if not signal.side:
            return False

        try:
            order = {
                "side": signal.side.value,
                "price": int(signal.price),
                "quantity": int(signal.quantity)
            }
            
            # Writing to the shared file triggers the C++ engine
            with open(self.orders_file, 'w') as f:
                json.dump([order], f)
            return True
        except Exception as e:
            print(f"❌ Execution Error: {e}")
            return False

    def update_position(self, signal):
        """Update local position tracking based on the signal assumed executed."""
        price_val = signal.price / 100.0
        
        if signal.side == Side.BUY:
            if self.position:
                avg_price = (self.position.entry_price * self.position.quantity + price_val * signal.quantity) / (self.position.quantity + signal.quantity)
                self.position.quantity += signal.quantity
                self.position.entry_price = avg_price
            else:
                self.position = Position(signal.quantity, price_val, price_val, 0.0)
                
        elif signal.side == Side.SELL and self.position:
            self.position.quantity -= signal.quantity
            if self.position.quantity <= 0:
                self.position = None

    def run(self, duration_seconds: int = 60):
        print(f"\n🤖 Engine Started - Strategy: {self.strategy.get_status().get('name')}")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while (time.time() - start_time) < duration_seconds:
                # 1. Fetch Data
                state = self.read_market_state()
                if not state:
                    time.sleep(0.1)
                    continue
                
                iteration += 1
                if hasattr(self.strategy, 'current_inventory'):
                     qty = self.position.quantity if self.position else 0.0
                     # Remettre en négatif si c'est une position Short (si votre C++ le gère)
                     self.strategy.current_inventory = qty
                
                # 2. Update Strategy
                self.strategy.update_data(state)
                
                # 3. Get Decision
                signal = self.strategy.compute_signal()
                
                # 4. Display Logic (Using the metadata from signal)
                print_market_display(state)
                # Map signal back to dict format expected by your existing display functions
                display_decision = {
                    "action": signal.side.value if signal.side else None,
                    "reason": signal.reason,
                    ** (signal.metadata or {})
                }
                print_decision(display_decision)
                
                # 5. Execute
                executed = self.execute_signal(signal)
                print_execution(executed, {"side": signal.side, "price": signal.price, "quantity": signal.quantity}, self.total_trades)
                
                # 6. Post-Trade Updates
                if executed:
                    self.update_position(signal)
                    self.total_trades += 1
                
                if self.position:
                    # Update PnL based on latest mid price
                    mid_price = state.get("mid_price", 0) / 100.0
                    self.position.current_price = mid_price
                    self.position.unrealized_pnl = (mid_price - self.position.entry_price) * self.position.quantity
                    print_position(self.position)

                print_summary(self.cash, self.total_trades, self.position)
                
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("Stopped.")
        except Exception:
            traceback.print_exc()
