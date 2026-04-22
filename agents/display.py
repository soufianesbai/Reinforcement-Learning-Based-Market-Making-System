"""
Display functions for market and trading information
"""

from datetime import datetime
from typing import Dict


def print_decision(decision: Dict):
    """Print decision analysis - simple and clear format"""
    action = decision['action'] or 'HOLD'
    reason = decision['reason']
    quantity = decision.get('quantity', 0)
    
    # Color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    

    if action == 'BUY':
        # BUY in green
        print(f"\n{GREEN}{'═'*60}")
        print(f"  🔴 BUY")
        print(f"{'═'*60}{RESET}")
        print(f"  Reason:    {reason}")
    elif action == 'SELL':
        # SELL in red
        print(f"\n{RED}{'═'*60}")
        print(f"  🟢 SELL")
        print(f"{'═'*60}{RESET}")
        print(f"  Reason:    {reason}")
    else:
        # HOLD in yellow
        print(f"\n{YELLOW}{'═'*60}")
        print(f"  ⏸️  HOLD")
        print(f"{'═'*60}{RESET}")
        print(f"  Reason:    {reason}")


def print_execution(order_placed: bool, decision: Dict, total_trades: int, cash: float = None, price: float = None, inventory: float = 0.0):
    """Print execution result - simple format"""
    if order_placed:
        action = decision['action']
        quantity = decision.get('quantity', 0)
        unit_price = price if price else decision.get('price', 0) / 100.0
        total_cost = quantity * unit_price
        
        if action == 'BUY':
            RED = '\033[31m'
            RESET = '\033[0m'
            _ts = datetime.now().strftime('%H:%M:%S')
            print(f"\n{RED}[{_ts}] ✅ Order placed - BUY{RESET}")
            print(f"   Quantity:  {quantity:.4f}")
            print(f"   Unit Price: ${unit_price:.2f}")
            print(f"   Total:     ${total_cost:.2f}")
        elif action == 'SELL':
            GREEN = '\033[32m'
            RESET = '\033[0m'
            _ts = datetime.now().strftime('%H:%M:%S')
            print(f"\n{GREEN}[{_ts}] ✅ Order placed - SELL{RESET}")
            print(f"   Quantity:  {quantity:.4f}")
            print(f"   Unit Price: ${unit_price:.2f}")
            print(f"   Total:     ${total_cost:.2f}")
        
    
    # Always show cash and inventory
    BOLD = '\033[1m'
    RESET = '\033[0m'
    if cash is not None:
        print(f"\n{BOLD}💰 Cash: ${cash:.2f}{RESET}")
    print(f"{BOLD}📦 Inventory: {inventory:.4f}{RESET}")


def print_final_summary(cash: float, initial_cash: float, total_trades: int, inventory: float = 0.0):
    """Print final summary at the end"""
    profit = cash - initial_cash
    profit_pct = (profit / initial_cash * 100) if initial_cash > 0 else 0
    
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    
    print(f"\n{BOLD}{'═'*60}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'═'*60}{RESET}")
    print(f"  Total Trades:     {total_trades}")
    print(f"  Initial Cash:     ${initial_cash:.2f}")
    print(f"  Current Cash:     ${cash:.2f}")
    
    if profit >= 0:
        print(f"  {GREEN}Profit:           ${profit:.2f} ({profit_pct:+.2f}%){RESET}")
    else:
        print(f"  {RED}Loss:             ${profit:.2f} ({profit_pct:+.2f}%){RESET}")
    
    # Show final inventory
    print(f"  {CYAN}Final Inventory:  {inventory:.4f}{RESET}")
    
    print(f"{BOLD}{'═'*60}{RESET}\n")
