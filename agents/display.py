"""
Display functions for market and trading information
"""

from datetime import datetime
from typing import Dict
from .types import Position


def print_market_display(state: Dict):
    """Print formatted market display"""
    if not state:
        return
    
    bid = state.get("bid_price", 0) / 100.0
    ask = state.get("ask_price", 0) / 100.0
    mid = state.get("mid_price", 0) / 100.0
    spread_bps = state.get("spread_bps", 0)
    iteration = state.get("iteration", 0)
    
    print(f"\n{'═'*90}")
    print(f"{'ITERATION':<15} {iteration:<20} {'TIME':<20} {datetime.now().strftime('%H:%M:%S'):<20}")
    print(f"{'═'*90}")
    print(f"Bid: ${bid:>10.2f}  |  Ask: ${ask:>10.2f}  |  Mid: ${mid:>10.2f}  |  Spread: {spread_bps:>7.2f} bps")
    
    print("\n📊 Orderbook:")
    print("  Bids                          Asks")
    bids = state.get("bids", [])
    asks = state.get("asks", [])
    
    for i in range(max(len(bids), len(asks))):
        bid_str = f"  ${bids[i]['price']/100:>9.2f} x {bids[i]['quantity']:>7.0f}" if i < len(bids) else "                        "
        ask_str = f"  ${asks[i]['price']/100:>9.2f} x {asks[i]['quantity']:>7.0f}" if i < len(asks) else ""
        print(f"{bid_str:<30} {ask_str}")


def print_decision(decision: Dict):
    """Print decision analysis"""
    print(f"\n{'═'*90}")
    print(f"📊 Decision Analysis")
    print(f"{'═'*90}")
    print(f"Action: {decision['action'] or 'HOLD':>10}")
    print(f"Reason: {decision['reason']}")
    if decision.get('sma'):
        print(f"SMA:    ${decision['sma']:>10.2f}")
    if decision.get('deviation') is not None:
        print(f"Deviation: {decision['deviation']:>6.2f}%")


def print_execution(order_placed: bool, decision: Dict, total_trades: int):
    """Print execution result"""
    if order_placed:
        print(f"\n✅ ORDER PLACED:")
        print(f"   Side:     {decision['side'].value}")
        print(f"   Price:    ${decision['price']/100:>10.2f}")
        print(f"   Quantity: {decision['quantity']:>10.0f}")
    else:
        print(f"\n⏸️  HOLD")


def print_position(position: Position | None):
    """Print current position"""
    print(f"\n{'═'*90}")
    print(f"📈 Position")
    print(f"{'═'*90}")
    if position:
        print(f"  {position}")
    else:
        print("  No position")


def print_summary(cash: float, total_trades: int, position: Position | None):
    """Print summary"""
    print(f"\n{'═'*90}")
    print(f"📋 Summary")
    print(f"{'═'*90}")
    print(f"  Cash:           ${cash:>12.2f}")
    print(f"  Total Trades:   {total_trades:>12}")
    print(f"  Positions:      {1 if position else 0:>12}")
    
    if position:
        total_value = cash + position.unrealized_pnl
        print(f"  Total Value:    ${total_value:>12.2f}")
