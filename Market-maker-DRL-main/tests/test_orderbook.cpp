#include <cassert>
#include <iostream>

#include "orderbook/order_book.hpp"

using namespace hft;

int main() {
    std::cout << "Running OrderBook tests...\n";

    // Test 1: Create order book
    orderbook::OrderBook ob(1);
    assert(ob.symbol_id() == 1);
    assert(ob.bid_depth() == 0);
    assert(ob.ask_depth() == 0);
    std::cout << "✓ OrderBook creation test passed\n";

    // Test 2: Add bid order
    ob.add_order(common::Side::BUY, 10000, 100);
    assert(ob.bid_depth() == 1);
    auto best_bid = ob.best_bid();
    assert(best_bid.has_value());
    assert(best_bid->price == 10000);
    assert(best_bid->quantity == 100);
    std::cout << "✓ Add bid order test passed\n";

    // Test 3: Add ask order
    ob.add_order(common::Side::SELL, 10100, 50);
    assert(ob.ask_depth() == 1);
    {
        auto best_ask = ob.best_ask();
        assert(best_ask.has_value());
        assert(best_ask->price == 10100);
        assert(best_ask->quantity == 50);
        (void)best_ask;  // Silence unused warning in Release builds
    }
    std::cout << "✓ Add ask order test passed\n";

    // Test 4: Multiple price levels
    ob.add_order(common::Side::BUY, 9900, 200);
    ob.add_order(common::Side::BUY, 10050, 150);
    assert(ob.bid_depth() == 3);
    best_bid = ob.best_bid();
    assert(best_bid->price == 10050);  // Highest bid
    std::cout << "✓ Multiple price levels test passed\n";

    // Test 5: Remove order
    ob.remove_order(common::Side::BUY, 10050, 150);
    assert(ob.bid_depth() == 2);
    best_bid = ob.best_bid();
    assert(best_bid->price == 10000);
    std::cout << "✓ Remove order test passed\n";

    // Test 6: Clear order book
    ob.clear();
    assert(ob.bid_depth() == 0);
    assert(ob.ask_depth() == 0);
    assert(!ob.best_bid().has_value());
    assert(!ob.best_ask().has_value());
    std::cout << "✓ Clear order book test passed\n";

    std::cout << "All OrderBook tests passed!\n";
    return 0;
}
