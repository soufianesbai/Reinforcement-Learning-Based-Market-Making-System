#include <cassert>
#include <iostream>

#include "common/timestamp.hpp"
#include "common/types.hpp"

using namespace hft;

int main() {
    std::cout << "Running Types tests...\n";

    // Test 1: Order creation
    common::Order order{
        .order_id = 12345,
        .symbol_id = 1,
        .side = common::Side::BUY,
        .type = common::OrderType::LIMIT,
        .price = 10000,
        .quantity = 100,
        .filled_quantity = 0,
        .status = common::OrderStatus::NEW};

    assert(order.order_id == 12345);
    assert(order.side == common::Side::BUY);
    assert(order.type == common::OrderType::LIMIT);
    assert(order.status == common::OrderStatus::NEW);
    (void)order;  // Silence unused warning in Release builds
    std::cout << "✓ Order creation test passed\n";

    // Test 2: Side enum
    assert(static_cast<uint8_t>(common::Side::BUY) == 0);
    assert(static_cast<uint8_t>(common::Side::SELL) == 1);
    std::cout << "✓ Side enum test passed\n";

    // Test 3: OrderType enum
    assert(static_cast<uint8_t>(common::OrderType::MARKET) == 0);
    assert(static_cast<uint8_t>(common::OrderType::LIMIT) == 1);
    std::cout << "✓ OrderType enum test passed\n";

    // Test 4: Timestamp functions
    auto now = common::now();
    auto nanos = common::to_nanos(now);
    auto reconstructed = common::from_nanos(nanos);
    assert(common::to_nanos(now) == common::to_nanos(reconstructed));
    (void)reconstructed;  // Silence unused warning in Release builds
    std::cout << "✓ Timestamp functions test passed\n";

    std::cout << "All Types tests passed!\n";
    return 0;
}
