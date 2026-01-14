#include <chrono>
#include <iostream>
#include <random>

#include "orderbook/order_book.hpp"

using namespace hft;
using namespace std::chrono;

constexpr size_t NUM_OPERATIONS = 1000000;

int main() {
    std::cout << "OrderBook Performance Benchmark\n";
    std::cout << "================================\n";

    orderbook::OrderBook ob(1);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<common::Price> price_dist(9000, 11000);
    std::uniform_int_distribution<common::Quantity> qty_dist(1, 1000);
    std::uniform_int_distribution<int> side_dist(0, 1);

    // Benchmark: Add orders
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        auto side = side_dist(rng) == 0 ? common::Side::BUY : common::Side::SELL;
        auto price = price_dist(rng);
        auto qty = qty_dist(rng);
        ob.add_order(side, price, qty);
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();

    std::cout << "Add orders: " << NUM_OPERATIONS << " operations\n";
    std::cout << "Total time: " << duration / 1e6 << " ms\n";
    std::cout << "Avg time per operation: " << duration / NUM_OPERATIONS << " ns\n";
    std::cout << "Operations per second: "
              << static_cast<size_t>(NUM_OPERATIONS * 1e9 / duration) << "\n\n";

    // Benchmark: Best bid/ask queries
    start = high_resolution_clock::now();
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        auto bid = ob.best_bid();
        auto ask = ob.best_ask();
        (void)bid;
        (void)ask;
    }
    end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start).count();

    std::cout << "Best bid/ask queries: " << NUM_OPERATIONS << " operations\n";
    std::cout << "Total time: " << duration / 1e6 << " ms\n";
    std::cout << "Avg time per operation: " << duration / NUM_OPERATIONS << " ns\n";
    std::cout << "Operations per second: "
              << static_cast<size_t>(NUM_OPERATIONS * 1e9 / duration) << "\n\n";

    std::cout << "Final order book depth: Bids=" << ob.bid_depth()
              << ", Asks=" << ob.ask_depth() << "\n";

    return 0;
}
