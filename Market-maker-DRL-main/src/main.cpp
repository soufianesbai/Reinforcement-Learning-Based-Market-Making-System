#include <csignal>
#include <iostream>
#include <memory>
#include <thread>

#include "common/logger.hpp"
#include "common/timestamp.hpp"
#include "market_data/feed_handler.hpp"
#include "orderbook/order_book.hpp"
#include "strategy/strategy.hpp"

namespace {
volatile std::sig_atomic_t shutdown_requested = 0;

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        shutdown_requested = 1;
    }
}
}  // namespace

int main(int argc [[maybe_unused]], char* argv[] [[maybe_unused]]) {
    using namespace hft;

    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize logging
    common::Logger::instance().set_level(common::LogLevel::INFO);
    LOG_INFO("HFT System starting...");

    try {
        // Initialize components
        auto strategy = std::make_unique<strategy::DummyStrategy>();
        strategy->initialize();
        LOG_INFO("Strategy initialized");

        // Create order books for symbols
        auto orderbook = std::make_unique<orderbook::OrderBook>(1);
        LOG_INFO("Order book created for symbol 1");

        // Start strategy
        strategy->start();
        LOG_INFO("Strategy started");

        // Main event loop
        LOG_INFO("Entering main event loop (Press Ctrl+C to stop)...");
        size_t iteration = 0;
        while (!shutdown_requested) {
            // This is a dummy event loop that would normally:
            // 1. Poll for market data updates
            // 2. Process order book updates
            // 3. Execute strategy logic
            // 4. Send orders to exchange
            // 5. Process order responses

            if (iteration % 1000000 == 0) {
                LOG_INFO("Event loop running... iteration: " + std::to_string(iteration));
            }

            ++iteration;

            // Prevent busy waiting
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        LOG_INFO("Shutdown signal received");

        // Cleanup
        strategy->stop();
        LOG_INFO("Strategy stopped");

        strategy->shutdown();
        LOG_INFO("Strategy shutdown complete");

        orderbook->clear();
        LOG_INFO("Order book cleared");

        LOG_INFO("HFT System shutdown complete");
        return 0;

    } catch (const std::exception& e) {
        LOG_CRITICAL(std::string("Fatal error: ") + e.what());
        return 1;
    }
}
