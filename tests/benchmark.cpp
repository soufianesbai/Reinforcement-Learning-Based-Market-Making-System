#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "OrderBook/OrderBook.h"
#include "OrderBook/Order.h"
#include "Core/Types.h"
#include "Core/OrderType.h"




// ==================== UTILITIES ====================

struct LatencyStats {
    double min_us;
    double max_us;
    double mean_us;
    double median_us;
    double p50_us;
    double p90_us;
    double p95_us;
    double p99_us;
    double stddev_us;
    long long count;
};

std::string formatNumber(long long num) {
    std::string str = std::to_string(num);
    int pos = str.length() - 3;
    while (pos > 0) {
        str.insert(pos, ",");
        pos -= 3;
    }
    return str;
}

LatencyStats computeStats(std::vector<double>& latencies) {
    LatencyStats stats{};
    stats.count = latencies.size();
    if (latencies.empty()) return stats;

    std::sort(latencies.begin(), latencies.end());

    stats.min_us = latencies.front();
    stats.max_us = latencies.back();
    stats.mean_us = std::accumulate(latencies.begin(), latencies.end(), 0.0) / stats.count;

    size_t mid = stats.count / 2;
    stats.median_us = (stats.count % 2 == 0)
        ? (latencies[mid - 1] + latencies[mid]) / 2.0
        : latencies[mid];

    stats.p50_us = latencies[static_cast<size_t>(stats.count * 0.50)];
    stats.p90_us = latencies[static_cast<size_t>(stats.count * 0.90)];
    stats.p95_us = latencies[static_cast<size_t>(stats.count * 0.95)];
    stats.p99_us = latencies[static_cast<size_t>(stats.count * 0.99)];

    double variance = 0.0;
    for (double l : latencies)
        variance += (l - stats.mean_us) * (l - stats.mean_us);
    stats.stddev_us = std::sqrt(variance / stats.count);

    return stats;
}

void printStats(const std::string& label, const LatencyStats& s) {
    std::cout << label << " (" << formatNumber(s.count) << " ops):\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Min:    " << s.min_us    << " us\n";
    std::cout << "  Mean:   " << s.mean_us   << " us\n";
    std::cout << "  Median: " << s.median_us << " us\n";
    std::cout << "  p90:    " << s.p90_us    << " us\n";
    std::cout << "  p95:    " << s.p95_us    << " us\n";
    std::cout << "  p99:    " << s.p99_us    << " us\n";
    std::cout << "  Max:    " << s.max_us    << " us\n";
    std::cout << "  Stddev: " << s.stddev_us << " us\n";
    double throughput = (s.mean_us > 0) ? 1000000.0 / s.mean_us : 0;
    std::cout << "  Throughput: " << formatNumber(static_cast<long long>(throughput)) << " ops/sec\n";
    std::cout << "\n";
}

// ==================== BENCHMARKS ====================

void BenchmarkAddOrderLatency(int numOrders) {
    Orderbook orderbook;
    std::vector<double> latencies;
    latencies.reserve(numOrders);

    std::mt19937 gen(42);
    std::uniform_int_distribution<Price> priceDist(90, 110);
    std::uniform_int_distribution<Quantity> qtyDist(1, 100);
    std::uniform_int_distribution<int> sideDist(0, 1);

    for (int i = 0; i < numOrders; ++i) {
        auto order = std::make_shared<Order>(
            OrderType::GoodTillCancel,
            static_cast<OrderId>(i),
            sideDist(gen) ? Side::Buy : Side::Sell,
            priceDist(gen),
            qtyDist(gen)
        );

        auto start = std::chrono::high_resolution_clock::now();
        orderbook.AddOrder(order);
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(us);
    }

    auto stats = computeStats(latencies);
    printStats("AddOrder", stats);
}

void BenchmarkCancelOrderLatency(int numOrders) {
    Orderbook orderbook;

    // Pre-populate with non-matching orders (wide spread)
    for (int i = 0; i < numOrders; ++i) {
        auto order = std::make_shared<Order>(
            OrderType::GoodTillCancel,
            static_cast<OrderId>(i),
            (i % 2) ? Side::Buy : Side::Sell,
            (i % 2) ? Price(50 + (i % 10)) : Price(150 + (i % 10)),
            Quantity(10)
        );
        orderbook.AddOrder(order);
    }

    // Shuffle cancel order to avoid sequential bias
    std::vector<OrderId> ids(numOrders);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 gen(42);
    std::shuffle(ids.begin(), ids.end(), gen);

    std::vector<double> latencies;
    latencies.reserve(numOrders);

    for (auto id : ids) {
        auto start = std::chrono::high_resolution_clock::now();
        orderbook.CancelOrder(id);
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(us);
    }

    auto stats = computeStats(latencies);
    printStats("CancelOrder", stats);
}

void BenchmarkModifyOrderLatency(int numOrders) {
    Orderbook orderbook;

    // Pre-populate
    for (int i = 0; i < numOrders; ++i) {
        auto order = std::make_shared<Order>(
            OrderType::GoodTillCancel,
            static_cast<OrderId>(i),
            (i % 2) ? Side::Buy : Side::Sell,
            (i % 2) ? Price(50 + (i % 10)) : Price(150 + (i % 10)),
            Quantity(10)
        );
        orderbook.AddOrder(order);
    }

    std::mt19937 gen(42);
    std::uniform_int_distribution<Price> priceDist(50, 60);
    std::uniform_int_distribution<Quantity> qtyDist(1, 20);

    std::vector<double> latencies;
    latencies.reserve(numOrders);

    for (int i = 0; i < numOrders; ++i) {
        // Only modify buy orders (odd IDs)
        OrderId id = static_cast<OrderId>(i * 2 + 1);
        if (id >= static_cast<OrderId>(numOrders)) break;

        OrderModify modify(id, Side::Buy, priceDist(gen), qtyDist(gen));

        auto start = std::chrono::high_resolution_clock::now();
        orderbook.MatchOrder(modify);
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(us);
    }

    auto stats = computeStats(latencies);
    printStats("ModifyOrder (MatchOrder)", stats);
}

void BenchmarkMatchingLatency(int numPairs) {
    std::vector<double> latencies;
    latencies.reserve(numPairs);

    for (int i = 0; i < numPairs; ++i) {
        Orderbook orderbook;

        // Place a resting sell order
        auto sell = std::make_shared<Order>(
            OrderType::GoodTillCancel,
            static_cast<OrderId>(i * 2),
            Side::Sell,
            Price(100),
            Quantity(10)
        );
        orderbook.AddOrder(sell);

        // Measure the matching buy order
        auto buy = std::make_shared<Order>(
            OrderType::GoodTillCancel,
            static_cast<OrderId>(i * 2 + 1),
            Side::Buy,
            Price(100),
            Quantity(10)
        );

        auto start = std::chrono::high_resolution_clock::now();
        auto trades = orderbook.AddOrder(buy);
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(us);
    }

    auto stats = computeStats(latencies);
    printStats("Matching (single trade)", stats);
}

void BenchmarkDeepBookMatchingLatency(int depth, int iterations) {
    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int iter = 0; iter < iterations; ++iter) {
        Orderbook orderbook;

        // Build a deep book with `depth` sell orders at the same price
        for (int i = 0; i < depth; ++i) {
            auto sell = std::make_shared<Order>(
                OrderType::GoodTillCancel,
                static_cast<OrderId>(i),
                Side::Sell,
                Price(100),
                Quantity(1)
            );
            orderbook.AddOrder(sell);
        }

        // One aggressive buy sweeps all levels
        auto buy = std::make_shared<Order>(
            OrderType::GoodTillCancel,
            static_cast<OrderId>(depth),
            Side::Buy,
            Price(100),
            Quantity(depth)
        );

        auto start = std::chrono::high_resolution_clock::now();
        auto trades = orderbook.AddOrder(buy);
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(us);
    }

    auto stats = computeStats(latencies);
    std::string label = "Deep matching (depth=" + std::to_string(depth) + ")";
    printStats(label, stats);
}

void BenchmarkGetOrderInfosLatency(int numOrders, int numCalls) {
    Orderbook orderbook;

    for (int i = 0; i < numOrders; ++i) {
        orderbook.AddOrder(std::make_shared<Order>(
            OrderType::GoodTillCancel,
            static_cast<OrderId>(i),
            (i % 2) ? Side::Buy : Side::Sell,
            (i % 2) ? Price(90 + (i % 10)) : Price(110 + (i % 10)),
            Quantity(10)
        ));
    }

    std::vector<double> latencies;
    latencies.reserve(numCalls);

    for (int i = 0; i < numCalls; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto infos = orderbook.GetOrderInfos();
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(us);
    }

    auto stats = computeStats(latencies);
    std::string label = "GetOrderInfos (" + formatNumber(numOrders) + " orders)";
    printStats(label, stats);
}

void BenchmarkHFTLatency(int numOperations) {
    Orderbook orderbook;
    std::mt19937 gen(42);
    std::uniform_int_distribution<Price> priceDist(98, 102);
    std::uniform_int_distribution<Quantity> qtyDist(1, 10);
    std::uniform_int_distribution<int> actionDist(0, 2);

    std::vector<OrderId> activeOrders;
    OrderId nextId = 0;

    std::vector<double> addLatencies, cancelLatencies, modifyLatencies;
    addLatencies.reserve(numOperations);
    cancelLatencies.reserve(numOperations);
    modifyLatencies.reserve(numOperations);

    for (int i = 0; i < numOperations; ++i) {
        int action = activeOrders.empty() ? 0 : actionDist(gen);

        if (action == 0 || activeOrders.empty()) {
            auto order = std::make_shared<Order>(
                OrderType::GoodTillCancel,
                nextId++,
                (i % 2) ? Side::Buy : Side::Sell,
                priceDist(gen),
                qtyDist(gen)
            );

            auto start = std::chrono::high_resolution_clock::now();
            auto trades = orderbook.AddOrder(order);
            auto end = std::chrono::high_resolution_clock::now();

            addLatencies.push_back(std::chrono::duration<double, std::micro>(end - start).count());

            if (!order->IsFilled())
                activeOrders.push_back(order->GetOrderId());

        } else if (action == 1) {
            size_t idx = gen() % activeOrders.size();

            auto start = std::chrono::high_resolution_clock::now();
            orderbook.CancelOrder(activeOrders[idx]);
            auto end = std::chrono::high_resolution_clock::now();

            cancelLatencies.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            activeOrders.erase(activeOrders.begin() + idx);

        } else {
            size_t idx = gen() % activeOrders.size();
            OrderModify modify(activeOrders[idx], Side::Buy, priceDist(gen), qtyDist(gen));

            auto start = std::chrono::high_resolution_clock::now();
            orderbook.MatchOrder(modify);
            auto end = std::chrono::high_resolution_clock::now();

            modifyLatencies.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
    }

    std::cout << "=== HFT Simulation (" << formatNumber(numOperations) << " ops) ===\n\n";

    if (!addLatencies.empty()) {
        auto s = computeStats(addLatencies);
        printStats("  HFT AddOrder", s);
    }
    if (!cancelLatencies.empty()) {
        auto s = computeStats(cancelLatencies);
        printStats("  HFT CancelOrder", s);
    }
    if (!modifyLatencies.empty()) {
        auto s = computeStats(modifyLatencies);
        printStats("  HFT ModifyOrder", s);
    }
}

// ==================== MAIN ====================

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "           ORDERBOOK LATENCY BENCHMARK\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << "--- AddOrder Latency ---\n";
    BenchmarkAddOrderLatency(10000);
    BenchmarkAddOrderLatency(100000);

    std::cout << "--- CancelOrder Latency ---\n";
    BenchmarkCancelOrderLatency(10000);
    BenchmarkCancelOrderLatency(100000);

    std::cout << "--- ModifyOrder Latency ---\n";
    BenchmarkModifyOrderLatency(10000);

    std::cout << "--- Matching Latency (single trade) ---\n";
    BenchmarkMatchingLatency(10000);

    std::cout << "--- Deep Book Matching Latency ---\n";
    BenchmarkDeepBookMatchingLatency(10, 5000);
    BenchmarkDeepBookMatchingLatency(100, 1000);
    BenchmarkDeepBookMatchingLatency(1000, 100);

    std::cout << "--- GetOrderInfos Latency ---\n";
    BenchmarkGetOrderInfosLatency(1000, 5000);
    BenchmarkGetOrderInfosLatency(10000, 1000);

    std::cout << "--- HFT Mixed Operations Latency ---\n";
    BenchmarkHFTLatency(100000);

    std::cout << std::string(70, '=') << "\n";
    std::cout << "           BENCHMARK COMPLETE\n";
    std::cout << std::string(70, '=') << "\n";

    return 0;
}
