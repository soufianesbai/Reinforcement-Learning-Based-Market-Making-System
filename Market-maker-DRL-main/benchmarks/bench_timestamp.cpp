#include <chrono>
#include <iostream>

#include "common/timestamp.hpp"

using namespace hft;
using namespace std::chrono;

constexpr size_t NUM_OPERATIONS = 10000000;

int main() {
    std::cout << "Timestamp Performance Benchmark\n";
    std::cout << "================================\n";

    // Benchmark: now() calls
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        auto ts = common::now();
        (void)ts;
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();

    std::cout << "now() calls: " << NUM_OPERATIONS << " operations\n";
    std::cout << "Total time: " << duration / 1e6 << " ms\n";
    std::cout << "Avg time per operation: " << duration / NUM_OPERATIONS << " ns\n\n";

    // Benchmark: to_nanos conversions
    auto ts = common::now();
    start = high_resolution_clock::now();
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        auto nanos = common::to_nanos(ts);
        (void)nanos;
    }
    end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start).count();

    std::cout << "to_nanos() conversions: " << NUM_OPERATIONS << " operations\n";
    std::cout << "Total time: " << duration / 1e6 << " ms\n";
    std::cout << "Avg time per operation: " << duration / NUM_OPERATIONS << " ns\n\n";

    // Benchmark: from_nanos conversions
    uint64_t nanos = 1234567890123456789ULL;
    start = high_resolution_clock::now();
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        auto timestamp = common::from_nanos(nanos);
        (void)timestamp;
    }
    end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start).count();

    std::cout << "from_nanos() conversions: " << NUM_OPERATIONS << " operations\n";
    std::cout << "Total time: " << duration / 1e6 << " ms\n";
    std::cout << "Avg time per operation: " << duration / NUM_OPERATIONS << " ns\n";

    return 0;
}
