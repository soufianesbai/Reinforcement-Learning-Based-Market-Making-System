#pragma once

#include <chrono>
#include <cstdint>

namespace hft::common {

using Clock = std::chrono::steady_clock;
using Timestamp = Clock::time_point;
using Nanoseconds = std::chrono::nanoseconds;

inline Timestamp now() {
    return Clock::now();
}

inline uint64_t to_nanos(Timestamp ts) {
    return std::chrono::duration_cast<Nanoseconds>(ts.time_since_epoch()).count();
}

inline Timestamp from_nanos(uint64_t nanos) {
    return Timestamp(Nanoseconds(nanos));
}

}  // namespace hft::common
