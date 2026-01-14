#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace hft::common {

using Price = int64_t;       // Fixed-point price representation (in ticks)
using Quantity = int64_t;    // Order quantity
using OrderId = uint64_t;    // Unique order identifier
using SymbolId = uint32_t;   // Symbol identifier

enum class Side : uint8_t { BUY = 0, SELL = 1 };

enum class OrderType : uint8_t { MARKET = 0, LIMIT = 1, IOC = 2, FOK = 3 };

enum class OrderStatus : uint8_t {
    NEW = 0,
    PARTIALLY_FILLED = 1,
    FILLED = 2,
    CANCELED = 3,
    REJECTED = 4
};

struct Order {
    OrderId order_id;
    SymbolId symbol_id;
    Side side;
    OrderType type;
    Price price;
    Quantity quantity;
    Quantity filled_quantity;
    OrderStatus status;
};

}  // namespace hft::common
