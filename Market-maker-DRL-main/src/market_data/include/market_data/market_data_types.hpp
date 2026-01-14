#pragma once

#include <cstdint>

#include "common/timestamp.hpp"
#include "common/types.hpp"

namespace hft::market_data {

struct Quote {
    common::SymbolId symbol_id;
    common::Price bid_price;
    common::Quantity bid_quantity;
    common::Price ask_price;
    common::Quantity ask_quantity;
    common::Timestamp timestamp;
};

struct Trade {
    common::SymbolId symbol_id;
    common::Price price;
    common::Quantity quantity;
    common::Side aggressor_side;
    common::Timestamp timestamp;
};

struct OrderBookUpdate {
    common::SymbolId symbol_id;
    common::Side side;
    common::Price price;
    common::Quantity quantity;
    bool is_delete;
    common::Timestamp timestamp;
};

}  // namespace hft::market_data
