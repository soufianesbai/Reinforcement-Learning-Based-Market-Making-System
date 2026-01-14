#pragma once

#include "common/types.hpp"

namespace hft::orderbook {

struct PriceLevel {
    common::Price price;
    common::Quantity quantity;
    uint32_t order_count;

    PriceLevel() : price(0), quantity(0), order_count(0) {}

    PriceLevel(common::Price p, common::Quantity q, uint32_t count = 1)
        : price(p), quantity(q), order_count(count) {}

    void add_quantity(common::Quantity q) {
        quantity += q;
        ++order_count;
    }

    void remove_quantity(common::Quantity q) {
        if (q > quantity) {
            quantity = 0;
            order_count = 0;
        } else {
            quantity -= q;
            if (order_count > 0) {
                --order_count;
            }
        }
    }

    [[nodiscard]] bool is_empty() const {
        return quantity == 0 || order_count == 0;
    }
};

}  // namespace hft::orderbook
