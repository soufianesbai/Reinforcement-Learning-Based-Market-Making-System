#pragma once

#include <map>
#include <optional>

#include "common/types.hpp"
#include "orderbook/price_level.hpp"

namespace hft::orderbook {

class OrderBook {
  public:
    explicit OrderBook(common::SymbolId symbol_id) : symbol_id_(symbol_id) {}

    void add_order(common::Side side, common::Price price, common::Quantity quantity) {
        if (side == common::Side::BUY) {
            auto it = bids_.find(price);
            if (it != bids_.end()) {
                it->second.add_quantity(quantity);
            } else {
                bids_.emplace(price, PriceLevel(price, quantity));
            }
        } else {
            auto it = asks_.find(price);
            if (it != asks_.end()) {
                it->second.add_quantity(quantity);
            } else {
                asks_.emplace(price, PriceLevel(price, quantity));
            }
        }
    }

    void remove_order(common::Side side, common::Price price, common::Quantity quantity) {
        if (side == common::Side::BUY) {
            auto it = bids_.find(price);
            if (it != bids_.end()) {
                it->second.remove_quantity(quantity);
                if (it->second.is_empty()) {
                    bids_.erase(it);
                }
            }
        } else {
            auto it = asks_.find(price);
            if (it != asks_.end()) {
                it->second.remove_quantity(quantity);
                if (it->second.is_empty()) {
                    asks_.erase(it);
                }
            }
        }
    }

    [[nodiscard]] std::optional<PriceLevel> best_bid() const {
        if (bids_.empty()) {
            return std::nullopt;
        }
        // std::map with std::greater<> sorts in descending order, so begin() is highest
        return bids_.begin()->second;
    }

    [[nodiscard]] std::optional<PriceLevel> best_ask() const {
        if (asks_.empty()) {
            return std::nullopt;
        }
        return asks_.begin()->second;
    }

    [[nodiscard]] common::SymbolId symbol_id() const {
        return symbol_id_;
    }

    [[nodiscard]] size_t bid_depth() const {
        return bids_.size();
    }

    [[nodiscard]] size_t ask_depth() const {
        return asks_.size();
    }

    void clear() {
        bids_.clear();
        asks_.clear();
    }

  private:
    common::SymbolId symbol_id_;
    std::map<common::Price, PriceLevel, std::greater<>> bids_;  // Descending order
    std::map<common::Price, PriceLevel> asks_;                  // Ascending order
};

}  // namespace hft::orderbook
