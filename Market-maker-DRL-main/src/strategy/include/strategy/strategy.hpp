#pragma once

#include <memory>

#include "common/types.hpp"
#include "market_data/market_data_types.hpp"
#include "orderbook/order_book.hpp"

namespace hft::strategy {

class Strategy {
  public:
    virtual ~Strategy() = default;

    // Market data callbacks
    virtual void on_quote(const market_data::Quote& quote) = 0;
    virtual void on_trade(const market_data::Trade& trade) = 0;
    virtual void on_orderbook_update(const market_data::OrderBookUpdate& update) = 0;

    // Order lifecycle callbacks
    virtual void on_order_ack(const common::Order& order) = 0;
    virtual void on_order_fill(const common::Order& order, common::Quantity filled_qty) = 0;
    virtual void on_order_cancel(const common::Order& order) = 0;
    virtual void on_order_reject(const common::Order& order) = 0;

    // Strategy lifecycle
    virtual void initialize() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void shutdown() = 0;

    [[nodiscard]] virtual const char* name() const = 0;
};

// Example dummy strategy
class DummyStrategy : public Strategy {
  public:
    void on_quote(const market_data::Quote&) override {}
    void on_trade(const market_data::Trade&) override {}
    void on_orderbook_update(const market_data::OrderBookUpdate&) override {}
    void on_order_ack(const common::Order&) override {}
    void on_order_fill(const common::Order&, common::Quantity) override {}
    void on_order_cancel(const common::Order&) override {}
    void on_order_reject(const common::Order&) override {}

    void initialize() override {}
    void start() override {}
    void stop() override {}
    void shutdown() override {}

    [[nodiscard]] const char* name() const override {
        return "DummyStrategy";
    }
};

}  // namespace hft::strategy
