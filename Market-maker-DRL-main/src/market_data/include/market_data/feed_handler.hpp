#pragma once

#include <functional>
#include <memory>

#include "market_data/market_data_types.hpp"
#include "network/connection.hpp"

namespace hft::market_data {

using QuoteHandler = std::function<void(const Quote&)>;
using TradeHandler = std::function<void(const Trade&)>;
using OrderBookUpdateHandler = std::function<void(const OrderBookUpdate&)>;

class FeedHandler {
  public:
    explicit FeedHandler(std::unique_ptr<network::Connection> connection)
        : connection_(std::move(connection)) {}

    virtual ~FeedHandler() = default;

    // Delete copy, allow move
    FeedHandler(const FeedHandler&) = delete;
    FeedHandler& operator=(const FeedHandler&) = delete;
    FeedHandler(FeedHandler&&) noexcept = default;
    FeedHandler& operator=(FeedHandler&&) noexcept = default;

    void set_quote_handler(QuoteHandler handler) {
        quote_handler_ = std::move(handler);
    }

    void set_trade_handler(TradeHandler handler) {
        trade_handler_ = std::move(handler);
    }

    void set_orderbook_update_handler(OrderBookUpdateHandler handler) {
        orderbook_update_handler_ = std::move(handler);
    }

    virtual void subscribe(common::SymbolId symbol_id) = 0;
    virtual void unsubscribe(common::SymbolId symbol_id) = 0;
    virtual void process_messages() = 0;

  protected:
    std::unique_ptr<network::Connection> connection_;
    QuoteHandler quote_handler_;
    TradeHandler trade_handler_;
    OrderBookUpdateHandler orderbook_update_handler_;
};

}  // namespace hft::market_data
