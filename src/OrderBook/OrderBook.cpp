#include "OrderBook/OrderBook.h"
#include <algorithm>
#include <numeric>
#include <limits>

Orderbook::Orderbook()
    : lastDayReset_(std::chrono::system_clock::now()) {
}

void Orderbook::SetExchangeRules(const ExchangeRules &rules) {
    exchangeRules_ = rules;
}

const ExchangeRules &Orderbook::GetExchangeRules() const {
    return exchangeRules_;
}

void Orderbook::SetDayResetTime(int hour, int minute) {
    if (hour >= 0 && hour <= 23 && minute >= 0 && minute <= 59) {
        dayResetHour_ = std::chrono::hours(hour);
        dayResetMinute_ = minute;
    }
}

bool Orderbook::CanMatch(Side side, Price price) const {
    if (side == Side::Buy) {
        if (asks_.empty()) {
            return false;
        }
        const auto &[bestAsk, _] = *asks_.begin();
        return price >= bestAsk;
    } else {
        if (bids_.empty()) {
            return false;
        }
        const auto &[bestBid, _] = *bids_.begin();
        return price <= bestBid;
    }
}

OrderValidation Orderbook::ValidateOrder(OrderPointer order) const {
    if (orders_.contains(order->GetOrderId())) {
        return OrderValidation::Reject(RejectReason::DuplicateOrderId);
    }

    Price orderPrice = order->GetPrice();
    bool isConvertedMarketOrder = (orderPrice == std::numeric_limits<Price>::max() ||
                                   orderPrice == std::numeric_limits<Price>::min());

    if (!isConvertedMarketOrder) {
        if (!exchangeRules_.IsValidPrice(orderPrice)) {
            return OrderValidation::Reject(RejectReason::InvalidPrice);
        }
    }

    if (!exchangeRules_.IsValidQuantity(order->GetRemainingQuantity())) {
        if (order->GetRemainingQuantity() < exchangeRules_.minQuantity) {
            return OrderValidation::Reject(RejectReason::BelowMinQuantity);
        } else if (order->GetRemainingQuantity() > exchangeRules_.maxQuantity) {
            return OrderValidation::Reject(RejectReason::AboveMaxQuantity);
        } else {
            return OrderValidation::Reject(RejectReason::InvalidQuantity);
        }
    }

    if (!isConvertedMarketOrder) {
        if (!exchangeRules_.IsValidNotional(order->GetPrice(), order->GetRemainingQuantity())) {
            return OrderValidation::Reject(RejectReason::BelowMinNotional);
        }
    }

    return OrderValidation::Accept();
}

void Orderbook::CheckAndResetDay() {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);
    auto lastResetTime = std::chrono::system_clock::to_time_t(lastDayReset_);

    std::tm nowTm = *std::localtime(&nowTime);

    std::tm todayResetTm = nowTm;
    todayResetTm.tm_hour = dayResetHour_.count();
    todayResetTm.tm_min = dayResetMinute_;
    todayResetTm.tm_sec = 0;
    auto todayResetTime = std::mktime(&todayResetTm);

    if (lastResetTime < todayResetTime && nowTime >= todayResetTime) {
        CancelGoodForDayOrders();
        lastDayReset_ = now;
    }
}

void Orderbook::CancelGoodForDayOrders() {
    std::vector<OrderId> ordersToCancel;

    for (const auto &[orderId, entry]: orders_) {
        if (entry.order_->GetOrderType() == OrderType::GoodForDay) {
            ordersToCancel.push_back(orderId);
        }
    }
    
    for (const auto &orderId: ordersToCancel) {
        CancelOrder(orderId);
    }
}

std::vector<std::pair<OrderPointer, Quantity>> Orderbook::CollectMatchesForFillOrKill(
    OrderPointer order,
    Quantity &remainingQuantity) {
    
    std::vector<std::pair<OrderPointer, Quantity>> matchingOrders;

    if (order->GetSide() == Side::Buy) {
        for (auto &[askPrice, askOrders]: asks_) {
            if (askPrice > order->GetPrice()) break;

            for (auto &ask: askOrders) {
                Quantity matchQty = std::min(remainingQuantity, ask->GetRemainingQuantity());
                matchingOrders.push_back({ask, matchQty});
                remainingQuantity -= matchQty;
                if (remainingQuantity == 0) break;
            }
            if (remainingQuantity == 0) break;
        }
    } else {
        for (auto &[bidPrice, bidOrders]: bids_) {
            if (bidPrice < order->GetPrice()) break;

            for (auto &bid: bidOrders) {
                Quantity matchQty = std::min(remainingQuantity, bid->GetRemainingQuantity());
                matchingOrders.push_back({bid, matchQty});
                remainingQuantity -= matchQty;
                if (remainingQuantity == 0) break;
            }
            if (remainingQuantity == 0) break;
        }
    }

    return matchingOrders;
}

Trades Orderbook::ExecuteMatchesForFillOrKill(
    OrderPointer order,
    const std::vector<std::pair<OrderPointer, Quantity>> &matchingOrders) {
    
    Trades trades;

    for (auto &[matchOrder, quantity]: matchingOrders) {
        order->Fill(quantity);
        matchOrder->Fill(quantity);

        if (order->GetSide() == Side::Buy) {
            trades.push_back(Trade{
                TradeInfo{order->GetOrderId(), order->GetPrice(), quantity},
                TradeInfo{matchOrder->GetOrderId(), matchOrder->GetPrice(), quantity}
            });
        } else {
            trades.push_back(Trade{
                TradeInfo{matchOrder->GetOrderId(), matchOrder->GetPrice(), quantity},
                TradeInfo{order->GetOrderId(), order->GetPrice(), quantity}
            });
        }

        if (matchOrder->IsFilled()) {
            CancelOrder(matchOrder->GetOrderId());
        }
    }

    return trades;
}

Trades Orderbook::MatchFillOrKill(OrderPointer order) {
    Quantity remainingQuantity = order->GetRemainingQuantity();

    auto matchingOrders = CollectMatchesForFillOrKill(order, remainingQuantity);

    if (remainingQuantity > 0) {
        return {};
    }

    return ExecuteMatchesForFillOrKill(order, matchingOrders);
}

Trades Orderbook::MatchOrders() {
    Trades trades;
    trades.reserve(orders_.size());

    while (true) {
        if (bids_.empty() || asks_.empty()) {
            break;
        }
        auto &[bidPrice, bids] = *bids_.begin();
        auto &[askPrice, asks] = *asks_.begin();

        if (bidPrice < askPrice) {
            break;
        }

        std::vector<OrderId> filledOrders;

        while (!bids.empty() && !asks.empty()) {
            auto &bid = bids.front();
            auto &ask = asks.front();

            Quantity quantity = std::min(bid->GetRemainingQuantity(), ask->GetRemainingQuantity());

            trades.push_back(Trade{
                TradeInfo{bid->GetOrderId(), bid->GetPrice(), quantity},
                TradeInfo{ask->GetOrderId(), ask->GetPrice(), quantity}
            });

            bid->Fill(quantity);
            ask->Fill(quantity);

            if (bid->IsFilled()) {
                filledOrders.push_back(bid->GetOrderId());
                bids.pop_front();
            }
            if (ask->IsFilled()) {
                filledOrders.push_back(ask->GetOrderId());
                asks.pop_front();
            }
        }

        for (const auto &orderId: filledOrders) {
            orders_.erase(orderId);
        }

        if (bids.empty()) {
            bids_.erase(bidPrice);
        }
        if (asks.empty()) {
            asks_.erase(askPrice);
        }
    }

    std::vector<OrderId> iocOrdersToCancel;

    for (const auto &[orderId, entry]: orders_) {
        if (entry.order_->GetOrderType() == OrderType::ImmediateOrCancel) {
            iocOrdersToCancel.push_back(orderId);
        }
    }
    
    for (const auto &orderId: iocOrdersToCancel) {
        CancelOrder(orderId);
    }

    return trades;
}

void Orderbook::ProcessNewOrder(const NewOrderMessage &msg) {
    try {
        auto order = std::make_shared<Order>(
            msg.orderType,
            msg.orderId,
            msg.side,
            msg.price,
            msg.quantity
        );
        auto trades = AddOrder(order);
        stats_.newOrders++;
        stats_.trades += trades.size();
    } catch (const std::invalid_argument &) {
        stats_.errors++;
    }
}

void Orderbook::ProcessCancel(const CancelOrderMessage &msg) {
    CancelOrder(msg.orderId);
    stats_.cancellations++;
}

void Orderbook::ProcessModify(const ModifyOrderMessage &msg) {
    OrderModify modify(msg.orderId, msg.side, msg.newPrice, msg.newQuantity);
    MatchOrder(modify);
    stats_.modifications++;
}

void Orderbook::ProcessTrade(const TradeMessage &msg) {
    stats_.trades++;
}

void Orderbook::ProcessSnapshot(const BookSnapshotMessage &msg) {
    bids_.clear();
    asks_.clear();
    orders_.clear();

    OrderId syntheticId = 0x8000000000000000ULL;

    for (const auto &level: msg.bids) {
        if (level.quantity == 0) continue;

        try {
            auto order = std::make_shared<Order>(
                OrderType::GoodTillCancel,
                syntheticId++,
                Side::Buy,
                level.price,
                level.quantity
            );

            auto &orders = bids_[level.price];
            orders.push_back(order);
            auto iterator = std::next(orders.begin(), orders.size() - 1);
            orders_.insert({order->GetOrderId(), OrderEntry{order, iterator}});
        } catch (const std::invalid_argument &) {
            continue;
        }
    }

    for (const auto &level: msg.asks) {
        if (level.quantity == 0) continue;

        try {
            auto order = std::make_shared<Order>(
                OrderType::GoodTillCancel,
                syntheticId++,
                Side::Sell,
                level.price,
                level.quantity
            );

            auto &orders = asks_[level.price];
            orders.push_back(order);
            auto iterator = std::next(orders.begin(), orders.size() - 1);
            orders_.insert({order->GetOrderId(), OrderEntry{order, iterator}});
        } catch (const std::invalid_argument &) {
            continue;
        }
    }

    isInitialized_ = true;
    lastSequenceNumber_ = msg.sequenceNumber;
    stats_.snapshots++;
}

Trades Orderbook::AddOrder(OrderPointer order) {
    CheckAndResetDay();

    if (order->GetOrderType() == OrderType::Market) {
        if (order->GetSide() == Side::Buy && !asks_.empty()) {
            order->ToGoodTillCancel(std::numeric_limits<Price>::max());
        } else if (order->GetSide() == Side::Sell && !bids_.empty()) {
            order->ToGoodTillCancel(std::numeric_limits<Price>::min());
        } else {
            return {};
        }
    }

    auto validation = ValidateOrder(order);
    if (!validation.isValid) {
        return {};
    }

    if (order->GetOrderType() == OrderType::ImmediateOrCancel && !CanMatch(order->GetSide(), order->GetPrice())) {
        return {};
    }

    if (order->GetOrderType() == OrderType::FillOrKill) {
        return MatchFillOrKill(order);
    }

    OrderPointers::iterator iterator;

    if (order->GetSide() == Side::Buy) {
        auto &orders = bids_[order->GetPrice()];
        orders.push_back(order);
        iterator = std::next(orders.begin(), orders.size() - 1);
    } else {
        auto &orders = asks_[order->GetPrice()];
        orders.push_back(order);
        iterator = std::next(orders.begin(), orders.size() - 1);
    }

    orders_.insert({order->GetOrderId(), OrderEntry{order, iterator}});

    return MatchOrders();
}

void Orderbook::CancelOrder(OrderId orderId) {
    if (!orders_.contains(orderId)) {
        return;
    }

    const auto &[order, orderIterator] = orders_.at(orderId);
    orders_.erase(orderId);

    if (order->GetSide() == Side::Sell) {
        auto price = order->GetPrice();
        auto &orders = asks_.at(price);
        orders.erase(orderIterator);
        if (orders.empty()) {
            asks_.erase(price);
        }
    } else {
        auto price = order->GetPrice();
        auto &orders = bids_.at(price);
        orders.erase(orderIterator);
        if (orders.empty()) {
            bids_.erase(price);
        }
    }
}

Trades Orderbook::MatchOrder(OrderModify order) {
    CheckAndResetDay();

    if (!orders_.contains(order.GetOrderId())) {
        return {};
    }

    const auto &[existingOrder, _] = orders_.at(order.GetOrderId());
    CancelOrder(order.GetOrderId());
    return AddOrder(order.ToOrderPointer(existingOrder->GetOrderType()));
}

std::size_t Orderbook::Size() const {
    return orders_.size();
}

OrderbookLevelInfos Orderbook::GetOrderInfos() const {
    LevelInfos bidInfos, askInfos;
    bidInfos.reserve(orders_.size());
    askInfos.reserve(orders_.size());

    auto CreateLevelInfos = [](Price price, const OrderPointers &orders) {
        return LevelInfo{
            price, std::accumulate(orders.begin(), orders.end(), (Quantity) 0,
                                   [](std::size_t runningSum, const OrderPointer &order) {
                                       return runningSum + order->GetRemainingQuantity();
                                   })
        };
    };

    for (const auto &[price, orders]: bids_) {
        bidInfos.push_back(CreateLevelInfos(price, orders));
    }

    for (const auto &[price, orders]: asks_) {
        askInfos.push_back(CreateLevelInfos(price, orders));
    }

    return OrderbookLevelInfos(bidInfos, askInfos);
}

bool Orderbook::ProcessMarketData(const MarketDataMessage &message) {
    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        std::visit([this](auto &&msg) {
            using T = std::decay_t<decltype(msg)>;

            if constexpr (std::is_same_v<T, NewOrderMessage>) {
                ProcessNewOrder(msg);
            } else if constexpr (std::is_same_v<T, CancelOrderMessage>) {
                ProcessCancel(msg);
            } else if constexpr (std::is_same_v<T, ModifyOrderMessage>) {
                ProcessModify(msg);
            } else if constexpr (std::is_same_v<T, TradeMessage>) {
                ProcessTrade(msg);
            } else if constexpr (std::is_same_v<T, BookSnapshotMessage>) {
                ProcessSnapshot(msg);
            }
        }, message);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        stats_.messagesProcessed++;
        stats_.totalProcessingTime += latency;
        stats_.maxLatency = std::max(stats_.maxLatency, latency);
        stats_.minLatency = std::min(stats_.minLatency, latency);

        return true;
    } catch (...) {
        stats_.errors++;
        return false;
    }
}

size_t Orderbook::ProcessMarketDataBatch(const std::vector<MarketDataMessage> &messages) {
    size_t successCount = 0;
    for (const auto &msg: messages) {
        if (ProcessMarketData(msg)) {
            successCount++;
        }
    }
    return successCount;
}

const MarketDataStats &Orderbook::GetMarketDataStats() const {
    return stats_;
}

void Orderbook::ResetMarketDataStats() {
    stats_.Reset();
}

bool Orderbook::IsInitialized() const {
    return isInitialized_;
}

uint64_t Orderbook::GetLastSequenceNumber() const {
    return lastSequenceNumber_;
}
