#pragma once

#include <map>
#include <unordered_map>
#include <chrono>
#include <vector>
#include "Core/OrderType.h"
#include "Core/Types.h"
#include "MarketData/MarketDataFeed.h"
#include "OrderBook/ExchangeRules.h"
#include "OrderBook/LevelInfo.h"
#include "OrderBook/Order.h"
#include "OrderBook/OrderModify.h"
#include "OrderBook/Trade.h"

class Orderbook {
private:
    struct OrderEntry {
        OrderPointer order_{nullptr};
        OrderPointers::iterator location_;
    };

    std::map<Price, OrderPointers, std::greater<Price>> bids_;
    std::map<Price, OrderPointers, std::less<Price>> asks_;
    std::unordered_map<OrderId, OrderEntry> orders_;

    std::chrono::system_clock::time_point lastDayReset_;
    std::chrono::hours dayResetHour_{15};
    int dayResetMinute_{59};

    MarketDataStats stats_;
    uint64_t lastSequenceNumber_ = 0;
    bool isInitialized_ = false;

    ExchangeRules exchangeRules_;

    bool CanMatch(Side side, Price price) const;
    OrderValidation ValidateOrder(OrderPointer order) const;
    void CheckAndResetDay();
    void CancelGoodForDayOrders();
    std::vector<std::pair<OrderPointer, Quantity>> CollectMatchesForFillOrKill(
        OrderPointer order, Quantity &remainingQuantity);
    Trades ExecuteMatchesForFillOrKill(
        OrderPointer order,
        const std::vector<std::pair<OrderPointer, Quantity>> &matchingOrders);
    Trades MatchFillOrKill(OrderPointer order);
    Trades MatchOrders();
    void ProcessNewOrder(const NewOrderMessage &msg);
    void ProcessCancel(const CancelOrderMessage &msg);
    void ProcessModify(const ModifyOrderMessage &msg);
    void ProcessTrade(const TradeMessage &msg);
    void ProcessSnapshot(const BookSnapshotMessage &msg);

public:
    Orderbook();

    void SetExchangeRules(const ExchangeRules &rules);
    const ExchangeRules &GetExchangeRules() const;
    void SetDayResetTime(int hour, int minute = 59);

    Trades AddOrder(OrderPointer order);
    void CancelOrder(OrderId orderId);
    Trades MatchOrder(OrderModify order);
    std::size_t Size() const;
    OrderbookLevelInfos GetOrderInfos() const;

    bool ProcessMarketData(const MarketDataMessage &message);
    size_t ProcessMarketDataBatch(const std::vector<MarketDataMessage> &messages);
    const MarketDataStats &GetMarketDataStats() const;
    void ResetMarketDataStats();
    bool IsInitialized() const;
    uint64_t GetLastSequenceNumber() const;
};
