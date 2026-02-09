#pragma once

#include <vector>
#include "Core/Types.h"

struct TradeInfo {
    OrderId orderId_;
    Price price_;
    Quantity quantity_;
};

class Trade {
public:
    Trade(const TradeInfo &bidTrade, const TradeInfo &askTrade);

    const TradeInfo &GetBidTrade() const;
    const TradeInfo &GetAskTrade() const;

private:
    TradeInfo bidTrade_;
    TradeInfo askTrade_;
};

using Trades = std::vector<Trade>;
