#include "OrderBook/Trade.h"

Trade::Trade(const TradeInfo &bidTrade, const TradeInfo &askTrade)
    : bidTrade_{bidTrade}
      , askTrade_{askTrade} {
}

const TradeInfo &Trade::GetBidTrade() const {
    return bidTrade_;
}

const TradeInfo &Trade::GetAskTrade() const {
    return askTrade_;
}
