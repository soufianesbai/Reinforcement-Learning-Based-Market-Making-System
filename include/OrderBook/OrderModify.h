#pragma once

#include "Core/OrderType.h"
#include "Core/Types.h"
#include "OrderBook/Order.h"

class OrderModify {
public:
    OrderModify(OrderId orderId, Side side, Price price, Quantity quantity);

    OrderId GetOrderId() const;
    Price GetPrice() const;
    Side GetSide() const;
    Quantity GetQuantity() const;
    OrderPointer ToOrderPointer(OrderType type) const;

private:
    OrderId orderId_;
    Price price_;
    Side side_;
    Quantity quantity_;
};
