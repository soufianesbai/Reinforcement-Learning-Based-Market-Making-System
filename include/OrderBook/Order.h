#pragma once

#include "Core/Constants.h"
#include "Core/OrderType.h"
#include "Core/Types.h"

class Order {
    // represents one individual order
public:
    Order(OrderType orderType, OrderId orderId, Side side, Price price, Quantity quantity);
    Order(OrderId orderId, Side side, Quantity quantity);

    OrderId GetOrderId() const;
    Side GetSide() const;
    Price GetPrice() const;
    OrderType GetOrderType() const;
    Quantity GetInitialQuantity() const;
    Quantity GetRemainingQuantity() const;
    Quantity GetFilledQuantity() const;
    bool IsFilled() const;

    void Fill(Quantity quantity);
    void ToGoodTillCancel(Price price);

private:
    OrderType orderType_;
    OrderId orderId_;
    Side side_;
    Price price_;
    Quantity initialQuantity_;
    Quantity remainingQuantity_;
};
