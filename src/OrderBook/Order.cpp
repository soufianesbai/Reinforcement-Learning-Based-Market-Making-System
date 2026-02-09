#include "OrderBook/Order.h"
#include <format>
#include <stdexcept>

Order::Order(OrderType orderType, OrderId orderId, Side side, Price price, Quantity quantity)
    : orderType_{orderType}
      , orderId_{orderId}
      , side_{side}
      , price_{price}
      , initialQuantity_{quantity}
      , remainingQuantity_{quantity} {
    if (quantity == 0) {
        throw std::invalid_argument("Order quantity must be greater than zero");
    }
}

Order::Order(OrderId orderId, Side side, Quantity quantity)
    : Order(OrderType::Market, orderId, side, Constants::InvalidPrice, quantity) {
}

OrderId Order::GetOrderId() const { return orderId_; }
Side Order::GetSide() const { return side_; }
Price Order::GetPrice() const { return price_; }
OrderType Order::GetOrderType() const { return orderType_; }
Quantity Order::GetInitialQuantity() const { return initialQuantity_; }
Quantity Order::GetRemainingQuantity() const { return remainingQuantity_; }
Quantity Order::GetFilledQuantity() const { return GetInitialQuantity() - GetRemainingQuantity(); }
bool Order::IsFilled() const { return GetRemainingQuantity() == 0; }

void Order::Fill(Quantity quantity) {
    if (quantity > GetRemainingQuantity()) {
        throw std::logic_error(std::format("Order ({}) cannot be filled for more than its remaining quantity.",
                                           GetOrderId()));
    }
    remainingQuantity_ -= quantity;
}

void Order::ToGoodTillCancel(Price price) {
    if (orderType_ != OrderType::Market) {
        throw std::logic_error("Cannot convert non-market order to GoodTillCancel");
    }

    price_ = price;
    orderType_ = OrderType::GoodTillCancel;
}
