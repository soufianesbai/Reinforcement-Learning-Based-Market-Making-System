#include "OrderBook/OrderModify.h"

OrderModify::OrderModify(OrderId orderId, Side side, Price price, Quantity quantity)
    : orderId_{orderId}
      , price_{price}
      , side_{side}
      , quantity_{quantity} {
}

OrderId OrderModify::GetOrderId() const {
    return orderId_;
}

Price OrderModify::GetPrice() const {
    return price_;
}

Side OrderModify::GetSide() const {
    return side_;
}

Quantity OrderModify::GetQuantity() const {
    return quantity_;
}

OrderPointer OrderModify::ToOrderPointer(OrderType type) const {
    return std::make_shared<Order>(type, GetOrderId(), GetSide(), GetPrice(), GetQuantity());
}
