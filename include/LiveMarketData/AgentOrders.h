#pragma once

#include <string>
#include <vector>
#include "OrderBook/OrderBook.h"

struct AgentOrder {
    Side side;
    Price price;
    Quantity quantity;
    OrderId orderId;
    bool isActive;

    AgentOrder(Side s, Price p, Quantity q, OrderId id)
        : side(s), price(p), quantity(q), orderId(id), isActive(true) {}
};

void LoadAgentOrders(const std::string &filename);
void ReinjectAgentOrders(Orderbook &orderbook);
void CancelAgentOrder(OrderId orderId);
const std::vector<AgentOrder> &GetPersistentAgentOrders();
int CountActiveAgentOrders();
