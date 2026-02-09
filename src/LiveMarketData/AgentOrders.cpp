#include "LiveMarketData/AgentOrders.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {
std::vector<AgentOrder> persistentAgentOrders;
OrderId nextAgentOrderId = 100000;
}

const std::vector<AgentOrder> &GetPersistentAgentOrders() {
    return persistentAgentOrders;
}

int CountActiveAgentOrders() {
    int active = 0;
    for (const auto &order : persistentAgentOrders) {
        if (order.isActive) {
            active++;
        }
    }
    return active;
}

void LoadAgentOrders(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.good()) {
        return;
    }

    try {
        json orders;
        file >> orders;
        file.close();

        if (!orders.is_array()) {
            return;
        }

        // Ajouter les ordres (permettre les doublons)
        for (const auto &order_data : orders) {
            std::string side_str = order_data["side"];
            Side side = (side_str == "BUY") ? Side::Buy : Side::Sell;
            Price price = order_data["price"];
            Quantity qty = order_data["quantity"];

            persistentAgentOrders.emplace_back(side, price, qty, nextAgentOrderId++);
            std::cout << "  📥 Nouvel ordre agent: " << side_str << " "
                      << (qty / 100.0) << "@" << (price / 100.0) << "\n";
        }

        // Vider le fichier après lecture
        std::ofstream clear_file(filename);
        clear_file << "[]";
        clear_file.close();

    } catch (const std::exception &) {
        // Erreur de parsing, ignorer
    }
}

void ReinjectAgentOrders(Orderbook &orderbook) {
    for (auto &agentOrder : persistentAgentOrders) {
        if (!agentOrder.isActive) {
            continue;
        }

        auto order = std::make_shared<Order>(
            OrderType::GoodTillCancel,
            agentOrder.orderId,
            agentOrder.side,
            agentOrder.price,
            agentOrder.quantity
        );

        auto trades = orderbook.AddOrder(order);

        // Si l'ordre a été complètement exécuté, le marquer comme inactif
        if (!trades.empty() && order->IsFilled()) {
            agentOrder.isActive = false;
        }
    }
}

void CancelAgentOrder(OrderId orderId) {
    for (auto &agentOrder : persistentAgentOrders) {
        if (agentOrder.orderId == orderId) {
            agentOrder.isActive = false;
            break;
        }
    }
}
