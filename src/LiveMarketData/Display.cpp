#include "LiveMarketData/Display.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include "LiveMarketData/AgentOrders.h"

using json = nlohmann::json;

std::string FormatTimestamp(const std::chrono::system_clock::time_point &tp) {
    auto time = std::chrono::system_clock::to_time_t(tp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void PrintOrderbook(const Orderbook &orderbook, const std::string &symbol, int levels) {
    auto infos = orderbook.GetOrderInfos();
    const auto &bids = infos.GetBids();
    const auto &asks = infos.GetAsks();

    // Clear screen
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif

    auto now = std::chrono::system_clock::now();

    int activeAgentOrders = CountActiveAgentOrders();

    std::cout << "========================================\n";
    std::cout << "  LIVE ORDERBOOK: " << symbol << "\n";
    std::cout << "  " << FormatTimestamp(now) << "\n";
    std::cout << "  🤖 Ordres Agent Actifs: " << activeAgentOrders << "\n";
    std::cout << "========================================\n\n";

    std::cout << std::setw(15) << "BID QTY" << " | "
              << std::setw(12) << "BID PRICE" << " | "
              << std::setw(12) << "ASK PRICE" << " | "
              << std::setw(15) << "ASK QTY" << "\n";
    std::cout << std::string(65, '-') << "\n";

    int maxLevels = std::max(std::min((int) bids.size(), levels),
                             std::min((int) asks.size(), levels));

    for (int i = 0; i < maxLevels; ++i) {
        // Print bid
        if (i < bids.size()) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(2)
                      << bids[i].quantity_ / 100.0 << " | "
                      << std::setw(12) << std::fixed << std::setprecision(2)
                      << bids[i].price_ / 100.0 << " | ";
        } else {
            std::cout << std::setw(15) << "-" << " | "
                      << std::setw(12) << "-" << " | ";
        }

        // Print ask
        if (i < asks.size()) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(2)
                      << asks[i].price_ / 100.0 << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2)
                      << asks[i].quantity_ / 100.0 << "\n";
        } else {
            std::cout << std::setw(12) << "-" << " | "
                      << std::setw(15) << "-" << "\n";
        }
    }

    std::cout << "========================================\n";

    // Calculate and display spread
    if (!bids.empty() && !asks.empty()) {
        double spread = (asks[0].price_ - bids[0].price_) / 100.0;
        double midPrice = (bids[0].price_ + asks[0].price_) / 200.0;
        double spreadBps = (spread / midPrice) * 10000;

        std::cout << "Best Bid: $" << std::fixed << std::setprecision(2)
                  << bids[0].price_ / 100.0 << "\n";
        std::cout << "Best Ask: $" << asks[0].price_ / 100.0 << "\n";
        std::cout << "Spread: $" << spread << " (" << std::setprecision(1)
                  << spreadBps << " bps)\n";
        std::cout << "Mid Price: $" << std::setprecision(2) << midPrice << "\n";
    }

    std::cout << "\nOrderbook Size: " << orderbook.Size() << " orders\n";

    // Display market data stats
    const auto &stats = orderbook.GetMarketDataStats();
    std::cout << "Messages Processed: " << stats.messagesProcessed << "\n";
    std::cout << "Average Latency: " << std::fixed << std::setprecision(3)
              << stats.GetAverageLatencyMicros() << " μs\n";

    // Afficher les ordres de l'agent
    const auto &agentOrders = GetPersistentAgentOrders();
    if (!agentOrders.empty()) {
        std::cout << "\n--- ORDRES DE L'AGENT ---\n";
        for (const auto &order : agentOrders) {
            if (!order.isActive) {
                continue;
            }
            std::string side_str = (order.side == Side::Buy) ? "BUY" : "SELL";
            std::cout << "  🤖 " << side_str << " "
                      << std::fixed << std::setprecision(2) << (order.quantity / 100.0)
                      << " @ $" << (order.price / 100.0)
                      << " [ID: " << order.orderId << "]\n";
        }
    }

    std::cout << "========================================\n";
    std::cout << "\nPress Ctrl+C to exit...\n";
}

void WriteMarketStateJson(const Orderbook &orderbook, const std::string &filename, int iteration) {
    try {
        auto infos = orderbook.GetOrderInfos();
        const auto &bids = infos.GetBids();
        const auto &asks = infos.GetAsks();

        json state;
        
        // Get best prices
        if (!bids.empty() && !asks.empty()) {
            state["bid_price"] = bids[0].price_;
            state["ask_price"] = asks[0].price_;
            
            double spread = (asks[0].price_ - bids[0].price_) / 100.0;
            double midPrice = (bids[0].price_ + asks[0].price_) / 200.0;
            double spreadBps = (spread / midPrice) * 10000;
            
            state["mid_price"] = (bids[0].price_ + asks[0].price_) / 2.0;
            state["spread"] = asks[0].price_ - bids[0].price_;
            state["spread_bps"] = spreadBps;
        }

        state["iteration"] = iteration;

        // Add all bid levels
        json bids_arr = json::array();
        for (const auto &bid : bids) {
            bids_arr.push_back({
                {"price", bid.price_},
                {"quantity", bid.quantity_}
            });
        }
        state["bids"] = bids_arr;

        // Add all ask levels
        json asks_arr = json::array();
        for (const auto &ask : asks) {
            asks_arr.push_back({
                {"price", ask.price_},
                {"quantity", ask.quantity_}
            });
        }
        state["asks"] = asks_arr;

        // Write to file
        std::ofstream file(filename);
        file << state.dump(2);
        file.close();
    } catch (const std::exception &e) {
        std::cerr << "Error writing market state: " << e.what() << "\n";
    }
}
