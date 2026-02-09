#include "LiveMarketData/Parser.h"

#include <chrono>
#include <string>
#include <nlohmann/json.hpp>

BookSnapshotMessage ParseBinanceSnapshot(const std::string &jsonStr) {
    auto json = nlohmann::json::parse(jsonStr);

    BookSnapshotMessage snapshot;
    snapshot.timestamp = std::chrono::system_clock::now();
    snapshot.sequenceNumber = json["lastUpdateId"];

    // Parse bids (buy orders)
    for (const auto &bid: json["bids"]) {
        SnapshotLevel level;
        level.price = static_cast<Price>(std::stod(bid[0].get<std::string>()) * 100); // Convert to cents
        level.quantity = static_cast<Quantity>(std::stod(bid[1].get<std::string>()) * 100); // Convert to smallest unit
        level.orderCount = 1; // Binance doesn't provide this
        snapshot.bids.push_back(level);
    }

    // Parse asks (sell orders)
    for (const auto &ask: json["asks"]) {
        SnapshotLevel level;
        level.price = static_cast<Price>(std::stod(ask[0].get<std::string>()) * 100);
        level.quantity = static_cast<Quantity>(std::stod(ask[1].get<std::string>()) * 100);
        level.orderCount = 1;
        snapshot.asks.push_back(level);
    }

    return snapshot;
}
