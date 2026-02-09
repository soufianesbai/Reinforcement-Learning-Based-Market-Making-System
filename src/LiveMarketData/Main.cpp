#include <chrono>
#include <curl/curl.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "LiveMarketData/AgentOrders.h"
#include "LiveMarketData/Display.h"
#include "LiveMarketData/Fetcher.h"
#include "LiveMarketData/Parser.h"
#include "OrderBook/OrderBook.h"

int main(int argc, char *argv[]) {
    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);

    // Default to SOL/USDT, but allow command line override
    std::string symbol = "SOLUSDT";
    int refreshInterval = 1; // seconds
    int displayLevels = 50;

    if (argc > 1) {
        symbol = argv[1];
    }
    if (argc > 2) {
        refreshInterval = std::stoi(argv[2]);
    }
    if (argc > 3) {
        displayLevels = std::stoi(argv[3]);
    }

    std::cout << "========================================\n";
    std::cout << "  Binance Live Market Data Feed\n";
    std::cout << "========================================\n";
    std::cout << "Symbol: " << symbol << "\n";
    std::cout << "Refresh Interval: " << refreshInterval << " seconds\n";
    std::cout << "Display Levels: " << displayLevels << "\n";
    std::cout << "\nConnecting to Binance API...\n\n";
    std::cout << "Usage: ./LiveMarketData [SYMBOL] [REFRESH_SECONDS] [LEVELS]\n";
    std::cout << "Example: ./LiveMarketData ETHUSDT 1 15\n\n";

    std::this_thread::sleep_for(std::chrono::seconds(2));

    Orderbook orderbook;

    try {
        // Main loop, fetch and display orderbook
        int iteration = 0;
        while (true) {
            iteration++;
            // Fetch snapshot from Binance
            std::string jsonResponse = FetchBinanceOrderbook(symbol, displayLevels);

            if (jsonResponse.empty()) {
                std::cerr << "Failed to fetch orderbook data\n";
                std::this_thread::sleep_for(std::chrono::seconds(refreshInterval));
                continue;
            }

            // Parse and process
            try {
                // Charger les nouveaux ordres de l'agent
                LoadAgentOrders("../data/agent_orders.json");

                BookSnapshotMessage snapshot = ParseBinanceSnapshot(jsonResponse);

                if (orderbook.ProcessMarketData(snapshot)) {
                    // Réinjecter les ordres de l'agent après le snapshot
                    ReinjectAgentOrders(orderbook);

                    // Écrire l'état du marché pour l'agent Python
                    WriteMarketStateJson(orderbook, "../data/market_state.json", iteration);

                    // Display the orderbook
                    PrintOrderbook(orderbook, symbol, displayLevels);
                } else {
                    std::cerr << "Failed to process market data\n";
                }
            } catch (const nlohmann::json::exception &e) {
                std::cerr << "JSON parsing error: " << e.what() << "\n";
                std::cerr << "Response: " << jsonResponse.substr(0, 200) << "...\n";
            } catch (const std::exception &e) {
                std::cerr << "Error processing data: " << e.what() << "\n";
            }

            // Wait before next update
            std::this_thread::sleep_for(std::chrono::seconds(refreshInterval));
        }
    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        curl_global_cleanup();
        return 1;
    }

    // Cleanup
    curl_global_cleanup();

    return 0;
}
