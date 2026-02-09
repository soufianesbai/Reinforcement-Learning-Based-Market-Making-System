#pragma once

#include <chrono>
#include <string>
#include "OrderBook/OrderBook.h"

std::string FormatTimestamp(const std::chrono::system_clock::time_point &tp);
void PrintOrderbook(const Orderbook &orderbook, const std::string &symbol, int levels = 10);
void WriteMarketStateJson(const Orderbook &orderbook, const std::string &filename, int iteration);
