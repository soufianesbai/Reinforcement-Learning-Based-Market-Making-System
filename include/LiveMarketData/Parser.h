#pragma once

#include <string>
#include "MarketData/MarketDataFeed.h"

BookSnapshotMessage ParseBinanceSnapshot(const std::string &jsonStr);
