#pragma once

#include <cstddef>
#include <string>

size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *userp);
std::string FetchBinanceOrderbook(const std::string &symbol, int limit = 20);
