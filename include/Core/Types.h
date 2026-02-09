#pragma once

#include <cstdint>
#include <list>
#include <memory>

class Order;

using Price = std::int32_t; // For readability
using Quantity = std::uint32_t;
using OrderId = std::uint64_t;
using OrderPointer = std::shared_ptr<Order>;
using OrderPointers = std::list<OrderPointer>;
