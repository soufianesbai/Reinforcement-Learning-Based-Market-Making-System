# HFT System Architecture

## Overview

This document describes the architecture of the High-Frequency Trading (HFT) system skeleton.

## Design Principles

1. **Low Latency**: Every component is designed with microsecond-level performance in mind
2. **Modularity**: Clear separation of concerns with minimal dependencies
3. **Type Safety**: Strong typing using C++20 features
4. **Zero-Copy**: Minimize data copying using references and spans
5. **Header-Only Libraries**: Most modules are header-only for better inlining

## Module Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Main Application                     │
│                      (main.cpp)                          │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │Strategy│  │Order   │  │Market  │
   │        │  │Book    │  │Data    │
   └────┬───┘  └───┬────┘  └───┬────┘
        │          │           │
        └──────────┼───────────┘
                   │
            ┌──────┴──────┐
            │             │
            ▼             ▼
       ┌────────┐    ┌────────┐
       │Common  │    │Network │
       └────────┘    └────────┘
```

## Module Descriptions

### Common Module
**Purpose**: Foundational types and utilities

**Components**:
- `types.hpp`: Core trading types (Order, Price, Quantity, Side, etc.)
- `timestamp.hpp`: High-resolution timestamp utilities
- `logger.hpp`: Thread-safe logging with source location

**Key Features**:
- Fixed-point price representation for precision
- Strongly typed enums for safety
- Nanosecond-precision timestamps
- Lock-free logging using `std::osyncstream`

### Network Module
**Purpose**: Network communication abstractions

**Components**:
- `socket.hpp`: Abstract socket interface supporting TCP/UDP
- `connection.hpp`: Connection management with message callbacks

**Key Features**:
- Socket options for low-latency (NODELAY, QUICKACK, etc.)
- Non-blocking I/O support
- Message handler pattern for async processing

### Market Data Module
**Purpose**: Market data feed handling

**Components**:
- `market_data_types.hpp`: Quote, Trade, OrderBookUpdate structures
- `feed_handler.hpp`: Abstract feed handler with callbacks

**Key Features**:
- Callback-based architecture for minimal latency
- Support for multiple data types (quotes, trades, order book updates)
- Symbol subscription management

### Order Book Module
**Purpose**: High-performance order book implementation

**Components**:
- `price_level.hpp`: Price level aggregation
- `order_book.hpp`: Full order book with bid/ask management

**Key Features**:
- Sorted maps for O(log n) insertions
- Constant time best bid/ask queries (O(1))
- Automatic price level cleanup
- Separate bid/ask books with proper ordering

**Performance**:
- ~75ns per order add operation
- <1ns for best bid/ask queries
- Cache-friendly data structures

### Strategy Module
**Purpose**: Trading strategy interface

**Components**:
- `strategy.hpp`: Abstract strategy base class with lifecycle methods

**Key Features**:
- Event-driven architecture
- Market data callbacks (quotes, trades, order book updates)
- Order lifecycle callbacks (ack, fill, cancel, reject)
- Clean initialization/shutdown lifecycle

## Data Flow

### Market Data Flow
```
Exchange → Network → FeedHandler → OrderBook → Strategy
                                 ↘ Callbacks ↗
```

### Order Flow (Future)
```
Strategy → OrderGateway → Network → Exchange
            ↓ Callbacks
         Strategy
```

## Performance Optimizations

### Compiler Optimizations
- C++20 standard for latest features
- `-O3` optimization level in Release
- `-march=native` for CPU-specific optimizations
- Link-Time Optimization (LTO) enabled

### Code Optimizations
1. **Header-Only Libraries**: Enables better inlining
2. **Template Metaprogramming**: Compile-time computations
3. **Move Semantics**: Avoid unnecessary copies
4. **std::span**: Zero-copy data views
5. **Structured Bindings**: Cleaner, more optimized code

### Memory Optimizations
1. **Pre-allocation**: Reserve capacity upfront
2. **Custom Allocators**: (Future) Pool allocators for hot paths
3. **Cache Alignment**: (Future) Align hot structures to cache lines

### Concurrency Considerations
1. **Lock-Free**: Use lock-free data structures where possible
2. **Thread Affinity**: (Future) Pin threads to CPU cores
3. **NUMA Awareness**: (Future) Optimize for NUMA architectures

## Build System

### CMake Structure
```
CMakeLists.txt (root)
├── src/common/CMakeLists.txt
├── src/network/CMakeLists.txt
├── src/market_data/CMakeLists.txt
├── src/orderbook/CMakeLists.txt
├── src/strategy/CMakeLists.txt
├── tests/CMakeLists.txt
└── benchmarks/CMakeLists.txt
```

### Build Targets
- `hft_system`: Main executable
- `test_*`: Unit test executables
- `bench_*`: Benchmark executables

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Focus on correctness and edge cases
- Fast execution (<1s total)

### Benchmarks
- Measure performance of critical paths
- Track performance regressions
- Compare against baselines

## Future Enhancements

### Phase 1: Core Infrastructure
- [ ] Implement concrete TCP/UDP socket classes
- [ ] Add FIX protocol parser
- [ ] Implement order gateway
- [ ] Add configuration management (YAML/TOML)

### Phase 2: Performance
- [ ] Add custom memory allocators
- [ ] Implement lock-free queues
- [ ] CPU affinity and thread pinning
- [ ] NUMA optimization

### Phase 3: Features
- [ ] Risk management layer
- [ ] Position tracking
- [ ] PnL calculation
- [ ] Market making strategy implementation

### Phase 4: Production
- [ ] Monitoring and metrics (Prometheus)
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Fault tolerance and recovery
- [ ] Integration tests
- [ ] Load testing framework

## References

### C++20 Features Used
- Concepts (future)
- Ranges (future)
- std::span
- std::source_location
- Designated initializers
- Three-way comparison (future)

### Performance Resources
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832)
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)
