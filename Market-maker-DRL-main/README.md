# High-Frequency Trading System

A high-performance, low-latency trading system skeleton built with C++20 and CMake.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Modern C++20**: Leveraging latest C++ features for performance and safety
- **Low Latency**: Optimized for microsecond-level response times
- **Type Safety**: Strong typing and compile-time checks
- **Extensible**: Easy to add new strategies and data feeds

## Directory Structure

```
.
├── src/
│   ├── common/           # Common utilities (logger, types, timestamp)
│   ├── network/          # Network abstractions (TCP/UDP sockets)
│   ├── market_data/      # Market data feed handlers
│   ├── orderbook/        # Order book implementation
│   ├── strategy/         # Trading strategy interface
│   └── main.cpp          # Application entry point
├── tests/                # Unit tests
├── benchmarks/           # Performance benchmarks
├── build/                # Build output (gitignored)
└── CMakeLists.txt        # Root CMake configuration

```

## Building

### Prerequisites

- CMake 3.20 or higher
- C++20 compatible compiler (GCC 11+, Clang 13+)
- Linux (for best performance with low-latency optimizations)

### Build Instructions

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build . -j$(nproc)

# Run the main executable
./hft_system
```

### Build Options

- `BUILD_TESTS=ON/OFF` - Build unit tests (default: ON)
- `BUILD_BENCHMARKS=ON/OFF` - Build performance benchmarks (default: ON)

```bash
cmake -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON ..
```

## Running

### Main Application

```bash
./build/hft_system
```

Press `Ctrl+C` to gracefully shutdown the system.

### Tests

```bash
cd build
ctest --verbose
# Or run individual tests
./tests/test_orderbook
./tests/test_types
```

### Benchmarks

```bash
./build/benchmarks/bench_orderbook
./build/benchmarks/bench_timestamp
```

## Architecture

### Common Module
- **Logger**: Thread-safe logging with configurable levels
- **Types**: Core trading types (Order, Price, Quantity, etc.)
- **Timestamp**: High-resolution timestamp utilities

### Network Module
- **Socket**: Abstract socket interface for TCP/UDP
- **Connection**: Connection management with message handlers

### Market Data Module
- **FeedHandler**: Market data feed abstraction
- **Types**: Quote, Trade, OrderBookUpdate structures

### Order Book Module
- **OrderBook**: High-performance order book implementation
- **PriceLevel**: Price level aggregation

### Strategy Module
- **Strategy**: Base strategy interface
- **DummyStrategy**: Example strategy implementation

## Performance Considerations

- **Zero-copy**: Minimize data copying with `std::span` and references
- **Lock-free**: Use lock-free data structures where possible
- **Memory pools**: Pre-allocate memory to avoid runtime allocations
- **CPU affinity**: Pin threads to specific CPU cores
- **NUMA awareness**: Optimize for NUMA architectures
- **Compiler optimizations**: `-O3 -march=native -flto`

## Code Style

The project uses `.clang-format` for consistent code formatting:

```bash
clang-format -i src/**/*.cpp src/**/*.hpp
```

## Next Steps

1. Implement concrete network socket classes (TCP/UDP)
2. Add market data feed parsers (FIX, ITCH, etc.)
3. Implement order gateway for exchange connectivity
4. Add risk management layer
5. Implement concrete trading strategies
6. Add configuration management
7. Integrate with monitoring/metrics system
8. Add comprehensive unit and integration tests

## License

[Add your license here]