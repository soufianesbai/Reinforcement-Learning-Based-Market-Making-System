#!/bin/bash
# Quick build and test script for HFT System

set -e

echo "==================================="
echo "HFT System Build & Test Script"
echo "==================================="

# Clean previous build
echo "Cleaning previous build..."
rm -rf build
mkdir build

# Configure
echo "Configuring CMake..."
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
cmake --build . -j$(nproc)

# Run tests
echo ""
echo "Running tests..."
ctest --output-on-failure

# Run benchmarks
echo ""
echo "Running benchmarks..."
echo "--- OrderBook Benchmark ---"
./benchmarks/bench_orderbook

echo ""
echo "--- Timestamp Benchmark ---"
./benchmarks/bench_timestamp

echo ""
echo "==================================="
echo "Build & Test Complete!"
echo "==================================="
echo ""
echo "To run the HFT system:"
echo "  ./build/hft_system"
