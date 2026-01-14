cmake_minimum_required(VERSION 3.20)
project(HFTSystem VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings and optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -march=native
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-flto>
    )
endif()

# Low-latency flags
add_compile_definitions(
    $<$<CONFIG:Release>:NDEBUG>
)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# Add subdirectories for modules
add_subdirectory(src/common)
add_subdirectory(src/network)
add_subdirectory(src/market_data)
add_subdirectory(src/orderbook)
add_subdirectory(src/strategy)

# Main executable
add_executable(hft_system src/main.cpp)
target_link_libraries(hft_system PRIVATE
    common
    network
    market_data
    orderbook
    strategy
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
