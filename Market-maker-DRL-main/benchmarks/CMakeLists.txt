# Benchmarks configuration
# Note: This is a basic setup. You would typically use Google Benchmark

add_executable(bench_orderbook bench_orderbook.cpp)
target_link_libraries(bench_orderbook PRIVATE orderbook common)
target_compile_options(bench_orderbook PRIVATE -O3 -march=native)

add_executable(bench_timestamp bench_timestamp.cpp)
target_link_libraries(bench_timestamp PRIVATE common)
target_compile_options(bench_timestamp PRIVATE -O3 -march=native)
