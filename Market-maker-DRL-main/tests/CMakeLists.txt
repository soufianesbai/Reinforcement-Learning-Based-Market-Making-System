# Tests configuration
# Note: This is a basic setup. You would typically use a testing framework like Google Test

add_executable(test_orderbook test_orderbook.cpp)
target_link_libraries(test_orderbook PRIVATE orderbook common)

add_executable(test_types test_types.cpp)
target_link_libraries(test_types PRIVATE common)

# Add tests
add_test(NAME OrderBookTest COMMAND test_orderbook)
add_test(NAME TypesTest COMMAND test_types)
