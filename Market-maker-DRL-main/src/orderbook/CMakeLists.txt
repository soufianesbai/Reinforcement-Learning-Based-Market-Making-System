add_library(orderbook INTERFACE)

target_include_directories(orderbook INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(orderbook INTERFACE common)

target_compile_features(orderbook INTERFACE cxx_std_20)

# Header files
set(ORDERBOOK_HEADERS
    include/orderbook/order_book.hpp
    include/orderbook/price_level.hpp
)

target_sources(orderbook INTERFACE ${ORDERBOOK_HEADERS})
