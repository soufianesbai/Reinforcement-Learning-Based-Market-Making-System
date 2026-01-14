add_library(market_data INTERFACE)

target_include_directories(market_data INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(market_data INTERFACE common network)

target_compile_features(market_data INTERFACE cxx_std_20)

# Header files
set(MARKET_DATA_HEADERS
    include/market_data/feed_handler.hpp
    include/market_data/market_data_types.hpp
)

target_sources(market_data INTERFACE ${MARKET_DATA_HEADERS})
