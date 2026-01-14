add_library(strategy INTERFACE)

target_include_directories(strategy INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(strategy INTERFACE common orderbook market_data)

target_compile_features(strategy INTERFACE cxx_std_20)

# Header files
set(STRATEGY_HEADERS
    include/strategy/strategy.hpp
)

target_sources(strategy INTERFACE ${STRATEGY_HEADERS})
