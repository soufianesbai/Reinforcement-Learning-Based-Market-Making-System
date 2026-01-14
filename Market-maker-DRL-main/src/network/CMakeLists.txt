add_library(network INTERFACE)

target_include_directories(network INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(network INTERFACE common)

target_compile_features(network INTERFACE cxx_std_20)

# Header files
set(NETWORK_HEADERS
    include/network/socket.hpp
    include/network/connection.hpp
)

target_sources(network INTERFACE ${NETWORK_HEADERS})
