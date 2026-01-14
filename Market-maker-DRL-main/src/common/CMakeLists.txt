add_library(common INTERFACE)

target_include_directories(common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(common INTERFACE cxx_std_20)

# Header files
set(COMMON_HEADERS
    include/common/logger.hpp
    include/common/types.hpp
    include/common/timestamp.hpp
)

# Make headers visible in IDEs
target_sources(common INTERFACE ${COMMON_HEADERS})
