cmake_minimum_required(VERSION 3.10)
project(LogUP LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
include_directories(goldilocks/src)

# --- CNPY setup ---
set(CNPY_SOURCE_DIR ${CMAKE_SOURCE_DIR}/cnpy)
set(CNPY_BINARY_DIR ${CMAKE_BINARY_DIR}/cnpy-build)

if(NOT EXISTS ${CNPY_SOURCE_DIR}/CMakeLists.txt)
    message(STATUS "Cloning cnpy into ${CNPY_SOURCE_DIR}")
    execute_process(
        COMMAND git clone git@github.com:rogersce/cnpy.git ${CNPY_SOURCE_DIR}
        RESULT_VARIABLE GIT_CLONE_RESULT
    )
    if(NOT GIT_CLONE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to clone cnpy from GitHub")
    endif()
endif()

add_subdirectory(${CNPY_SOURCE_DIR} ${CNPY_BINARY_DIR})
include_directories(${CNPY_SOURCE_DIR})

# --- goldilocks library ---
file(GLOB DEP_SOURCES goldilocks/src/*.cpp)
add_library(goldilocks STATIC ${DEP_SOURCES})
target_compile_options(goldilocks PRIVATE -O3 -march=native -MMD -MP -pthread)
find_package(OpenMP REQUIRED)
target_link_libraries(goldilocks PRIVATE OpenMP::OpenMP_CXX pthread)

# --- main sources ---
file(GLOB SOURCES src/*.cpp)
add_executable(bench bench.cpp)
target_sources(bench PRIVATE ${SOURCES})

# --- Build type specific flags ---
target_compile_options(bench PRIVATE
    $<$<CONFIG:Release>:-Wall -O3 -march=native -fopenmp>
    $<$<CONFIG:Debug>:-Wall -g -O1 -march=native -fno-omit-frame-pointer -fsanitize=address,undefined -fopenmp>
)

# Use "cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Debug .." to build in debug mode
# Use "cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Release .." to build in release mode

# Add sanitizer *link* flags as well for Debug builds
target_link_options(bench PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# --- Libraries ---
target_link_libraries(bench PRIVATE cnpy pthread gmp ssl crypto goldilocks)

# --- Custom target ---
add_custom_target(run
    COMMAND bench
    DEPENDS bench
    COMMENT "Running bench"
)
