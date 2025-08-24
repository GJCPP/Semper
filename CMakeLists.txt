cmake_minimum_required(VERSION 3.10)

include(FetchContent)

project(LogUP LANGUAGES CXX)

# set(CMAKE_CXX_COMPILER g++-13)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
include_directories(goldilocks/src)

# Use cnpy directory under source tree
set(CNPY_SOURCE_DIR ${CMAKE_SOURCE_DIR}/cnpy)
set(CNPY_BINARY_DIR ${CMAKE_BINARY_DIR}/cnpy-build)

# Only clone if not already present
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

# Now build it as part of your project
add_subdirectory(${CNPY_SOURCE_DIR} ${CNPY_BINARY_DIR})
include_directories(${CNPY_SOURCE_DIR})

# build goldilocks.a
file(GLOB DEP_SOURCES
    goldilocks/src/*.cpp
)
add_library(goldilocks STATIC ${DEP_SOURCES})
link_directories(${CMAKE_SOURCE_DIR}/cnpy/build)
target_compile_options(goldilocks PRIVATE -O3 -mavx2 -MMD -MP -pthread)
# target_compile_options(goldilocks PRIVATE -O2 -mavx2 -MMD -MP -pthread)
find_package(OpenMP REQUIRED)
target_link_libraries(goldilocks PRIVATE
    OpenMP::OpenMP_CXX
    pthread
)

file(GLOB SOURCES
    src/*.cpp
)

# target test
# add_executable(test test.cpp)
# target_sources(test PRIVATE ${SOURCES})
# target_compile_options(test PRIVATE -Wall -O3 -mavx2 -fopenmp)
# target_link_libraries(test PRIVATE pthread gmp ssl crypto goldilocks)

# target bench
add_executable(bench bench.cpp)
target_sources(bench PRIVATE ${SOURCES})



# target_compile_options(bench PRIVATE -Wall -O3 -mavx2 -fopenmp)
# target_compile_options(bench PRIVATE -Wall -mavx2 -fopenmp) # Debugging
# set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

target_compile_options(bench PRIVATE
    $<$<CONFIG:Release>:-Wall -O3 -mavx2>
    $<$<CONFIG:Debug>:-Wall -g -DDEBUG -mavx2 -fopenmp>)

# Use "cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Debug .." to build in debug mode
# Use "cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Release .." to build in release mode

target_link_libraries(bench PRIVATE cnpy pthread gmp ssl crypto goldilocks)


add_custom_target(run
    COMMAND bench
    DEPENDS bench
    COMMENT "Running ben"
)
