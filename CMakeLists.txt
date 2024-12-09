cmake_minimum_required(VERSION 3.20)
project(VecFit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

set(DEFAULT_OUTPUT_SUFFIX "eda240709")
add_definitions(-DDEFAULT_OUTPUT_SUFFIX="${DEFAULT_OUTPUT_SUFFIX}")

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_definitions(-DGIT_COMMIT_HASH="${GIT_COMMIT_HASH}")

option(USE_INTELCC "Enable icpx compiler" OFF)
option(USE_MKL "Use MKL as Eigen's backend" ON)
option(GEN_PYTHON_LIB "Generate Python Lib" OFF)
option(USE_JEMALLOC "Enable jemalloc" OFF)

if(GEN_PYTHON_LIB AND USE_JEMALLOC)
    message(WARNING "Do not use jemalloc due to python lib generation is enabled")
    set(USE_JEMALLOC OFF)
endif()

if(USE_INTELCC)
    set(CMAKE_CXX_COMPILER ${ONEAPI_ROOT}/compiler/latest/bin/icpx)
endif()

if(GEN_PYTHON_LIB)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

include(CheckIPOSupported)

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT USE_INTELCC)
    check_ipo_supported(RESULT result OUTPUT output)

    if(result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

if(USE_INTELCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(fmt)
find_package(spdlog)

add_definitions(-DSPDLOG_ACTIVE)
add_definitions(-DPROJECT_VERSION="${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
add_definitions(-DTIME_MEASURE)

if(USE_MKL)
    set(MKLROOT ${ONEAPI_ROOT}/mkl/2024.2)
    message("MKLROOT: ${MKLROOT}")
    include_directories(${MKLROOT}/include)
    add_definitions(-DEIGEN_USE_MKL_ALL)
endif()

add_subdirectory(lib)
add_subdirectory(tools)