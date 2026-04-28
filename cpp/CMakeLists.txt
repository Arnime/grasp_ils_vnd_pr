# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
project(givp VERSION 0.5.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Header-only library target ──────────────────────────────────────────────
add_library(givp INTERFACE)
add_library(givp::givp ALIAS givp)

target_include_directories(givp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(givp INTERFACE cxx_std_17)

# Compiler warnings for own code
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(MSVC)
        target_compile_options(givp INTERFACE /W4)
    else()
        target_compile_options(givp INTERFACE -Wall -Wextra -Wpedantic)
    endif()
endif()

# ── Optional subdirectories ─────────────────────────────────────────────────
option(GIVP_BUILD_TESTS      "Build tests"      ON)
option(GIVP_BUILD_BENCHMARKS "Build benchmarks" ON)

if(GIVP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(GIVP_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ── Install rules ────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS givp EXPORT givpTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT givpTargets
    FILE givpTargets.cmake
    NAMESPACE givp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/givp
)
