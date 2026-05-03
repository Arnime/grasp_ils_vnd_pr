# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

include(FetchContent)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG        v4.3.11
)
FetchContent_MakeAvailable(nanobench)

add_executable(givp_benchmarks benchmarks.cpp)
target_link_libraries(givp_benchmarks PRIVATE givp::givp nanobench)
target_compile_features(givp_benchmarks PRIVATE cxx_std_17)

add_executable(givp_literature_comparison run_literature_comparison.cpp)
target_link_libraries(givp_literature_comparison PRIVATE givp::givp)
target_compile_features(givp_literature_comparison PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(givp_benchmarks PRIVATE /O2)
    target_compile_options(givp_literature_comparison PRIVATE /O2)
else()
    target_compile_options(givp_benchmarks PRIVATE -O3)
    target_compile_options(givp_literature_comparison PRIVATE -O3)
endif()
