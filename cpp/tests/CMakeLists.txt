# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(givp_tests
    test_basic.cpp
    test_config.cpp
    test_convergence.cpp
    test_advanced.cpp
)

target_link_libraries(givp_tests PRIVATE givp::givp Catch2::Catch2WithMain)
target_compile_features(givp_tests PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(givp_tests PRIVATE /W4 /WX)
else()
    target_compile_options(givp_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(Catch)
catch_discover_tests(givp_tests)
