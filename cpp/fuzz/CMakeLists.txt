# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

add_executable(givp_fuzz fuzz_givp.cpp)
target_link_libraries(givp_fuzz PRIVATE givp::givp)
target_compile_features(givp_fuzz PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(givp_fuzz PRIVATE /W4)
else()
    target_compile_options(givp_fuzz PRIVATE -Wall -Wextra -Wpedantic)
endif()
