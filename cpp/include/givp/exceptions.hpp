// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <stdexcept>
#include <string>

namespace givp {

/// Base exception for all GIVP errors.
class GivpError : public std::runtime_error {
public:
    explicit GivpError(const std::string& msg) : std::runtime_error(msg) {}
};

class InvalidBounds : public GivpError {
public:
    explicit InvalidBounds(const std::string& msg) : GivpError("invalid bounds: " + msg) {}
};

class InvalidInitialGuess : public GivpError {
public:
    explicit InvalidInitialGuess(const std::string& msg)
        : GivpError("invalid initial guess: " + msg) {}
};

class InvalidConfig : public GivpError {
public:
    explicit InvalidConfig(const std::string& msg) : GivpError("invalid config: " + msg) {}
};

class EmptyPool : public GivpError {
public:
    explicit EmptyPool(const std::string& msg) : GivpError("empty elite pool: " + msg) {}
};

} // namespace givp
