# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Custom exceptions for the GIVP package."""

abstract type GivpError <: Exception end

struct InvalidBoundsError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::InvalidBoundsError) = print(io, "InvalidBoundsError: ", e.msg)

struct InvalidInitialGuessError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::InvalidInitialGuessError) = print(io, "InvalidInitialGuessError: ", e.msg)

struct InvalidConfigError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::InvalidConfigError) = print(io, "InvalidConfigError: ", e.msg)

struct EvaluatorError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::EvaluatorError) = print(io, "EvaluatorError: ", e.msg)

struct EmptyPoolError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::EmptyPoolError) = print(io, "EmptyPoolError: ", e.msg)
