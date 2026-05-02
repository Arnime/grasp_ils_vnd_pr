# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""
    GivpError

Abstract supertype for all GIVPOptimizer exceptions.
All package-specific errors are subtypes of this type.
"""
abstract type GivpError <: Exception end

"""
    InvalidBoundsError(msg)

Thrown when the bounds argument is malformed (e.g. `lb > ub`, wrong length,
or non-finite values).
"""
struct InvalidBoundsError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::InvalidBoundsError) = print(io, "InvalidBoundsError: ", e.msg)

"""
    InvalidInitialGuessError(msg)

Thrown when `initial_guess` is outside the declared bounds or has the wrong
length.
"""
struct InvalidInitialGuessError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::InvalidInitialGuessError) =
    print(io, "InvalidInitialGuessError: ", e.msg)

"""
    InvalidConfigError(msg)

Thrown when a [`GIVPConfig`](@ref) field holds an out-of-range or inconsistent
value.
"""
struct InvalidConfigError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::InvalidConfigError) = print(io, "InvalidConfigError: ", e.msg)

"""
    EvaluatorError(msg)

Thrown when the user-supplied objective function raises an unexpected
exception during evaluation.
"""
struct EvaluatorError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::EvaluatorError) = print(io, "EvaluatorError: ", e.msg)

"""
    EmptyPoolError(msg)

Thrown when the elite pool is unexpectedly empty during path relinking.
"""
struct EmptyPoolError <: GivpError
    msg::String
end
Base.showerror(io::IO, e::EmptyPoolError) = print(io, "EmptyPoolError: ", e.msg)
