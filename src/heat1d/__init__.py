"""Finite-difference solvers for the 1D heat equation."""

from .linalg import conjugate_gradient, solve_tridiagonal
from .problem import HeatProblem
from .solvers import (
    Solution,
    crank_nicolson_condition_number,
    max_error,
    solve_crank_nicolson,
    solve_explicit,
)

__all__ = [
    "HeatProblem",
    "Solution",
    "conjugate_gradient",
    "solve_tridiagonal",
    "solve_explicit",
    "solve_crank_nicolson",
    "crank_nicolson_condition_number",
    "max_error",
]
