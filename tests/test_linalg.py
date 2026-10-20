"""Checks the hand-written linear solvers against numpy.linalg.solve."""

import numpy as np

from heat1d import conjugate_gradient, crank_nicolson_condition_number, solve_tridiagonal


def test_conjugate_gradient_matches_numpy_solve():
    rng = np.random.default_rng(0)
    n = 40
    B = rng.standard_normal((n, n))
    A = B @ B.T + n * np.eye(n)  # symmetric positive definite
    b = rng.standard_normal(n)
    x, iters = conjugate_gradient(lambda v: A @ v, b, rtol=1e-12)
    assert np.allclose(x, np.linalg.solve(A, b), rtol=1e-10, atol=1e-12)
    assert 0 < iters <= n


def test_thomas_matches_numpy_solve():
    rng = np.random.default_rng(1)
    n = 30
    lower = rng.uniform(-1, 1, n - 1)
    upper = rng.uniform(-1, 1, n - 1)
    diag = 4.0 + rng.uniform(0, 1, n)  # diagonally dominant, so no pivoting needed
    rhs = rng.standard_normal(n)
    A = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
    assert np.allclose(solve_tridiagonal(lower, diag, upper, rhs), np.linalg.solve(A, rhs))


def test_condition_number_formula_matches_numpy():
    nx, r = 50, 7.5
    m = nx - 1
    A = -2 * np.eye(m) + np.eye(m, k=1) + np.eye(m, k=-1)
    M = np.eye(m) - 0.5 * r * A
    assert np.isclose(crank_nicolson_condition_number(r, nx), np.linalg.cond(M))
