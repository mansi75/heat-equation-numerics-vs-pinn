"""Linear solvers written in plain NumPy.

conjugate_gradient: iterative solver for symmetric positive definite systems.
    It only needs a function that multiplies a vector by the matrix, so the
    matrix never has to be stored. This is the method that scales to large
    2D/3D problems.

solve_tridiagonal: direct O(n) solver (Thomas algorithm) for tridiagonal
    systems. In 1D this is the cheapest exact option, so it is used as a
    reference to check the conjugate gradient results.
"""

import numpy as np


def conjugate_gradient(apply_A, b, x0=None, rtol=1e-12, max_iter=None):
    """Solve A x = b for symmetric positive definite A.

    apply_A : function returning A @ v for a vector v
    x0      : starting guess (the previous time step is a good one)
    rtol    : stop when ||b - A x|| <= rtol * ||b||

    Returns (x, number_of_iterations).
    """
    b = np.asarray(b, dtype=float)
    n = b.size
    max_iter = 10 * n if max_iter is None else max_iter
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        return np.zeros(n), 0

    r = b - apply_A(x)  # residual: how far we are from solving the system
    p = r.copy()  # search direction
    rs = r @ r
    for k in range(max_iter + 1):
        if np.sqrt(rs) <= rtol * b_norm:
            return x, k
        if k == max_iter:
            break
        Ap = apply_A(p)
        step = rs / (p @ Ap)  # best step length along p
        x += step * p
        r -= step * Ap
        rs_new = r @ r
        p = r + (rs_new / rs) * p  # next direction, A-orthogonal to the previous ones
        rs = rs_new
    raise RuntimeError(f"conjugate gradient did not converge in {max_iter} iterations")


def solve_tridiagonal(lower, diag, upper, rhs):
    """Thomas algorithm for a tridiagonal system.

    lower: sub-diagonal (length n-1), diag: main diagonal (length n),
    upper: super-diagonal (length n-1).
    """
    n = len(diag)
    c = np.zeros(n - 1)
    d = np.zeros(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x
