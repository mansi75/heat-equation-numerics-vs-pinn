"""Finite-difference solvers for the 1D heat equation.

Both solvers replace u_xx with the standard second difference

    u_xx(x_i) ~ (u[i+1] - 2 u[i] + u[i-1]) / dx^2

and then step forward in time. They differ in how they take each step.

Explicit (forward Euler in time):
    u_new = u + r * (second difference of u),   r = alpha * dt / dx^2
    Cheap per step, but only stable when r <= 1/2, so dt must shrink
    like dx^2 as the grid gets finer.

Crank-Nicolson (average of the old and new time levels):
    (I - r/2 A) u_new = (I + r/2 A) u
    Needs a linear solve every step, but is stable for any dt and is
    second-order accurate in time as well as in space.
"""

from dataclasses import dataclass, field

import numpy as np

from .linalg import conjugate_gradient, solve_tridiagonal
from .problem import HeatProblem


@dataclass
class Solution:
    x: np.ndarray  # grid points
    u: np.ndarray  # temperature at the final time
    dt: float  # time step actually used
    r: float  # alpha * dt / dx^2, the number that controls stability
    n_steps: int
    times: list = field(default_factory=list)  # times of saved snapshots
    history: list = field(default_factory=list)  # saved snapshots of u
    cg_iterations: list = field(default_factory=list)  # CG iterations per step


def _step_count(t_final, dt):
    """Number of steps that lands exactly on t_final without exceeding dt."""
    n_steps = int(np.ceil(t_final / dt - 1e-9))
    return n_steps, t_final / n_steps


def _apply_A(v):
    """Second difference of the interior values, with u = 0 at both ends.

    This is the tridiagonal matrix A = tridiag(1, -2, 1) applied to v,
    without ever building the matrix.
    """
    Av = -2.0 * v
    Av[1:] += v[:-1]
    Av[:-1] += v[1:]
    return Av


def solve_explicit(problem: HeatProblem, nx, dt, save_every=None, check_stability=True):
    x, dx = problem.grid(nx)
    n_steps, dt = _step_count(problem.t_final, dt)
    r = problem.alpha * dt / dx**2
    if check_stability and r > 0.5:
        raise ValueError(
            f"explicit scheme is unstable for r = {r:.3f} > 0.5; "
            f"use dt <= {0.5 * dx**2 / problem.alpha:.3e}"
        )

    u = problem.initial_condition(x)
    u[0] = u[-1] = 0.0
    sol = Solution(x=x, u=u, dt=dt, r=r, n_steps=n_steps)
    if save_every:
        sol.times.append(0.0)
        sol.history.append(u.copy())

    for n in range(1, n_steps + 1):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        if save_every and n % save_every == 0:
            sol.times.append(n * dt)
            sol.history.append(u.copy())
    return sol


def solve_crank_nicolson(
    problem: HeatProblem, nx, dt, linear_solver="cg", save_every=None, rtol=1e-12
):
    x, dx = problem.grid(nx)
    n_steps, dt = _step_count(problem.t_final, dt)
    r = problem.alpha * dt / dx**2

    u = problem.initial_condition(x)
    u[0] = u[-1] = 0.0
    sol = Solution(x=x, u=u, dt=dt, r=r, n_steps=n_steps)
    if save_every:
        sol.times.append(0.0)
        sol.history.append(u.copy())

    def apply_M(v):  # left-hand matrix (I - r/2 A), symmetric positive definite
        return v - 0.5 * r * _apply_A(v)

    m = nx - 1  # number of interior points
    diag = np.full(m, 1.0 + r)
    off = np.full(m - 1, -0.5 * r)

    for n in range(1, n_steps + 1):
        interior = u[1:-1]
        rhs = interior + 0.5 * r * _apply_A(interior)
        if linear_solver == "cg":
            # the previous time step is used as the starting guess
            new, iters = conjugate_gradient(apply_M, rhs, x0=interior, rtol=rtol)
            sol.cg_iterations.append(iters)
        elif linear_solver == "thomas":
            new = solve_tridiagonal(off, diag, off, rhs)
        else:
            raise ValueError(f"unknown linear_solver: {linear_solver}")
        u[1:-1] = new
        if save_every and n % save_every == 0:
            sol.times.append(n * dt)
            sol.history.append(u.copy())
    return sol


def crank_nicolson_condition_number(r, nx):
    """Condition number of (I - r/2 A), from its known eigenvalues.

    The eigenvalues are 1 + 2 r sin^2(k pi / (2 nx)) for k = 1 .. nx-1, so the
    condition number is roughly 1 + 2r: bigger time steps make the system
    harder for conjugate gradient.
    """
    k = np.arange(1, nx)
    eig = 1.0 + 2.0 * r * np.sin(k * np.pi / (2 * nx)) ** 2
    return eig.max() / eig.min()


def max_error(sol: Solution, problem: HeatProblem):
    """Largest pointwise difference from the exact solution at the final time."""
    return np.max(np.abs(sol.u - problem.exact(sol.x, problem.t_final)))
