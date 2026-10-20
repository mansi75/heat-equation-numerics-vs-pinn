"""Checks on the explicit and Crank-Nicolson solvers, including the order of
accuracy. A wrong slope is the most reliable sign of a bug."""

import numpy as np
import pytest

from heat1d import HeatProblem, max_error, solve_crank_nicolson, solve_explicit

problem = HeatProblem()


def observed_order(h, err):
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


def test_explicit_rejects_unstable_time_step():
    nx = 50
    dx = problem.length / nx
    with pytest.raises(ValueError, match="unstable"):
        solve_explicit(problem, nx, dt=0.6 * dx**2 / problem.alpha)


def test_crank_nicolson_cg_matches_thomas():
    cg = solve_crank_nicolson(problem, 80, dt=0.05, linear_solver="cg")
    thomas = solve_crank_nicolson(problem, 80, dt=0.05, linear_solver="thomas")
    assert np.max(np.abs(cg.u - thomas.u)) < 1e-10
    assert len(cg.cg_iterations) == cg.n_steps


def test_both_methods_are_second_order_in_space():
    grids = np.array([20, 40, 80, 160])
    dxs = problem.length / grids
    explicit = np.array([
        max_error(solve_explicit(problem, nx, dt=0.4 * dx**2 / problem.alpha), problem)
        for nx, dx in zip(grids, dxs)
    ])
    crank_nicolson = np.array([max_error(solve_crank_nicolson(problem, nx, dt=dx), problem)
                               for nx, dx in zip(grids, dxs)])
    assert np.all(np.abs(observed_order(dxs, explicit) - 2.0) < 0.1)
    assert np.all(np.abs(observed_order(dxs, crank_nicolson) - 2.0) < 0.1)


def test_crank_nicolson_is_second_order_in_time():
    nx = 100
    reference = solve_crank_nicolson(problem, nx, dt=1e-3, linear_solver="thomas")
    dts = np.array([0.2, 0.1, 0.05, 0.025])
    errs = np.array([np.max(np.abs(solve_crank_nicolson(problem, nx, dt=dt).u - reference.u))
                     for dt in dts])
    assert np.all(np.abs(observed_order(dts, errs) - 2.0) < 0.1)
