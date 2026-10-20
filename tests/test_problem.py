"""Checks on the exact solution that every solver is measured against."""

import numpy as np

from heat1d import HeatProblem


def test_exact_solution_matches_boundary_and_initial_conditions():
    problem = HeatProblem()
    x = np.linspace(0.0, problem.length, 101)
    expected = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    assert np.allclose(problem.initial_condition(x), expected)
    for t in (0.0, 0.3, 1.0):
        ends = problem.exact(np.array([0.0, problem.length]), t)
        assert np.allclose(ends, 0.0, atol=1e-14)


def test_exact_solution_satisfies_heat_equation():
    """u_t - alpha * u_xx should vanish, checked with small finite differences."""
    problem = HeatProblem()
    x = np.linspace(0.1, 0.9, 9)
    t, h, k = 0.4, 1e-4, 1e-5
    u_t = (problem.exact(x, t + k) - problem.exact(x, t - k)) / (2 * k)
    u_xx = (problem.exact(x + h, t) - 2 * problem.exact(x, t) + problem.exact(x - h, t)) / h**2
    assert np.max(np.abs(u_t - problem.alpha * u_xx)) < 1e-5
