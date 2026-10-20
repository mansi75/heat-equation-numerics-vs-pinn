"""The 1D heat equation test problem and its exact solution.

    u_t = alpha * u_xx            for 0 < x < L, 0 < t <= T
    u(0, t) = u(L, t) = 0         (both ends of the rod held at zero)
    u(x, 0) = sin(pi x / L) + 0.5 sin(3 pi x / L)

The starting temperature is a sum of sine waves. Each sine wave keeps its
shape and simply decays at its own rate, exp(-alpha * (k pi / L)^2 * t),
so the exact solution can be written down. That is what every numerical
method in this repo is measured against.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeatProblem:
    alpha: float = 0.02  # diffusion coefficient: how quickly heat spreads
    length: float = 1.0  # length of the rod
    t_final: float = 1.0  # time we solve up to
    # (wave number k, amplitude) for each sine wave in the starting temperature
    modes: tuple = ((1, 1.0), (3, 0.5))

    def exact(self, x, t):
        """Exact temperature u(x, t)."""
        x = np.asarray(x, dtype=float)
        u = np.zeros_like(x)
        for k, amplitude in self.modes:
            wave = k * np.pi / self.length
            u += amplitude * np.sin(wave * x) * np.exp(-self.alpha * wave**2 * t)
        return u

    def initial_condition(self, x):
        return self.exact(x, 0.0)

    def grid(self, nx):
        """Split the rod into nx equal intervals (nx + 1 points, ends included)."""
        x = np.linspace(0.0, self.length, nx + 1)
        return x, x[1] - x[0]
