"""Figure 3: the explicit method's stability limit, r = alpha * dt / dx^2 <= 1/2.

Each grid-scale wave is multiplied by G = 1 - 4 r sin^2(k dx / 2) every step.
For r <= 1/2, |G| <= 1 for every wave, so nothing grows. Just above 1/2, the
most jagged wave (alternating up/down from point to point) has |G| > 1. It
starts at round-off level (~1e-16) and grows exponentially until it swamps
the solution.
"""

import numpy as np
from common import EXACT, EXPLICIT, THIRD, plt, save

from heat1d import HeatProblem, solve_explicit

problem = HeatProblem()
nx = 100
dx = problem.length / nx

runs = {}
for r in (0.45, 0.55):
    sol = solve_explicit(problem, nx, dt=r * dx**2 / problem.alpha, save_every=1, check_stability=False)
    err = [np.max(np.abs(u - problem.exact(sol.x, t))) for t, u in zip(sol.times, sol.history)]
    runs[r] = (sol, np.array(err))
    G = abs(1 - 4 * sol.r)  # growth factor of the most jagged wave
    print(f"r = {sol.r:.3f}: growth factor of the most jagged wave |G| = {G:.3f}, "
          f"final max error = {err[-1]:.2e}")

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

colours = {0.45: EXPLICIT, 0.55: THIRD}
for r, (sol, err) in runs.items():
    left.semilogy(sol.times[1:], err[1:], color=colours[r],  # skip t = 0, where the error is exactly 0
                  label=f"r = {r} ({'stable' if r <= 0.5 else 'unstable'})")
left.set_title("Error over time, explicit method")
left.set_xlabel("time t")
left.set_ylabel("max error")
left.legend(loc="upper left")

# show the unstable run at the moment the zigzag becomes visible
sol, err = runs[0.55]
i = int(np.argmax(err > 0.05))
t = sol.times[i]
stable_sol, _ = runs[0.45]
j = int(np.argmin(np.abs(np.array(stable_sol.times) - t)))
x_fine = np.linspace(0, problem.length, 400)
right.plot(x_fine, problem.exact(x_fine, t), color=EXACT, lw=1.5, label="exact")
right.plot(stable_sol.x, stable_sol.history[j], color=EXPLICIT, lw=1.5, ls="--", label="r = 0.45")
right.plot(sol.x, sol.history[i], color=THIRD, lw=1.2, label="r = 0.55")
right.set_title(f"Solution at t = {t:.2f}")
right.set_xlabel("position x")
right.set_ylabel("temperature u")
right.legend(loc="lower center")
print(f"snapshot at t = {t:.3f}")

save(fig, "stability.png")
