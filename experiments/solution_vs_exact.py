"""Figure 1: Crank-Nicolson snapshots against the exact solution, and the
pointwise error of both methods at the final time."""

import numpy as np
from common import CRANK_NICOLSON, EXPLICIT, plt, save

from heat1d import HeatProblem, max_error, solve_crank_nicolson, solve_explicit

problem = HeatProblem()
nx = 50
dx = problem.length / nx

cn = solve_crank_nicolson(problem, nx, dt=0.05, save_every=5)
ex = solve_explicit(problem, nx, dt=0.4 * dx**2 / problem.alpha)

print(f"Explicit:        dt = {ex.dt:.4g}, r = {ex.r:.2f}, steps = {ex.n_steps}, "
      f"max error = {max_error(ex, problem):.2e}")
print(f"Crank-Nicolson:  dt = {cn.dt:.4g}, r = {cn.r:.2f}, steps = {cn.n_steps}, "
      f"max error = {max_error(cn, problem):.2e}")

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

x_fine = np.linspace(0, problem.length, 400)
greys = ["#c9c8c1", "#9a9993", "#6b6b66", "#2b2b2a"]  # light = early, dark = late
shown = [0.0, 0.25, 0.5, 1.0]
for shade, t_show in zip(greys, shown):
    i = int(np.argmin(np.abs(np.array(cn.times) - t_show)))
    t, u = cn.times[i], cn.history[i]
    left.plot(x_fine, problem.exact(x_fine, t), color=shade, lw=1.8, label=f"exact, t = {t:g}")
    left.plot(cn.x[::2], u[::2], "o", color=CRANK_NICOLSON, ms=4,
              label="Crank–Nicolson" if t_show == 1.0 else None)
left.set_title("Temperature along the rod over time")
left.set_xlabel("position x")
left.set_ylabel("temperature u")
left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8.5)

for sol, colour, name in [(ex, EXPLICIT, f"explicit (r = {ex.r:.2f}, {ex.n_steps} steps)"),
                          (cn, CRANK_NICOLSON, f"Crank–Nicolson (r = {cn.r:.1f}, {cn.n_steps} steps)")]:
    err = np.abs(sol.u - problem.exact(sol.x, problem.t_final))
    right.semilogy(sol.x[1:-1], err[1:-1], color=colour, label=name)
right.set_title(f"Error at t = {problem.t_final:g} (nx = {nx})")
right.set_xlabel("position x")
right.set_ylabel("|numerical − exact|")
right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8.5)

save(fig, "solution_vs_exact.png")
