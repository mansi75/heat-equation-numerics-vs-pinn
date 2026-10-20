"""Figure 4: how the time step affects the cost of each Crank-Nicolson step.

Each step solves (I - r/2 A) u_new = rhs with conjugate gradient. The
condition number of that matrix is about 1 + 2r, and CG theory says the
number of iterations grows roughly like the square root of the condition
number. Larger time steps mean fewer steps overall, but each one costs more.
"""

from dataclasses import replace

import numpy as np
from common import CRANK_NICOLSON, REFERENCE, RESULTS, plt, save

from heat1d import HeatProblem, crank_nicolson_condition_number, solve_crank_nicolson

nx = 200
n_steps = 20
base = HeatProblem()
dx = base.length / nx

rs = np.logspace(-1, 3, 13)
mean_iters, kappas = [], []
for r in rs:
    dt = r * dx**2 / base.alpha
    problem = replace(base, t_final=n_steps * dt)  # same number of steps for every r
    sol = solve_crank_nicolson(problem, nx, dt=dt)
    mean_iters.append(np.mean(sol.cg_iterations))
    kappas.append(crank_nicolson_condition_number(sol.r, nx))

lines = ["| r | condition number | sqrt(condition number) | mean CG iterations per step |", "|---:|---:|---:|---:|"]
lines += [f"| {r:.3g} | {k:.1f} | {np.sqrt(k):.1f} | {m:.1f} |" for r, k, m in zip(rs, kappas, mean_iters)]
table = "\n".join(lines)
print(table)
(RESULTS / "cg_conditioning.md").write_text(table + "\n")

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.loglog(rs, mean_iters, "s-", color=CRANK_NICOLSON, label="mean CG iterations per step")
ax.loglog(rs, np.sqrt(kappas), "--", color=REFERENCE, lw=1.3, label="√(condition number)")
ax.set_title(f"Conjugate gradient cost vs time step (nx = {nx})")
ax.set_xlabel("r = α dt / dx²")
ax.set_ylabel("count")
ax.legend(loc="upper left")
save(fig, "cg_conditioning.png")
