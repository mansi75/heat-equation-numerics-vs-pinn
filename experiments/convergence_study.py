"""Figure 2: how the error shrinks as the grid and time step are refined,
and how much computing time each method needs for a given accuracy.

(a) Space: refine the grid. Explicit uses r = 0.4 (so dt ~ dx^2, as stability
    requires); Crank-Nicolson uses dt = dx. Both should show error ~ dx^2.
(b) Time: fix a fine grid and refine dt for Crank-Nicolson, measuring against
    a reference run with a tiny dt on the same grid. This isolates the time
    error, which should shrink like dt^2.
(c) Cost: error against wall-clock time (mean of repeated runs).
"""

import time

import numpy as np
from common import CRANK_NICOLSON, EXPLICIT, REFERENCE, RESULTS, plt, save

from heat1d import HeatProblem, max_error, solve_crank_nicolson, solve_explicit

problem = HeatProblem()
REPEATS = 7


def timed(fn):
    """Run fn REPEATS times; return the last result and mean/std of runtimes."""
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return out, np.mean(times), np.std(times, ddof=1)


def observed_orders(h, err):
    h, err = np.asarray(h), np.asarray(err)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


# (a) refinement in space ----------------------------------------------------
grids = [10, 20, 40, 80, 160, 320, 640, 1280]
rows = []
for nx in grids:
    dx = problem.length / nx
    ex, ex_t, ex_sd = timed(lambda: solve_explicit(problem, nx, dt=0.4 * dx**2 / problem.alpha))
    cn, cn_t, cn_sd = timed(lambda: solve_crank_nicolson(problem, nx, dt=dx))
    rows.append(dict(nx=nx, dx=dx,
                     ex_steps=ex.n_steps, ex_err=max_error(ex, problem), ex_t=ex_t, ex_sd=ex_sd,
                     cn_steps=cn.n_steps, cn_err=max_error(cn, problem), cn_t=cn_t, cn_sd=cn_sd,
                     cn_cg=np.mean(cn.cg_iterations)))

dxs = [r["dx"] for r in rows]
ex_orders = observed_orders(dxs, [r["ex_err"] for r in rows])
cn_orders = observed_orders(dxs, [r["cn_err"] for r in rows])

# (b) refinement in time (Crank-Nicolson only) -----------------------------
nx_fine = 200
reference = solve_crank_nicolson(problem, nx_fine, dt=1e-4, linear_solver="thomas")
dts = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
time_err = [np.max(np.abs(solve_crank_nicolson(problem, nx_fine, dt=dt).u - reference.u)) for dt in dts]
time_orders = observed_orders(dts, time_err)

# print and save a results table ---------------------------------------------
lines = [
    "| nx | explicit steps | explicit error | explicit time (ms) | CN steps | CN error | CN time (ms) | CN mean CG iters |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['nx']} | {r['ex_steps']} | {r['ex_err']:.2e} | {1e3*r['ex_t']:.1f} ± {1e3*r['ex_sd']:.1f} "
        f"| {r['cn_steps']} | {r['cn_err']:.2e} | {1e3*r['cn_t']:.1f} ± {1e3*r['cn_sd']:.1f} | {r['cn_cg']:.1f} |"
    )
lines += [
    "",
    "Observed order in space (explicit): " + ", ".join(f"{o:.2f}" for o in ex_orders),
    "Observed order in space (Crank-Nicolson): " + ", ".join(f"{o:.2f}" for o in cn_orders),
    "",
    f"Time refinement, Crank-Nicolson, nx = {nx_fine}:",
    "| dt | error vs reference |",
    "|---:|---:|",
]
lines += [f"| {dt:g} | {e:.2e} |" for dt, e in zip(dts, time_err)]
lines += ["", "Observed order in time (Crank-Nicolson): " + ", ".join(f"{o:.2f}" for o in time_orders)]
table = "\n".join(lines)
print(table)
(RESULTS / "convergence.md").write_text(table + "\n")

# figure -----------------------------------------------------------------------
fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9))

a.loglog(dxs, [r["ex_err"] for r in rows], "o-", color=EXPLICIT, label="explicit (r = 0.4)")
a.loglog(dxs, [r["cn_err"] for r in rows], "s-", color=CRANK_NICOLSON, label="Crank–Nicolson (dt = dx)")
guide = rows[0]["cn_err"] * (np.array(dxs) / dxs[0]) ** 2 * 0.5
a.loglog(dxs, guide, "--", color=REFERENCE, lw=1.3, label="slope 2 (error ∝ dx²)")
a.set_title("(a) Error vs grid spacing")
a.set_xlabel("grid spacing dx")
a.set_ylabel("max error at t = 1")
a.legend(fontsize=8.5, loc="upper left")

b.loglog(dts, time_err, "s-", color=CRANK_NICOLSON, label=f"Crank–Nicolson (nx = {nx_fine})")
guide = time_err[0] * (np.array(dts) / dts[0]) ** 2 * 0.5
b.loglog(dts, guide, "--", color=REFERENCE, lw=1.3, label="slope 2 (error ∝ dt²)")
b.set_title("(b) Time error vs time step")
b.set_xlabel("time step dt")
b.set_ylabel("max difference from reference")
b.legend(fontsize=8.5, loc="upper left")

# timings under ~1 ms are mostly Python overhead and timer noise, so start at nx = 40
timed_rows = [r for r in rows if r["nx"] >= 40]
c.loglog([1e3 * r["ex_t"] for r in timed_rows], [r["ex_err"] for r in timed_rows], "o-", color=EXPLICIT, label="explicit")
c.loglog([1e3 * r["cn_t"] for r in timed_rows], [r["cn_err"] for r in timed_rows], "s-", color=CRANK_NICOLSON, label="Crank–Nicolson")
first, last = timed_rows[0], timed_rows[-1]
note = dict(textcoords="offset points", fontsize=7.5, color="#6b6b66")
c.annotate(f"nx = {first['nx']}", (1e3 * first["ex_t"], first["ex_err"]), xytext=(7, 2), **note)
c.annotate(f"nx = {first['nx']}", (1e3 * first["cn_t"], first["cn_err"]), xytext=(7, 2), **note)
c.annotate(f"nx = {last['nx']}", (1e3 * last["ex_t"], last["ex_err"]), xytext=(0, -14), ha="center", **note)
c.annotate(f"nx = {last['nx']}", (1e3 * last["cn_t"], last["cn_err"]), xytext=(-8, -4), ha="right", **note)
c.set_title("(c) Accuracy vs computing time")
c.set_xlabel(f"wall-clock time (ms, mean of {REPEATS} runs)")
c.set_ylabel("max error at t = 1")
c.legend(fontsize=8.5, loc="upper right")

fig.tight_layout(w_pad=2.5)
save(fig, "convergence.png")
