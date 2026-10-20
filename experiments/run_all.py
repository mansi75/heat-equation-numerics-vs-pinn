"""Run every experiment and regenerate all figures and result tables."""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for name in ["solution_vs_exact", "convergence_study", "stability_demo", "cg_conditioning"]:
    print(f"\n=== {name} ===")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
