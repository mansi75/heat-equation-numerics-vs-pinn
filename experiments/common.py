"""Shared plot style and output folders for the experiment scripts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# one colour per method, used the same way in every figure
EXPLICIT = "#eb6834"
CRANK_NICOLSON = "#2a78d6"
THIRD = "#1baf7a"
EXACT = "#2b2b2a"
REFERENCE = "#8a8a85"
INK = "#2b2b2a"
MUTED = "#6b6b66"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK,
        "axes.edgecolor": "#c9c8c1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5df",
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
)


def save(fig, name):
    path = FIGURES / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")
