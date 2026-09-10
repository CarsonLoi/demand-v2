from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

NAVY, TEAL, GREY, GOLD, RED, GREEN = (
    "#0F2942", "#2E8BA8", "#5F7183", "#E8B547", "#C74B4B", "#2FA87C")

# a stable colour per clean year for the year-over-year line charts
YEAR_COLOURS = {
    2015: "#6B8CAE", 2016: "#4E7A9E", 2017: "#2E8BA8", 2018: "#1F6F86",
    2019: "#0F2942", 2023: "#E8B547", 2024: "#D98C3A", 2025: "#C74B4B", 2026: "#2FA87C",
}


def style_ax(ax) -> None:
    ax.grid(alpha=0.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=9)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")
