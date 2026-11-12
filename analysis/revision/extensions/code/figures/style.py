"""Shared figure style for the revised IJIM manuscript (static print figures)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np
from pathlib import Path

import os
# ROOT is the revision folder; set RIS_REVISION_ROOT to run elsewhere
ROOT = Path(os.environ.get("RIS_REVISION_ROOT", Path(__file__).resolve().parents[3]))
BASE = ROOT / "extensions"
TAB = BASE / "tables"
NATURAL = ROOT / "inputs/plotting/regional_means_natural.csv"
OUT = Path(os.environ.get("RIS_FIGURE_OUTPUT", Path(__file__).resolve().parent / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)

# ink and chrome (reference palette, light mode)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#898781"
TRACK = "#ecebe6"
NEUTRAL = "#f0efec"

# categorical slots 1-3 (validated all-pairs): region identity, fixed across figures
REG = {"NA": "#2a78d6", "EUW": "#16a070", "KR": "#eb6834"}
REG_MARK = {"NA": "D", "EUW": "s", "KR": "o"}
REG_ORDER = ["NA", "EUW", "KR"]

# samples: encoded by marker shape and fill in ink (not hue)
SAMPLES = ["discovery", "holdout", "strict"]
SAMPLE_LABEL = {"discovery": "Discovery", "holdout": "Player holdout", "strict": "Strict holdout"}
SAMPLE_STYLE = {
    "discovery": dict(marker="o", mfc=INK, mec=INK),
    "holdout": dict(marker="s", mfc=MUTED, mec=MUTED),
    "strict": dict(marker="D", mfc="white", mec=INK),
}

# ordinal ramp (blue) for delay groups: near = darkest
DELAY = {"near": "#104281", "middle": "#2a78d6", "far": "#86b6ef"}
DELAY_LABEL = {"near": "Objective ≤60 s after cutoff", "middle": "61–180 s", "far": ">180 s"}

ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_LABEL = {"TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid", "BOTTOM": "Bottom", "UTILITY": "Support"}
TIER_ORDER = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
TIER_LABEL = {"IRON": "Iron", "BRONZE": "Bronze", "SILVER": "Silver", "GOLD": "Gold",
              "PLATINUM": "Plat.", "EMERALD": "Emer.", "DIAMOND": "Diam."}

MM = 1 / 25.4
FULL_W = 190 * MM


def setup():
    family = "Arial"  # metric-compatible fallback on Windows/macOS
    for style in ("Regular", "Bold", "Italic"):
        fp = Path(f"/usr/share/fonts/truetype/liberation/LiberationSans-{style}.ttf")
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            family = "Liberation Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "errorbar.capsize": 0,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def hgrid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)


def panel_title(ax, letter, text, x=-0.02, y=1.02, ha="left", fig_coords=None):
    """Draw a panel letter at the specified axes position."""
    t = ax.text(x, y, f"{letter}", transform=ax.transAxes, fontsize=9, fontweight="bold",
                color=INK, ha=ha, va="bottom")
    import matplotlib.transforms as mtrans
    off = mtrans.offset_copy(ax.transAxes, fig=ax.figure, x=10, y=0, units="points")
    # Panel descriptions are provided in the figure captions.
    return t


def read(rel, **kw):
    return pd.read_csv(BASE / rel, keep_default_na=False, na_values=[""], **kw)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.04,
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
