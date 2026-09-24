"""Signed relationships, regional structure and the resolution of grouping evidence."""

from pathlib import Path
import os
import json

OUT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "qa/mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd

DATA = OUT / "source_data/main/relationships"
REGIONS = ["NA", "EUW", "KR"]
COLORS = {"NA": "#4C4A83", "EUW": "#8C939A", "KR": "#F04432"}
MARKERS = {"NA": "D", "EUW": "s", "KR": "o"}
CMAP = LinearSegmentedColormap.from_list(
    "signed_burgundy", ["#454475", "#BABAD1", "#FBFAF9", "#CE9193", "#800C17"]
)
NORM = Normalize(-1, 1)
DICTIONARY = pd.read_csv(DATA / "metric_dictionary.csv", keep_default_na=False)
METRICS = DICTIONARY.metric.tolist()
SUMMARY = pd.read_csv(DATA / "relationship_summaries.csv", keep_default_na=False)
TESTS = pd.read_csv(DATA / "exact_grouping_diagnostics.csv", keep_default_na=False)
NULL = pd.read_csv(
    DATA / "primary_exact_label_distributions.csv", keep_default_na=False
)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.titlesize": 7,
        "axes.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#292929",
        "text.color": "#252525",
        "xtick.color": "#454545",
        "ytick.color": "#454545",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def fig(w, h):
    f = plt.figure(figsize=(w / 25.4, h / 25.4))
    f._mm = (w, h)
    return f


def at(f, x, y, text, **kw):
    w, h = f._mm
    return f.text(x / w, 1 - y / h, text, va="top", **kw)


def axes(f, x, y, w, h):
    W, H = f._mm
    a = f.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    a.tick_params(length=2, width=0.5, pad=2.4)
    return a


def title(f, l, x, y, text):
    f.text(
        x / 180, 1 - (y + 2) / f._mm[1], l, va="baseline", fontsize=8, fontweight="bold"
    )
    at(f, x + 6, y + 0.2, text, fontsize=6.5)


def read_matrix(key):
    return (
        pd.read_csv(DATA / (key + ".csv"), index_col=0).loc[METRICS, METRICS].to_numpy()
    )


# Fixed group gaps encode the dictionary, not clustering optimized on these data.
POS = np.arange(18, dtype=float)
POS[3:] += 0.9
POS[12:] += 0.9
POS[17:] += 0.9
GROUPS = [
    (0, 2, "Maintenance"),
    (3, 11, "Team signalling"),
    (12, 16, "Performance references"),
    (17, 17, "Diagnostic"),
]
extent = POS[-1] + 0.55


def matrix(f, x, y, w, h, values, labels=True, ticks=True, size=68):
    a = axes(f, x, y, w, h)
    a.set_xlim(-0.6, extent)
    a.set_ylim(extent, -0.6)
    for i in range(18):
        for j in range(i):
            a.add_patch(
                Rectangle(
                    (POS[j] - 0.48, POS[i] - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor="#D7D7D7",
                    lw=0.32,
                    zorder=0,
                )
            )
            v = float(values[i, j])
            a.scatter(
                POS[j],
                POS[i],
                s=size * abs(v),
                facecolor=CMAP(NORM(v)),
                edgecolor="#585665" if v < 0 else "none",
                lw=0.35,
                zorder=2,
            )
    a.set_yticks(
        POS,
        (
            [f"{i+1:>2}  {label}" for i, label in enumerate(DICTIONARY.label)]
            if labels
            else [str(i + 1) for i in range(18)]
        ),
    )
    a.set_xticks(POS, [str(i + 1) for i in range(18)] if ticks else [])
    a.tick_params(axis="both", length=0, labelsize=6.2 if labels else 5.5, pad=3)
    for sp in a.spines.values():
        sp.set_visible(False)
    return a


def colorbar(f, x, y, w, h, label=True):
    a = axes(f, x, y, w, h)
    cb = f.colorbar(
        plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
        cax=a,
        orientation="horizontal",
        ticks=[-1, -0.5, 0, 0.5, 1],
    )
    cb.solids.set_rasterized(False)
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(length=1.6, width=0.4, labelsize=6, pad=2)
    if label:
        cb.set_label("Pearson r", fontsize=6.2, labelpad=3)
    return cb


AUDITS = {}


def qa(f, name):
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    outer = f.bbox
    entries = []
    outside = []
    for text in f.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.width < 0.1 or box.height < 0.1:
            continue
        entries.append((text.get_text(), box))
        if (
            box.x0 < outer.x0 - 0.5
            or box.y0 < outer.y0 - 0.5
            or box.x1 > outer.x1 + 0.5
            or box.y1 > outer.y1 + 0.5
        ):
            outside.append(text.get_text())
    overlaps = []
    for i, (s, a) in enumerate(entries):
        for t, b in entries[i + 1 :]:
            dx = min(a.x1, b.x1) - max(a.x0, b.x0)
            dy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if dx > 0.7 and dy > 0.7:
                overlaps.append([s, t])
    data_intersections = []
    for ax in f.axes:
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            coords = collection.get_offset_transform().transform(
                collection.get_offsets()
            )
            sizes = collection.get_sizes()
            for k, (cx, cy) in enumerate(coords):
                radius = (sizes[min(k, len(sizes) - 1)] ** 0.5) / 2 * f.dpi / 72
                for text, box in entries:
                    dx = min(cx + radius, box.x1) - max(cx - radius, box.x0)
                    dy = min(cy + radius, box.y1) - max(cy - radius, box.y0)
                    if dx > 0.7 and dy > 0.7:
                        data_intersections.append(text)
    AUDITS[name] = {
        "size_mm": f._mm,
        "outside_text": outside,
        "text_overlap_candidates": overlaps,
        "text_elements": len(entries),
        "text_data_intersections": data_intersections,
        "font_family": "Arial",
        "matrix_range": [-1, 1],
    }
    for ext in ["png", "pdf", "svg", "tiff"]:
        kw = {"dpi": 600 if ext == "tiff" else 300}
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        f.savefig(OUT / "figures" / f"{name}.{ext}", **kw)
    plt.close(f)


def main():
    f = fig(180, 126)
    title(f, "a", 2, 2, "Indicator relationships")
    matrix(f, 46, 12, 76, 96, read_matrix("player_region_centered_Pearson"), size=72)
    # Legend uses the otherwise empty upper triangle; it does not obscure any cell.
    at(f, 72, 11, "Indicators", fontsize=6.5, fontweight="bold")
    for yy, txt in zip(
        [16, 21, 26, 31],
        [
            "1–3   Maintenance",
            "4–12   Team signalling",
            "13–17   Performance references",
            "18   Vision-score diagnostic",
        ],
    ):
        at(f, 72, yy, txt, fontsize=6)
    colorbar(f, 78, 41, 34, 2.6)
    at(f, 85, 54, "Circle area = |r|", fontsize=6)

    title(f, "b", 127, 2, "Within-region structure")
    handles = [
        Line2D([], [], color=COLORS[r], marker=MARKERS[r], ms=3, lw=0, label=r)
        for r in REGIONS
    ]
    f.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(132 / 180, 1 - 9 / 126),
        ncol=3,
        frameon=False,
        fontsize=6,
        handletextpad=0.3,
        handlelength=0.7,
        columnspacing=0.9,
        borderaxespad=0,
    )
    a = axes(f, 146, 20, 30, 30)
    a.set_xlim(0, 0.5)
    a.set_ylim(2.65, -0.65)
    for y, (key, label) in enumerate(
        [
            ("within_M", "Within M\n3 pairs"),
            ("within_S", "Within S\n36 pairs"),
            ("between_M_S", "M–S\n27 pairs"),
        ]
    ):
        at(f, 128, 22 + y * 9.1, label, fontsize=6)
        a.axhline(y, color="#E2E2E2", lw=0.4, zorder=0)
        for offset, r in zip([-0.20, 0, 0.20], REGIONS):
            value = float(
                SUMMARY.loc[
                    SUMMARY.matrix.eq(f"player_{r}_Pearson")
                    & SUMMARY.relationship.eq(key),
                    "mean_abs_r",
                ].item()
            )
            a.scatter(
                value,
                y + offset,
                s=13,
                marker=MARKERS[r],
                facecolor=COLORS[r],
                edgecolor="white",
                lw=0.3,
                zorder=3,
            )
    a.set_yticks([])
    a.set_xticks([0, 0.25, 0.5], ["0", "0.25", "0.50"])
    a.set_xlabel("Mean |r|", labelpad=3)
    a.spines["left"].set_visible(False)

    title(f, "c", 127, 68, "Grouping resolution")
    handles = [
        Line2D([], [], marker="D", color="#262626", ms=3, lw=0, label="Observed"),
        Line2D([], [], color="#B8B8B8", lw=3, label="All labelings"),
    ]
    f.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(129 / 180, 1 - 74 / 126),
        ncol=2,
        frameon=False,
        fontsize=5.8,
        columnspacing=0.8,
        handletextpad=0.4,
        handlelength=0.8,
        borderaxespad=0,
    )
    a = axes(f, 129, 81, 47, 33)
    a.set_xlim(-0.12, 0.24)
    a.set_ylim(1.5, -0.5)
    a.set_yticks([])
    a.set_xticks([-0.1, 0, 0.1, 0.2], ["−0.1", "0", "0.1", "0.2"])
    a.set_xlabel("Within − between, mean |r|", labelpad=3, fontsize=6)
    a.spines["left"].set_visible(False)
    a.axvline(0, color="#A6A6A6", lw=0.55, ls=(0, (2, 2)), zorder=0)
    bins = np.linspace(-0.12, 0.24, 61)
    for y, test in enumerate(
        ["maintenance_vs_signalling", "signal_functional_subgroups"]
    ):
        values = NULL.loc[NULL.test.eq(test), "statistic"].to_numpy(float)
        counts, edges = np.histogram(values, bins=bins)
        assert counts.sum() == len(values)
        heights = counts / max(counts) * 0.23
        xx = (edges[:-1] + edges[1:]) / 2
        a.fill_between(
            xx,
            y - heights,
            y + heights,
            step="mid",
            facecolor="#D0D0D0",
            edgecolor="#929292",
            lw=0.4,
            zorder=1,
        )
        row = TESTS.loc[
            TESTS.matrix.eq("player_region_centered_Pearson") & TESTS.test.eq(test)
        ].iloc[0]
        a.scatter(
            float(row.statistic),
            y,
            s=18,
            marker="D",
            facecolor="#262626",
            edgecolor="white",
            lw=0.35,
            zorder=3,
        )
        label = "Maintenance / signalling" if y == 0 else "Signal subgroups"
        a.text(-0.12, y - 0.43, label, fontsize=6.1, ha="left", va="center")
        a.text(
            0.237,
            y + 0.38,
            "Holm P = " + ("0.0091" if y == 0 else "0.550"),
            fontsize=6,
            ha="right",
            va="center",
        )
    qa(f, "Fig1_variable_relationships")


def supplementary():
    # Three complete within-region matrices reveal individual correlations hidden by block means.
    f = fig(180, 142)

    for letter, r, x in zip(["a", "b", "c"], REGIONS, [8, 68, 128]):
        at(f, x, 12, letter, fontsize=8, fontweight="bold")
        at(f, x + 6, 12.2, r, fontsize=7, fontweight="bold", color=COLORS[r])
        matrix(
            f,
            x + 1,
            23,
            47,
            66,
            read_matrix(f"player_{r}_Pearson"),
            labels=False,
            size=32,
        )
    colorbar(f, 67, 99, 44, 2.5)
    labels = DICTIONARY.label.tolist()
    for col, start in enumerate([0, 6, 12]):
        for row in range(6):
            idx = start + row
            at(
                f,
                5 + 60 * col,
                115 + row * 4.1,
                f"{idx+1:>2}  {labels[idx]}",
                fontsize=6,
            )
    at(f, 6, 97, "Area = |r|; outlines mark r < 0", fontsize=6)
    qa(f, "Supplementary_FigS1_regional_relationships")


if __name__ == "__main__":
    main()
    supplementary()
    (OUT / "qa/layout_audit.json").write_text(
        json.dumps(AUDITS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(AUDITS, ensure_ascii=False, indent=2))
