"""Compose publication figures from frozen source tables; no statistical fitting."""

from pathlib import Path
import os, json, argparse

OUT = Path(__file__).resolve().parents[2]
FIGURES = OUT / "figures/main"
FIGURES.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(OUT / "validation/mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

REG = ["NA", "EUW", "KR"]
COL = {"NA": "#4C4A83", "EUW": "#8C939A", "KR": "#F04432"}
MARK = {"NA": "D", "EUW": "s", "KR": "o"}
LS = {"NA": "solid", "EUW": (0, (4, 2)), "KR": (0, (5, 1.5, 1, 1.5))}
PINGS = [
    s + "_pings_pm"
    for s in [
        "enemy_missing",
        "enemy_vision",
        "command",
        "need_vision",
        "assist_me",
        "on_my_way",
        "push",
        "all_in",
        "get_back",
    ]
]
PING_NAMES = [
    "Enemy missing",
    "Enemy vision",
    "Attention",
    "Need vision",
    "Assist me",
    "On my way",
    "Push",
    "All in",
    "Get back",
]
TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
ABBR = ["Ir", "Br", "Si", "Go", "Pl", "Em", "Di"]
M = "shared_information_maintenance"
S = "team_signalling"
COMPS = ["valid_wards_placed", "valid_ward_kills", "control_ward_purchases"]
AUDIT = {}


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": "#292929",
            "axes.labelcolor": "#292929",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read(package, name, kind="source_data"):
    v = pd.read_csv(OUT / kind / "main" / package / name, keep_default_na=False)
    if "region" in v:
        v["region"] = v.region.replace({"NA1": "NA", "EUW1": "EUW"})
    return v


def fig(h):
    f = plt.figure(figsize=(180 / 25.4, h / 25.4))
    f._mm = (180, h)
    return f


def txt(f, x, y, s, **kw):
    return f.text(x / 180, 1 - y / f._mm[1], s, va="top", **kw)


def panel(f, x, y, s):
    return f.text(
        x / 180, 1 - (y + 2) / f._mm[1], s, va="baseline", fontsize=8, fontweight="bold"
    )


def ax(f, x, y, w, h):
    a = f.add_axes([x / 180, 1 - (y + h) / f._mm[1], w / 180, h / f._mm[1]])
    a.tick_params(length=2.2, width=0.5, pad=2.5)
    return a


def legend(f, x=90, y=1):
    hs = [
        Line2D([], [], color=COL[r], ls=LS[r], marker=MARK[r], ms=2.8, lw=1, label=r)
        for r in REG
    ]
    f.legend(
        handles=hs,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(x / 180, 1 - y / f._mm[1]),
        frameon=False,
        borderaxespad=0,
        handlelength=1.8,
        columnspacing=1,
        handletextpad=0.4,
        fontsize=6,
    )


def error(a, x, y, lo, hi, r, vertical=False, size=3.4):
    kw = dict(
        fmt=MARK[r],
        color=COL[r],
        ms=size,
        mec="white",
        mew=0.25,
        capsize=1.8,
        elinewidth=0.9,
        zorder=4,
    )
    if vertical:
        a.errorbar(x, y, yerr=[[y - lo], [hi - y]], **kw)
    else:
        a.errorbar(x, y, xerr=[[x - lo], [hi - x]], **kw)


def save(f, stem):
    f.canvas.draw()
    ren = f.canvas.get_renderer()
    W, H = f.canvas.get_width_height()
    items = []
    for t in f.findobj(match=Text):
        if t.get_visible() and t.get_text().strip():
            b = t.get_window_extent(ren)
            if b.width > 0 and b.height > 0:
                items.append((t.get_text(), b))
    outside = [
        s
        for s, b in items
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5
    ]
    overlap = []
    for i, (s, a) in enumerate(items):
        for t, b in items[i + 1 :]:
            if (
                min(a.x1, b.x1) - max(a.x0, b.x0) > 1
                and min(a.y1, b.y1) - max(a.y0, b.y0) > 1
            ):
                overlap.append([s, t])
    labels = {
        t.get_text(): [t.get_position()[0] * 180, (1 - t.get_position()[1]) * f._mm[1]]
        for t in f.texts
        if t.get_text() in list("abcdefghi")
    }
    AUDIT[stem] = {
        "dimensions_mm": f._mm,
        "outside_text": outside,
        "overlap_candidates": overlap,
        "panel_labels_mm": labels,
    }
    for ext in ["png", "pdf", "svg", "tiff"]:
        kw = {"dpi": 600 if ext == "tiff" else 300}
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        f.savefig(FIGURES / f"{stem}.{ext}", **kw)
    plt.close(f)
    print(stem, "outside", len(outside), "overlap", len(overlap), flush=True)


def relationships():
    """Render the full and regional indicator relationship panels."""
    import draw_relationships as renderer

    renderer.qa = save
    renderer.main()
    renderer.supplementary()
    style()


def regional():
    means = read("regional", "regional_adjusted_means.csv")
    summ = read("regional", "player_distribution_summaries.csv")
    dens = read("regional", "all_player_densities.csv")
    signals = read("regional", "Fig2d_signal_adjusted_means.csv")
    raw = read("regional", "Fig2ef_signal_occurrence_frequency.csv")

    def gm(m, r):
        return means[means.measure.eq(m) & means.region.eq(r)].iloc[0]

    f = fig(160)
    panel(f, 2, 3, "a")
    panel(f, 103, 3, "b")
    a = ax(f, 15, 14, 78, 39)
    dd = dens[dens.measure.eq("overall")]
    height = 0.49 / dd.density.max()
    for r, y in zip(REG, [2, 1, 0]):
        v = dd[dd.region.eq(r)]
        s = summ[summ.measure.eq("overall") & summ.region.eq(r)].iloc[0]
        m = gm("overall", r)
        a.fill_between(
            v.score,
            y + 0.06,
            y + 0.06 + height * v.density,
            color=COL[r],
            alpha=0.20,
            lw=0,
        )
        a.plot(v.score, y + 0.06 + height * v.density, color=COL[r], lw=0.8)
        a.hlines(y - 0.07, s.q05, s.q95, color=COL[r], lw=0.65)
        a.vlines([s.q05, s.q95], y - 0.1, y - 0.04, color=COL[r], lw=0.6)
        a.add_patch(
            Rectangle(
                (s.q25, y - 0.14),
                s.q75 - s.q25,
                0.14,
                facecolor="white",
                edgecolor=COL[r],
                lw=0.7,
                zorder=3,
            )
        )
        a.vlines(s["median"], y - 0.14, y, color=COL[r], lw=0.9, zorder=4)
        error(a, m.estimate, y - 0.27, m.ci_low, m.ci_high, r)
    a.set_xlim(-1.2, 2)
    a.set_ylim(-0.48, 2.65)
    a.set_xticks([-1, 0, 1, 2])
    a.set_yticks([2.1, 1.1, 0.1], REG)
    a.tick_params(axis="y", length=0, pad=5)
    a.spines["left"].set_visible(False)
    a.set_xlabel("Adjusted overall score (score units)", labelpad=4)
    b = ax(f, 119, 14, 52, 39)
    for r in REG:
        rr = [gm(m, r) for m in ["maintenance", "signalling"]]
        b.plot([0, 1], [v.estimate for v in rr], color=COL[r], ls=LS[r], lw=1)
        for x, m in enumerate(rr):
            error(b, x, m.estimate, m.ci_low, m.ci_high, r, vertical=True)
        b.text(1.12, rr[-1].estimate, r, va="center", fontsize=6, color=COL[r])
    b.axhline(0, color="#C5C5CA", lw=0.5, ls=(0, (2, 3)), zorder=0)
    b.set_xlim(-0.2, 1.52)
    b.set_ylim(-0.20, 0.28)
    b.set_xticks([0, 1], ["Maintenance", "Signalling"])
    b.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    b.set_ylabel("Adjusted mean (score units)", labelpad=3)
    for x, s in zip([2, 66, 126], "cde"):
        panel(f, x, 69, s)
    c = ax(f, 29, 80, 31, 64)
    d = ax(f, 79, 80, 40, 64)
    e = ax(f, 136, 80, 41, 64)
    for a in [c, d, e]:
        a.set_ylim(8.55, -0.55)
        a.set_yticks(range(9))
        a.set_yticklabels([])
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
        for y in np.arange(0.5, 8.5):
            a.axhline(y, color="#E9E9ED", lw=0.45, zorder=0)
    c.set_yticklabels(PING_NAMES)
    c.tick_params(axis="y", pad=5, labelsize=6.2)
    c.axvline(0, color="#B7B7BD", lw=0.6, ls=(0, (3, 3)), zorder=1)
    for j, m in enumerate(PINGS):
        for r, dy in zip(REG, [-0.23, 0, 0.23]):
            v = signals[signals.region.eq(r) & signals.metric.eq(m)].iloc[0]
            error(c, v.estimate, j + dy, v.ci_low, v.ci_high, r, size=3)
            v = raw[raw.region.eq(r) & raw.metric.eq(m)].iloc[0]
            error(
                e,
                v.conditional_mean_per_10_min,
                j + dy,
                v.conditional_ci_low,
                v.conditional_ci_high,
                r,
                size=3,
            )
            k = REG.index(r)
            d.scatter(
                k - 0.22,
                j,
                s=45 * v.prevalence,
                color=COL[r],
                alpha=0.88,
                lw=0,
                zorder=3,
            )
            d.text(
                k - 0.01,
                j,
                f"{v.prevalence*100:.1f}",
                fontsize=6.2,
                ha="left",
                va="center",
            )
    c.set_xlim(-0.19, 0.29)
    c.set_xticks([-0.1, 0, 0.1, 0.2])
    c.set_xlabel("Adjusted signal mean\n(score units)", labelpad=4)
    d.set_xlim(-0.46, 2.43)
    d.set_xticks(range(3), REG)
    d.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        pad=4,
    )
    d.spines["bottom"].set_visible(False)
    d.set_xlabel("Records with signal (%)", labelpad=7)
    e.set_xlim(0, 3.8)
    e.set_xticks([0, 1, 2, 3])
    e.set_xlabel("Frequency when present\n(signals per 10 min)", labelpad=4)
    hs = [
        Line2D([], [], color=COL[r], marker=MARK[r], ls="none", ms=3, label=r)
        for r in REG
    ]
    f.legend(
        handles=hs,
        loc="upper center",
        bbox_to_anchor=(155 / 180, 1 - 72 / 160),
        ncol=3,
        frameon=False,
        borderaxespad=0,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.8,
        fontsize=6,
    )
    for xpos, p in zip([82, 97, 112], [0.1, 0.5, 0.8]):
        d.scatter(
            [xpos / 180],
            [(160 - 156) / 160],
            s=45 * p,
            color="#777780",
            transform=f.transFigure,
            clip_on=False,
            lw=0,
        )
        txt(f, xpos + 2.2, 154.8, f"{p*100:.0f}%", fontsize=5.5)
    save(f, "Fig2_regional_configuration")

    # Full empirical distributions and quantile contrasts retain their existing estimands.
    ecdf = read("regional", "Supplementary_FigS2_all_player_ECDF.csv")
    shift = read("regional", "Supplementary_FigS2_quantile_differences.csv")
    f = fig(137)
    legend(f, 140, 2)
    for i, (m, x) in enumerate(
        zip(["overall", "maintenance", "signalling"], [14, 73, 132])
    ):
        panel(f, x - 12, 11, "abc"[i])
        a = ax(f, x, 21, 42, 38)
        for r in REG:
            v = ecdf[ecdf.measure.eq(m) & ecdf.region.eq(r)]
            a.step(
                v.score,
                v.cumulative_fraction * 100,
                where="post",
                color=COL[r],
                ls=LS[r],
                lw=1,
            )
        a.set_ylim(0, 101)
        a.set_yticks([0, 25, 50, 75, 100])
        a.set_xlim(-2.3, 6)
        a.set_xticks([-2, 0, 2, 4, 6])
        a.set_xlabel(m.title() + " score (score units)", labelpad=4)
        if i == 0:
            a.set_ylabel("Cumulative players (%)", labelpad=3)
        else:
            a.set_yticklabels([])
    hs = [
        Line2D([], [], color="#34343C", ls="solid", label="Maintenance"),
        Line2D([], [], color="#858590", ls=(0, (4, 2)), label="Signalling"),
    ]
    f.legend(
        handles=hs,
        loc="upper center",
        bbox_to_anchor=(135 / 180, 1 - 69 / 137),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        fontsize=6,
    )
    for i, (pair, x) in enumerate(zip(["KR-NA", "KR-EUW", "EUW-NA"], [14, 73, 132])):
        panel(f, x - 12, 74, "def"[i])
        a = ax(f, x, 87, 42, 37)
        txt(f, x + 21, 81, pair.replace("-", " − "), ha="center", fontsize=6.5)
        a.axhline(0, color="#BBBBBF", lw=0.6)
        for m, color, ls, mark in [
            ("maintenance", "#34343C", "solid", "o"),
            ("signalling", "#858590", (0, (4, 2)), "s"),
        ]:
            v = shift[shift.measure.eq(m) & shift.contrast.eq(pair)]
            a.plot(v.percentile, v.quantile_difference, color=color, ls=ls, lw=1.1)
            p = v[v.percentile.isin([10, 50, 90])]
            a.plot(
                p.percentile,
                p.quantile_difference,
                ls="none",
                marker=mark,
                ms=2.5,
                color=color,
            )
        a.set_xlim(0, 100)
        a.set_xticks([1, 25, 50, 75, 99])
        a.set_ylim(-0.27, 0.62)
        a.set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
        a.set_xlabel("Percentile within region", labelpad=4)
        if i == 0:
            a.set_ylabel("Quantile difference (score units)", labelpad=4)
        else:
            a.set_yticklabels([])
    save(f, "Supplementary_FigS2_distributions")


def context():
    tiers = read("context", "tier_adjusted_means.csv")
    positions = read("context", "position_adjusted_means.csv")

    def tp(a, m):
        for r in REG:
            v = (
                tiers[tiers.region.eq(r) & tiers.measure.eq(m)]
                .set_index("level")
                .loc[TIERS]
            )
            a.fill_between(
                range(7), v.ci_low, v.ci_high, color=COL[r], alpha=0.13, lw=0
            )
            a.plot(
                range(7),
                v.estimate,
                color=COL[r],
                ls=LS[r],
                marker=MARK[r],
                ms=2.9,
                mec="white",
                mew=0.25,
                lw=1,
            )
        a.set_xlim(-0.2, 6.2)
        a.set_xticks(range(7), ABBR)
        a.set_xlabel("Sampling tier", labelpad=4)
        a.yaxis.set_major_locator(MaxNLocator(5))
        if m in [M, S]:
            a.set_ylim(-0.7, 0.85)
            a.set_ylabel("Adjusted score", labelpad=3)
        else:
            a.set_ylim(bottom=0)
            a.set_ylabel("Pings per player per 10 min", labelpad=3)

    f = fig(185)
    legend(f)
    for i, (m, x, name) in enumerate(
        [(M, 16, "Information maintenance"), (S, 106, "Team signalling")]
    ):
        panel(f, x - 14, 10, "ab"[i])
        txt(f, x + 32, 11, name, ha="center", fontsize=6.5)
        a = ax(f, x, 23, 65, 33)
        tp(a, m)
    for i, (m, x, name) in enumerate(
        [
            ("get_back_pings_pm", 16, "Get back"),
            ("push_pings_pm", 76, "Push"),
            ("all_in_pings_pm", 136, "All in"),
        ]
    ):
        panel(f, x - 14, 72, "cde"[i])
        txt(f, x + 19.5, 73, name, ha="center", fontsize=6.5)
        a = ax(f, x, 85, 39, 32)
        tp(a, m)
    for i, (m, x, name) in enumerate(
        [(M, 16, "Information maintenance"), (S, 106, "Team signalling")]
    ):
        panel(f, x - 14, 132, "fg"[i])
        txt(f, x + 32, 133, name, ha="center", fontsize=6.5)
        a = ax(f, x, 145, 65, 26)
        for k, r in enumerate(REG):
            v = (
                positions[positions.region.eq(r) & positions.measure.eq(m)]
                .set_index("level")
                .loc[["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]]
            )
            a.errorbar(
                np.arange(5) + (k - 1) * 0.18,
                v.estimate,
                yerr=[v.estimate - v.ci_low, v.ci_high - v.estimate],
                fmt=MARK[r],
                ms=3.5,
                mec="white",
                mew=0.3,
                color=COL[r],
                capsize=2,
                elinewidth=0.9,
            )
        a.set_xlim(-0.5, 4.5)
        a.set_xticks(range(5), ["Top", "Jungle", "Mid", "Bot", "Support"])
        a.set_ylabel("Adjusted score", labelpad=3)
        a.set_xlabel("Team position", labelpad=4)
        a.yaxis.set_major_locator(MaxNLocator(5))
    save(f, "Fig3_context_profiles")
    f = fig(178)
    legend(f)
    for i, (m, name) in enumerate(zip(PINGS, PING_NAMES)):
        x = 16 + 60 * (i % 3)
        y = 21 + 55 * (i // 3)
        panel(f, x - 14, y - 12, "abcdefghi"[i])
        txt(f, x + 19.5, y - 9, name, ha="center", fontsize=6.5)
        a = ax(f, x, y, 39, 30)
        tp(a, m)
    save(f, "Supplementary_FigS3_all_signal_tier_profiles")


def timeline():
    profiles = read("timeline", "Fig3abc_minute_profiles.csv")
    curves = read("timeline", "Fig3d_adjusted_probability_curves.csv")
    density = read("timeline", "Fig3d_observed_difference_distribution.csv")
    f = fig(148)
    legend(f)
    for i, (c, x, label) in enumerate(
        zip(
            COMPS,
            [16, 76, 136],
            ["Wards placed", "Wards cleared", "Control wards purchased"],
        )
    ):
        panel(f, x - 14, 10, "abc"[i])
        a = ax(f, x, 21, 40, 31)
        for r in REG:
            v = profiles[profiles.component.eq(c) & profiles.region.eq(r)]
            a.fill_between(
                v.time_midpoint, v.ci_low, v.ci_high, color=COL[r], alpha=0.13, lw=0
            )
            a.plot(
                v.time_midpoint,
                v.mean_rate,
                color=COL[r],
                ls=LS[r],
                marker=MARK[r],
                ms=2.6,
                mec="white",
                mew=0.25,
                lw=1,
            )
        a.set_xlim(0, 10)
        a.set_xticks([0, 2, 4, 6, 8, 10])
        a.set_ylim(bottom=0)
        a.yaxis.set_major_locator(MaxNLocator(5))
        a.set_ylabel(label + "\n(per team per min)", labelpad=3)
        a.set_xlabel("Time since start (min)", labelpad=4)
    panel(f, 2, 68, "d")
    a = ax(f, 24, 75, 148, 34)
    for r in REG:
        v = curves[curves.region.eq(r)]
        a.fill_between(
            v.maintenance_difference_sd,
            v.ci_low * 100,
            v.ci_high * 100,
            color=COL[r],
            alpha=0.13,
            lw=0,
        )
        a.plot(
            v.maintenance_difference_sd,
            v.probability * 100,
            color=COL[r],
            ls=LS[r],
            lw=1.2,
        )
        v = v[v.maintenance_difference_sd.isin([-1, 0, 1])]
        a.plot(
            v.maintenance_difference_sd,
            v.probability * 100,
            ls="none",
            marker=MARK[r],
            ms=3.2,
            color=COL[r],
            mec="white",
            mew=0.3,
        )
    a.axvline(0, color="#B4B4BB", lw=0.65, ls=(0, (3, 3)), zorder=0)
    a.set_xlim(-2, 2)
    a.set_ylim(42, 56)
    a.set_yticks([42, 46, 50, 54])
    a.set_xticks([-2, -1, 0, 1, 2])
    a.tick_params(axis="x", labelbottom=False)
    a.set_ylabel("Blue-team probability of the\nnext neutral objective (%)", labelpad=4)
    a.text(0.025, 0.97, "−1 to +1 SD", transform=a.transAxes, fontsize=6.2, va="top")
    for i, r in enumerate(REG):
        v = curves[curves.region.eq(r)].set_index("maintenance_difference_sd")
        delta = 100 * (v.loc[1, "probability"] - v.loc[-1, "probability"])
        a.text(
            0.025,
            0.82 - i * 0.12,
            f"{r}  +{delta:.2f} pp",
            transform=a.transAxes,
            color=COL[r],
            fontsize=6.2,
            va="top",
        )
    ymax = density["count"].max() / 1000 * 1.1
    for i, r in enumerate(REG):
        h = ax(f, 24, 113 + i * 7, 148, 5)
        v = density[density.region.eq(r)]
        h.fill_between(v.mid, 0, v["count"] / 1000, color=COL[r], alpha=0.15, lw=0)
        h.plot(v.mid, v["count"] / 1000, color=COL[r], ls=LS[r], lw=0.65)
        h.set_xlim(-2, 2)
        h.set_ylim(0, ymax)
        h.set_yticks([0, round(ymax // 1)])
        h.tick_params(axis="y", labelsize=4.8, pad=2, length=1)
        txt(f, 3, 113 + i * 7, r, fontsize=5.6, color=COL[r])
        h.set_xticks([-2, -1, 0, 1, 2])
        if i < 2:
            h.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            h.set_xlabel(
                "Early maintenance difference (blue − red; common SD)", labelpad=4
            )
    txt(f, 3, 137, "Matches\n(×10³)", fontsize=5.2)
    save(f, "Fig4_timeline_and_objective_control")
    contrasts = read("timeline", "minute_regional_contrasts.csv", "tables")
    f = fig(155)
    for col, (c, x) in enumerate(zip(COMPS, [17, 77, 137])):
        txt(
            f,
            x + 19.5,
            2,
            ["Placement", "Clearing", "Purchases"][col],
            ha="center",
            fontsize=6.5,
        )
        v = contrasts[contrasts.component.eq(c)]
        lim = max(abs(v.simultaneous_low).max(), abs(v.simultaneous_high).max()) * 1.12
        for row, (pair, y) in enumerate(
            zip(["KR-NA", "KR-EUW", "EUW-NA"], [15, 62, 109])
        ):
            panel(f, x - 15, y - 5, "abcdefghi"[row * 3 + col])
            a = ax(f, x, y + 3, 39, 29)
            v = contrasts[contrasts.component.eq(c) & contrasts.contrast.eq(pair)]
            t = v.minute - 0.5
            a.axhline(0, color="#B2B2B8", lw=0.6, ls=(0, (3, 3)))
            a.fill_between(
                t,
                v.simultaneous_low,
                v.simultaneous_high,
                color="#6F6F7B",
                alpha=0.12,
                lw=0,
            )
            a.fill_between(t, v.ci_low, v.ci_high, color="#6F6F7B", alpha=0.25, lw=0)
            a.plot(t, v.mean_difference, color="#3E3E49", marker="o", ms=2.1, lw=1)
            a.set_xlim(0, 10)
            a.set_xticks([0, 2, 4, 6, 8, 10])
            a.set_ylim(-lim, lim)
            a.yaxis.set_major_locator(MaxNLocator(4))
            if col == 0:
                a.set_ylabel(pair.replace("-", " − ") + "\nRate difference", labelpad=3)
            if row == 2:
                a.set_xlabel("Time since start (min)", labelpad=4)
    save(f, "Supplementary_FigS4_minute_regional_contrasts")
    spline = read("timeline", "Supplementary_probability_spline_curves.csv")
    sens = read("timeline", "frozen_sensitivity_models.csv", "tables")
    merged = curves.merge(
        spline,
        on=["region", "maintenance_difference_sd"],
        suffixes=("_linear", "_spline"),
        validate="one_to_one",
    )
    merged["difference_percentage_points"] = 100 * (
        merged.probability_spline - merged.probability_linear
    )
    merged.to_csv(
        OUT / "source_data/main/S5a_spline_minus_linear.csv",
        index=False,
        encoding="utf-8-sig",
    )
    f = fig(130)
    legend(f)
    panel(f, 2, 11, "a")
    a = ax(f, 24, 23, 148, 32)
    for r in REG:
        v = merged[merged.region.eq(r)]
        a.plot(
            v.maintenance_difference_sd,
            v.difference_percentage_points,
            color=COL[r],
            ls=LS[r],
            lw=1.1,
        )
    a.axhline(0, color="#AAAAAF", lw=0.6)
    a.set_xlim(-2, 2)
    a.set_xticks([-2, -1, 0, 1, 2])
    a.set_ylim(-0.36, 0.36)
    a.set_yticks([-0.3, 0, 0.3])
    a.set_ylabel("Spline − linear probability\n(percentage points)", labelpad=3)
    a.set_xlabel("Early maintenance difference (common SD)", labelpad=4)
    panel(f, 2, 74, "b")
    a = ax(f, 28, 86, 48, 30)
    v = sens[sens.specification.eq("three_components_simultaneously")]
    for i, c in enumerate(COMPS):
        r = v[v.predictor.eq(c + "_diff_z")].iloc[0]
        a.errorbar(
            r.odds_ratio,
            2 - i,
            xerr=[[r.odds_ratio - r.or_low], [r.or_high - r.odds_ratio]],
            fmt="o",
            color="#454550",
            ms=3.5,
            capsize=2,
            lw=0.9,
        )
    a.set_yticks([2, 1, 0], ["Placement", "Clearing", "Purchases"])
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    a.axvline(1, color="#B2B2B8", ls=(0, (3, 3)), lw=0.6)
    a.set_xlim(0.99, 1.095)
    a.set_ylim(-0.5, 2.5)
    a.set_xticks([1, 1.04, 1.08])
    a.set_xlabel("Odds ratio per component SD", labelpad=4)
    panel(f, 91, 74, "c")
    a = ax(f, 139, 83, 36, 34)
    specs = [
        ("strict_nominal_10min_earlier_state_frame", "Strict 10-min cutoff"),
        ("window_0_15", "15-min window"),
        ("purchase_net_of_undo", "Net purchases"),
        ("first_legally_assigned_objective", "First assigned objective"),
        ("omit_sampling_time_tier", "Without sampling tier"),
    ]
    disjoint = [s for s in sens.specification.unique() if "disjoint" in s]
    assert len(disjoint) == 1
    specs.append((disjoint[0], "No repeated participants"))
    for i, (spec, name) in enumerate(specs):
        r = sens[sens.specification.eq(spec)].iloc[0]
        a.errorbar(
            r.odds_ratio,
            5 - i,
            xerr=[[r.odds_ratio - r.or_low], [r.or_high - r.odds_ratio]],
            fmt="o",
            color="#454550",
            ms=3.3,
            capsize=2,
            lw=0.9,
        )
    a.set_yticks(range(5, -1, -1), [x[1] for x in specs])
    a.tick_params(axis="y", length=0, labelsize=5.8)
    a.spines["left"].set_visible(False)
    a.set_xlim(1.06, 1.13)
    a.set_xticks([1.06, 1.09, 1.12])
    a.set_ylim(-0.55, 5.55)
    a.set_xlabel("Odds ratio per score SD", labelpad=4)
    save(f, "Supplementary_FigS5_functional_sensitivity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", choices=["relationships", "regional", "context", "timeline"]
    )
    args = parser.parse_args()
    style()
    original_save = save

    def save(f, stem):
        if stem.startswith("Fig"):
            original_save(f, stem)
        else:
            plt.close(f)

    for name, fn in [
        ("relationships", relationships),
        ("regional", regional),
        ("context", context),
        ("timeline", timeline),
    ]:
        if args.only is None or args.only == name:
            fn()
    old = OUT / "validation/layout_audit.json"
    previous = json.loads(old.read_text()) if old.exists() else {}
    previous.update(AUDIT)
    old.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
